"""Shot planning and resumable execution for real video backends."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import hashlib, json, time
import numpy as np


@dataclass
class ShotState:
    index: int
    start: int
    end: int
    status: str = "pending"
    attempts: int = 0
    cache_key: str = ""
    error: str | None = None
    output_path: str | None = None


def shot_ranges(frame_count: int, cuts: list[int]) -> list[tuple[int, int]]:
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    points = [0] + sorted({int(c) for c in cuts if 0 < int(c) < frame_count}) + [frame_count]
    return list(zip(points, points[1:]))


def _backend_signature(backend):
    keys=("model_id","mode","dtype","dtype_name","device","steps","guidance","offload","chunk_size","overlap","url","workflow","node_map")
    result={key:getattr(backend,key) for key in keys if hasattr(backend,key)}
    workflow=result.get("workflow")
    if workflow and Path(str(workflow)).is_file():
        path=Path(str(workflow));result["workflow_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def run_shots(frames: list[np.ndarray], cuts: list[int], masks: list[np.ndarray] | None,
              controls, references, backend, work_dir: str | Path, overrides: dict | None = None,
              retries: int = 1) -> tuple[list[np.ndarray], list[dict], list[np.ndarray] | None]:
    """Run every shot as an independent backend job with independent controls and cache.

    Retries only rerun the real backend operation. Failed shots propagate an exception
    with the shot index and keep earlier cache artifacts for resume.
    """
    from copy import deepcopy
    work = Path(work_dir); shot_dir = work / "shots"; shot_dir.mkdir(parents=True, exist_ok=True)
    all_outputs: list[np.ndarray] = []; effective_masks=[]; report=[]; overrides = overrides or {}; any_masks=False
    states=[]
    for idx,(start,end) in enumerate(shot_ranges(len(frames),cuts)):
        local = deepcopy(controls); local.references = deepcopy(references)
        local.scene_cuts=[]; local.frame_count=end-start
        local.settings["shot_index"]=idx
        override=overrides.get(str(idx),overrides.get(idx,{})) or {}
        for key in ("prompt","negative_prompt","identity_strength","motion_strength","reference_strength","preservation_strength"):
            if key in override:setattr(local,key,override[key])
        local.settings.update(override.get("settings",{}))
        if override.get("references"):
            from .controls import Reference
            local.references.references=[Reference(**r) if isinstance(r,dict) else Reference(path=str(r)) for r in override["references"]]
        selected_masks=masks[start:end] if masks is not None else None
        if selected_masks is not None:
            from .segmentation import save_masks
            local.object_masks=save_masks(selected_masks,shot_dir/f"shot_{idx:04d}_masks")
        shot_frames=frames[start:end]
        local.settings["reference_consistency"]={"scope":"cross_shot_shared","references":[
            {"sha256":hashlib.sha256(Path(r.path).read_bytes()).hexdigest(),"category":r.category,"weight":r.weight}
            for r in local.references.references]}
        from .video import write_video
        shot_source=shot_dir/f"shot_{idx:04d}_source.mp4"
        write_video(shot_frames,shot_source,controls.fps or 24.0,codec="h264")
        local.source_video=str(shot_source)
        if selected_masks is None and backend.name=="wan22-diffusers" and local.settings.get("wan_animate_preprocess",{}).get("wan_repo"):
            prep=local.settings["wan_animate_preprocess"]
            ref=next((r for r in local.references.references if r.category.upper() in {"CHARACTER","OBJECT","PRODUCT","OUTFIT"}),None)
            if ref is None:raise ValueError(f"Shot {idx} Wan Animate preprocessing needs a reference image")
            from .wan_preprocess import preprocess_wan_animate
            from .video import read_frames
            source_hash=hashlib.sha256(b"".join(frame.tobytes() for frame in shot_frames)).hexdigest()[:16]
            ref_hash=hashlib.sha256(Path(ref.path).read_bytes()).hexdigest()[:16]
            prep_dir=shot_dir/f"shot_{idx:04d}_wan_controls_{source_hash}_{ref_hash}"
            artifacts=preprocess_wan_animate(shot_source,ref.path,prep_dir,prep["wan_repo"],prep["checkpoint_dir"],int(prep.get("fps",-1)),tuple(prep.get("resolution_area",(1280,720))))
            local.settings.update({k:artifacts[k] for k in ("pose_video","face_video","background_video","mask_video")})
            mask_frames,_=read_frames(artifacts["mask_video"])
            selected_masks=[m[:,:,0].copy() for m in mask_frames]
            from .segmentation import save_masks
            local.object_masks=save_masks(selected_masks,shot_dir/f"shot_{idx:04d}_official_masks")
        digest=hashlib.sha256()
        for f in shot_frames:digest.update(f.tobytes())
        if selected_masks is not None:
            any_masks=True;effective_masks.extend(selected_masks)
            for m in selected_masks:digest.update(m.tobytes())
        cache_key=hashlib.sha256(json.dumps({"video":digest.hexdigest(),"prompt":local.prompt,"negative":local.negative_prompt,
            "references":[(hashlib.sha256(Path(r.path).read_bytes()).hexdigest(),r.category,r.weight) for r in local.references.references],
            "operation":local.operation.value,"settings":local.settings,"backend":backend.name,"backend_config":_backend_signature(backend)},sort_keys=True,default=str).encode()).hexdigest()
        cached=shot_dir/f"{cache_key}.npz"; done=shot_dir/f"{cache_key}.json"
        state=ShotState(idx,start,end,cache_key=cache_key)
        state_file=shot_dir/f"shot_{idx:04d}.json"
        if cached.is_file() and done.is_file():
            with np.load(cached) as archive: out=[x.copy() for x in archive["frames"]]
            if len(out)==len(shot_frames):
                state.status="cached";state.output_path=str(cached);state_file.write_text(json.dumps(state.__dict__,indent=2),encoding="utf-8");all_outputs.extend(out);states.append(state.__dict__);continue
        state.status="running"; state_file.write_text(json.dumps(state.__dict__,indent=2),encoding="utf-8")
        last=None
        for attempt in range(max(0,int(retries))+1):
            state.attempts=attempt+1
            try:
                out=backend.generate(local,shot_frames,shot_dir/f"shot_{idx:04d}_run_{attempt+1}")
                if len(out)!=len(shot_frames):raise RuntimeError(f"backend returned {len(out)} frames, expected {len(shot_frames)}")
                if any(f.shape!=shot_frames[0].shape for f in out):raise RuntimeError("backend output frame geometry differs within the shot")
                np.savez_compressed(cached,frames=np.stack(out))
                done.write_text(json.dumps({"status":"complete","shot":idx,"range":[start,end],"backend":backend.name},indent=2),encoding="utf-8")
                state.status="complete"; state.output_path=str(cached); state.error=None
                state_file.write_text(json.dumps(state.__dict__,indent=2),encoding="utf-8")
                all_outputs.extend(out); break
            except Exception as exc:
                last=exc; state.error=f"{type(exc).__name__}: {exc}"
                state.status="retrying" if attempt < retries else "failed"
                state_file.write_text(json.dumps(state.__dict__,indent=2),encoding="utf-8")
                if attempt < retries:time.sleep(min(.25*(2**attempt),2))
        states.append(state.__dict__)
        if last is not None and state.status=="failed":
            (work/"shots.json").write_text(json.dumps(states,indent=2),encoding="utf-8")
            raise RuntimeError(f"Shot {idx} (frames {start}:{end}) failed after {state.attempts} attempts: {last}") from last
    (work/"shots.json").write_text(json.dumps(states,indent=2),encoding="utf-8")
    return all_outputs, states, (effective_masks if any_masks else None)
