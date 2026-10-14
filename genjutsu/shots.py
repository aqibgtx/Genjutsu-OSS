"""Shot-local controls and resumable real-backend execution with disk outputs."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import json
import time
import numpy as np

from .cache import backend_signature, file_digest, generation_key
from .frame_store import FrameStore


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


def shot_ranges(frame_count, cuts):
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if frame_count == 0:
        return []
    points = [0] + sorted({int(c) for c in cuts if 0 < int(c) < frame_count}) + [frame_count]
    return list(zip(points, points[1:]))


def _slice_npy(path,start,end,count,directory):
    """Copy a contiguous NPY temporal range in bounded blocks, preserving dtype."""
    path=Path(path)
    with path.open("rb") as source:
        version=np.lib.format.read_magic(source)
        readers={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}
        if version not in readers:raise ValueError(f"Unsupported temporal NPY header {version}")
        shape,fortran,dtype=readers[version](source)
        if not shape or shape[0] not in {count,count-1}:return str(path)
        if fortran or dtype.hasobject:raise ValueError("Temporal NPY controls must be C-contiguous numeric arrays")
        if directory is None:raise ValueError("Slicing file-backed temporal controls requires artifact_dir")
        stop=end if shape[0]==count else max(start,end-1)
        stride=int(np.prod(shape[1:]))*dtype.itemsize
        offset=source.tell()
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        target=directory/f"{path.stem}_{start}_{stop}.npy"
        source.seek(offset+start*stride)
        with target.open("wb") as destination:
            np.lib.format.write_array_header_2_0(destination,{"descr":np.lib.format.dtype_to_descr(dtype),"fortran_order":False,"shape":(stop-start,*shape[1:])})
            remaining=(stop-start)*stride
            while remaining:
                block=source.read(min(1024*1024,remaining))
                if not block:raise ValueError("Truncated temporal NPY control")
                destination.write(block);remaining-=len(block)
        return str(target)


def _temporal_slice(value, start, end, count, time_offset, artifact_dir=None):
    """Slice per-frame/per-transition arrays and explicitly indexed track records."""
    if isinstance(value, dict):
        if value and all(str(k).isdigit() for k in value):
            return {str(int(k)-start): deepcopy(v) for k, v in value.items() if start <= int(k) < end}
        return {k: _temporal_slice(v, start, end, count, time_offset,Path(artifact_dir)/str(k) if artifact_dir else None) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        frame_key = next((key for key in ("frame_index", "frame") if value and isinstance(value[0], dict) and key in value[0]), None)
        if frame_key:
            result = []
            for record in value:
                index = int(record[frame_key])
                if start <= index < end:
                    row = deepcopy(record)
                    row["source_frame_index"] = index
                    row[frame_key] = index-start
                    if "timestamp" in row:
                        row["timestamp"] -= time_offset
                    result.append(row)
            return result
        if len(value) == count:
            return deepcopy(value[start:end])
        if len(value) == count-1:
            return deepcopy(value[start:max(start, end-1)])
        return deepcopy(value)
    if isinstance(value,(str,Path)) and Path(value).suffix.lower()==".npy" and Path(value).is_file():
        return _slice_npy(value,start,end,count,artifact_dir)
    return deepcopy(value)


def _slice_paths(paths, start, end, count):
    if len(paths) == 1 and Path(paths[0]).is_dir():
        files = sorted(Path(paths[0]).glob("*.png"))
        if len(files) != count:
            raise ValueError(f"Temporal control directory has {len(files)} frames, expected {count}")
        return [str(p) for p in files[start:end]]
    if len(paths) == count:
        return paths[start:end]
    if len(paths) <= 1:
        return deepcopy(paths)
    raise ValueError(f"Temporal control sequence has {len(paths)} frames, expected {count}")


def slice_controls(controls, start, end, *, artifact_dir=None):
    """Local frame indices/PTS start at zero; source indices/time remain explicit."""
    count = controls.frame_count
    if not 0 <= start < end <= count:
        raise ValueError("Shot range must fall within the source frame count")
    local = deepcopy(controls)
    tb = Fraction(controls.time_base)
    rate = Fraction(controls.fps_rational) if Fraction(controls.fps_rational) > 0 else Fraction(str(controls.fps))
    pts = controls.frame_pts or [round(i/rate/tb) for i in range(count)]
    durations = controls.frame_durations or [round(1/rate/tb)]*count
    origin = pts[start]
    offset = float((origin-pts[0])*tb)
    local.frame_count = end-start
    local.scene_cuts = []
    local.frame_pts = [p-origin for p in pts[start:end]]
    local.frame_durations = durations[start:end]
    local.timestamps = [float(p*tb) for p in local.frame_pts]
    local.frame_indices = list(range(end-start))
    local.source_frame_indices = (controls.source_frame_indices or list(range(count)))[start:end]
    local.source_start_time = controls.source_start_time+offset
    local.duration = float((local.frame_pts[-1]+durations[end-1])*tb)
    gaps = [b-a for a, b in zip(local.frame_pts, local.frame_pts[1:])]
    local.is_vfr = bool(gaps and max(gaps)-min(gaps) > 1)
    if local.is_vfr:
        local.fps_rational = str(Fraction(end-start)/((local.frame_pts[-1]+durations[end-1])*tb))
        local.fps = float(Fraction(local.fps_rational))
    else:
        local.fps_rational = str(rate)
        local.fps = float(rate)
    local.audio_metadata = {**local.audio_metadata, "source_video_offset": offset, "shot_duration": local.duration}
    for field in ("subject_masks", "object_masks", "preservation_masks", "depth_sequence"):
        setattr(local, field, _slice_paths(getattr(controls, field), start, end, count))
    local.pose_sequence = deepcopy(controls.pose_sequence[start:end]) if controls.pose_sequence else []
    if controls.pose_sequence and len(controls.pose_sequence) != count:
        raise ValueError("Pose sequence frame count differs from source")
    for field in ("tracking_data", "optical_flow", "camera_motion"):
        setattr(local, field, _temporal_slice(getattr(controls, field), start, end, count, offset,Path(artifact_dir)/field if artifact_dir else None))
    local.settings["source_range"] = [start, end]
    local.settings["source_time_range"] = [local.source_start_time, local.source_start_time+local.duration]
    return local


def _backend_signature(backend):
    return backend_signature(backend)


def run_shots(frames, cuts, masks, controls, references, backend, work_dir, overrides=None, retries=1,
              *, cache_dir=None, cache_enabled=True, prepare_shot=None):
    """Run independent jobs; aggregate outputs and masks on disk, never in RAM.

    prepare_shot performs configured automatic selection after applying each
    shot's overrides. Model cache reuse requires a resolved checkpoint identity.
    """
    from .video import write_video, read_frames
    from .segmentation import save_masks
    work = Path(work_dir)
    shot_dir = work/"shots"
    shot_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cache_dir) if cache_dir else shot_dir/"cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    all_outputs, effective_masks = FrameStore(), FrameStore()
    states, any_masks, overrides = [], False, overrides or {}
    signature = backend_signature(backend)
    cache_enabled = cache_enabled and signature["cache_safe"]
    for idx, (start, end) in enumerate(shot_ranges(len(frames), cuts)):
        local = slice_controls(controls, start, end,artifact_dir=shot_dir/f"shot_{idx:04d}_temporal")
        # Only this shot's effective settings belong in its cache identity.
        local.settings.pop("shots",None)
        local.references = deepcopy(references)
        local.settings["shot_index"] = idx
        override = overrides.get(str(idx), overrides.get(idx, {})) or {}
        for key in ("prompt", "negative_prompt", "identity_strength", "motion_strength", "reference_strength", "preservation_strength"):
            if key in override:
                setattr(local, key, override[key])
        local.settings.update(override.get("settings", {}))
        if "selection_prompt" in override:
            local.settings["selection_prompt"] = override["selection_prompt"]
        if "references" in override:
            from .controls import Reference
            local.references.references = [Reference(**r) if isinstance(r, dict) else Reference(path=str(r)) for r in override["references"]]
        if local.references.references:
            local.settings["reference_path"] = local.references.references[0].path
        else:
            local.settings.pop("reference_path", None)
        shot_frames = frames[start:end]
        selected_masks = masks[start:end] if masks is not None else None
        if prepare_shot:
            selected_masks = prepare_shot(local, shot_frames, selected_masks, shot_dir/f"shot_{idx:04d}")
        if selected_masks is not None:
            if len(selected_masks)!=len(shot_frames):raise ValueError("Shot mask count differs from source")
            local.object_masks = save_masks(selected_masks, shot_dir/f"shot_{idx:04d}_masks")
        local.settings["reference_consistency"] = {"scope": "cross_shot_shared", "references": [
            {"sha256": file_digest(r.path), "category": r.category, "weight": r.weight} for r in local.references.references]}
        shot_source = shot_dir/f"shot_{idx:04d}_source.mp4"
        write_video(shot_frames, shot_source, local.fps_rational if Fraction(local.fps_rational) > 0 else local.fps,
                    timestamps=local.timestamps, frame_durations=[float(d*Fraction(local.time_base)) for d in local.frame_durations], time_base=local.time_base)
        local.source_video = str(shot_source)
        # Explicitly supplied whole-clip conditioning is sliced before inference.
        for key in ("pose_video", "face_video", "background_video", "mask_video"):
            path = local.settings.get(key) or getattr(backend, key, None)
            if not path:
                continue
            condition, info = read_frames(path, disk_backed=True)
            if len(condition) == len(frames):
                condition = condition[start:end]
            elif len(condition) != end-start:
                raise ValueError(f"{key} must cover the full source or this shot")
            target = shot_dir/f"shot_{idx:04d}_{key}.mp4"
            write_video(condition, target, local.fps_rational if Fraction(local.fps_rational)>0 else local.fps,
                        timestamps=local.timestamps, frame_durations=[float(d*Fraction(local.time_base)) for d in local.frame_durations], time_base=local.time_base)
            local.settings[key] = str(target)
        if backend.name == "wan22-diffusers" and getattr(backend, "mode", "animate") == "animate":
            if local.is_vfr:
                raise ValueError("Wan Animate preprocessing requires CFR input. Explicitly retime the source to CFR before choosing Animate; VACE preserves source VFR timestamps.")
            prep = local.settings.get("wan_animate_preprocess", {})
            if not all(local.settings.get(key) for key in ("pose_video", "face_video", "background_video", "mask_video")) and prep.get("wan_repo"):
                from .wan_preprocess import preprocess_wan_animate
                ref = next((r for r in local.references.references if r.category.upper() in {"CHARACTER", "OBJECT", "PRODUCT", "OUTFIT"}), None)
                if ref is None:
                    raise ValueError(f"Shot {idx} Wan Animate preprocessing needs a reference image")
                artifacts = preprocess_wan_animate(shot_source, ref.path, shot_dir/f"shot_{idx:04d}_wan_controls", prep["wan_repo"], prep["checkpoint_dir"], int(prep.get("fps", -1)), tuple(prep.get("resolution_area", (1280, 720))))
                local.settings.update({key: artifacts[key] for key in ("pose_video", "face_video", "background_video", "mask_video")})
            if selected_masks is None and local.settings.get("mask_video"):
                from .pipeline import _read_mask
                selected_masks = _read_mask(local.settings["mask_video"], end-start, (local.height, local.width))
                local.object_masks = save_masks(selected_masks, shot_dir/f"shot_{idx:04d}_official_masks")
        if selected_masks is not None:
            any_masks = True
            effective_masks.extend(selected_masks)
        else:
            effective_masks.extend(np.full((local.height, local.width), 255, np.uint8) for _ in shot_frames)
        key = generation_key(local, shot_frames, selected_masks, signature)
        cached = cache_root/key
        state = ShotState(idx, start, end, cache_key=key)
        state_file = shot_dir/f"shot_{idx:04d}.json"
        local.settings["generation_signature"] = signature
        local.save(shot_dir/f"shot_{idx:04d}_controls.json")

        def persist():
            state_file.write_text(json.dumps(state.__dict__, indent=2), encoding="utf-8")
            (work/"shots.json").write_text(json.dumps(states+[state.__dict__], indent=2), encoding="utf-8")

        if cache_enabled and (cached/"complete.json").is_file():
            try:
                out = FrameStore.open(cached)
                if len(out) != end-start or out.shape != shot_frames[0].shape:
                    raise ValueError("Cached shot geometry/count mismatch")
                state.status, state.output_path = "cached", str(cached)
                all_outputs.extend(out)
                persist()
                states.append(state.__dict__)
                continue
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
        state.status = "running"
        persist()
        last = None
        for attempt in range(max(0, int(retries))+1):
            state.attempts = attempt+1
            try:
                out = backend.generate(local, shot_frames, shot_dir/f"shot_{idx:04d}_run_{attempt+1}")
                if len(out) != len(shot_frames):
                    raise RuntimeError(f"backend returned {len(out)} frames, expected {len(shot_frames)}")
                if any(f.shape != shot_frames[0].shape or f.dtype != np.uint8 for f in out):
                    raise RuntimeError("Backend output must match source RGB uint8 geometry")
                if cache_enabled:
                    (cached/"complete.json").unlink(missing_ok=True)
                    stored = FrameStore.from_frames(out, cached)
                    (cached/"complete.json").write_text(json.dumps({"key": key, "backend": backend.name}), encoding="utf-8")
                    state.output_path = str(stored.directory)
                state.status, state.error = "complete", None
                all_outputs.extend(out)
                persist()
                break
            except Exception as exc:
                last = exc
                state.error = f"{type(exc).__name__}: {exc}"
                state.status = "retrying" if attempt < retries else "failed"
                persist()
                if attempt < retries:
                    time.sleep(min(.25*(2**attempt), 2))
        states.append(state.__dict__)
        if state.status == "failed":
            raise RuntimeError(f"Shot {idx} (frames {start}:{end}) failed after {state.attempts} attempts: {last}") from last
    (work/"shots.json").write_text(json.dumps(states, indent=2), encoding="utf-8")
    return all_outputs.finish(), states, (effective_masks.finish() if any_masks else None)
