"""High-level transformation orchestration with explicit model/backend capability checks."""
from __future__ import annotations
from pathlib import Path
import json
import hashlib
import numpy as np
from PIL import Image
from .analysis import scene_cuts, estimate_camera
from .backends import create_backend
from .compositing import composite
from .cache import StageCache
from .config import load_config
from .controls import ControlPackage, Operation, Reference, ReferencePackage
from .motion import chunk_ranges
from .prompt import compile_prompt
from .temporal import stabilize_generated
from .video import probe_video, read_frames, write_video
from .routing import route_config

def analyze_video(path: str|Path) -> tuple[ControlPackage,list[np.ndarray]]:
    frames,info=read_frames(path)
    return ControlPackage(source_video=info.path,fps=info.fps,width=info.width,height=info.height,frame_count=info.frame_count,duration=info.duration,audio_metadata=info.audio_metadata or {},scene_cuts=scene_cuts(frames),camera_motion=estimate_camera(frames)),frames

def _read_mask(path: str|Path, expected: int, shape: tuple[int,int]) -> list[np.ndarray]:
    if isinstance(path,(list,tuple)):
        files=[Path(item) for item in path]
        if len(files)==1 and files[0].suffix.lower() in {".mp4",".mov",".mkv",".webm"}:
            video_frames,_=read_frames(files[0]);files=None
    else:
        p=Path(path)
        if p.is_dir():files=sorted(p.glob("*.png"))
        elif p.suffix.lower() in {".mp4",".mov",".mkv",".webm"}:
            video_frames,_=read_frames(p); files=None
        else:files=[p]
    if files is None:
        masks=[np.asarray(Image.fromarray(frame).convert("L")) for frame in video_frames]
    else:
        if not files: raise FileNotFoundError(f"No mask PNGs at {path}")
        if len(files)==1 and expected>1: files=files*expected
        if len(files)!=expected: raise ValueError(f"Mask sequence has {len(files)} frames but source has {expected}")
        masks=[]
        for f in files:
            m=np.asarray(Image.open(f).convert("L"))
            if m.shape!=shape: raise ValueError(f"Mask {f} dimensions {m.shape[::-1]} do not match source {shape[::-1]}")
            masks.append(m)
    if len(masks)!=expected: raise ValueError(f"Mask sequence has {len(masks)} frames but source has {expected}")
    if any(m.shape!=shape for m in masks):raise ValueError(f"Mask video dimensions must match source {shape[::-1]}")
    return masks

def transform_video(source: str|Path, output: str|Path, operation: str|Operation, prompt: str,
                    references: list[str]|None=None, edit_mask: str|Path|None=None,
                    config_path: str|Path|None="config.yaml", negative_prompt: str="", seed: int=0,
                    codec: str="h264", preserve_audio: bool=True, **settings) -> Path:
    config=load_config(config_path); controls,frames=analyze_video(source)
    if settings.get("backend_override"):
        config.setdefault("backend",{})["type"]=settings["backend_override"]
    controls.operation=Operation(operation); controls.prompt=prompt; controls.negative_prompt=negative_prompt
    controls.settings.update(settings); controls.settings["seed"]=int(seed)
    for path in references or []:
        from .references import ReferenceManager
        # Generic reference classification is made explicit in settings or defaults to category OBJECT.
        manager=ReferenceManager(); manager.add(path,settings.get("reference_category","OBJECT"),settings.get("reference_weight",1.0)); controls.references.references.extend(manager.package.references)
    if edit_mask:
        masks=_read_mask(edit_mask,len(frames),(controls.height,controls.width))
        is_video_mask=(isinstance(edit_mask,(list,tuple)) and len(edit_mask)==1 and Path(edit_mask[0]).suffix.lower() in {".mp4",".mov",".mkv",".webm"}) or (not isinstance(edit_mask,(list,tuple)) and Path(edit_mask).suffix.lower() in {".mp4",".mov",".mkv",".webm"})
        if is_video_mask:
            from .segmentation import save_masks
            controls.object_masks=[str(Path(save_masks(masks,Path(output).parent/(Path(output).stem+".work")/"input_mask_frames")[0]).parent)]
        else:controls.object_masks=[str(p) for p in edit_mask] if isinstance(edit_mask,(list,tuple)) else [str(edit_mask)]
    else: masks=None
    controls.prompt=compile_prompt(controls)
    if config["processing"].get("chunk_frames",81)<1: raise ValueError("processing.chunk_frames must be positive")
    config=route_config(config,controls.operation,controls)
    backend=create_backend(config)
    controls.settings.update(config.get("backend",{}).get("wan_animate_preprocess",{}))
    controls.settings["wan_animate_preprocess"]=config.get("backend",{}).get("wan_animate_preprocess",{})
    if controls.references.references:
        controls.settings["reference_path"]=controls.references.references[0].path
    out=Path(output).resolve(); work=out.parent/(out.stem+".work"); work.mkdir(parents=True,exist_ok=True)
    if edit_mask is None and controls.operation in {Operation.ENVIRONMENT_RECAST,Operation.FULL_RECAST,Operation.STYLE_RECAST}:
        masks=[np.full((controls.height,controls.width),255,np.uint8) for _ in frames]
        from .segmentation import save_masks
        controls.object_masks=save_masks(masks,work/"edit_masks")
    elif edit_mask is None:
        selection={**config.get("selection",{}),**settings.get("selection",{})}
        text=selection.get("text_prompt",settings.get("selection_prompt"))
        if text:
            from .selection import select_and_track
            cuts=[0]+controls.scene_cuts+[len(frames)]
            masks=[np.zeros((controls.height,controls.width),np.uint8) for _ in frames]
            shot_prompts=controls.settings.get("shots",{})
            for shot_index,(start,end) in enumerate(zip(cuts,cuts[1:])):
                override=shot_prompts.get(str(shot_index),{}) if isinstance(shot_prompts,dict) else {}
                target=override.get("selection_prompt",text)
                current,metadata=select_and_track(frames[start:end],target,selection.get("detector_model"),selection.get("sam2_config"),selection.get("sam2_checkpoint"),int(selection.get("frame_index",0)),selection.get("device","cuda"),float(selection.get("box_threshold",.30)),float(selection.get("text_threshold",.25)),int(selection.get("detection_index",0)))
                masks[start:end]=current
                controls.settings.setdefault("selection_results",[]).append({"shot":shot_index,**metadata})
            from .segmentation import save_masks
            controls.object_masks=save_masks(masks,work/"auto_masks")
    cache_cfg=config.get("cache",{}); stage_cache=StageCache(cache_cfg.get("directory",".genjutsu-cache"),bool(cache_cfg.get("enabled",True)))
    mask_key=[]
    for raw_path in controls.object_masks:
        mask_path=Path(raw_path).resolve()
        if mask_path.is_dir():
            mask_key.extend((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(mask_path.glob("*.png")))
        elif mask_path.exists(): mask_key.append((str(mask_path),mask_path.stat().st_size,mask_path.stat().st_mtime_ns))
        else: mask_key.append(str(mask_path))
    fingerprint={"source":str(Path(source).resolve()),"size":Path(source).stat().st_size,"mtime":Path(source).stat().st_mtime_ns,"operation":controls.operation.value,"prompt":controls.prompt,"negative_prompt":controls.negative_prompt,"references":[(r.path,hashlib.sha256(Path(r.path).read_bytes()).hexdigest(),r.category,r.weight) for r in controls.references.references],"masks":mask_key,"backend":config.get("backend",{}),"settings":controls.settings,"shots":controls.scene_cuts,"seed":seed}
    cache_dir=stage_cache.path("generated",stage_cache.key("generated",fingerprint)); cached_file=cache_dir/"frames.npz"
    cache_hit=False
    if stage_cache.enabled and cached_file.is_file() and (cache_dir/"complete.json").is_file():
        with np.load(cached_file) as cached:
            generated=[f.copy() for f in cached["frames"]]
            if "masks" in cached.files:masks=[m.copy() for m in cached["masks"]]
            cache_hit=(masks is not None or controls.operation in {Operation.ENVIRONMENT_RECAST,Operation.FULL_RECAST,Operation.STYLE_RECAST})
    if not cache_hit:
        from .shots import run_shots
        generated,shot_states,shot_masks=run_shots(frames,controls.scene_cuts,masks,controls,controls.references,backend,work,
            controls.settings.get("shots",{}),int(config.get("processing",{}).get("shot_retries",1)))
        if masks is None and shot_masks is not None:masks=shot_masks
        if len(generated)!=len(frames): raise RuntimeError(f"Backend returned {len(generated)} frames for a {len(frames)}-frame source. Refusing to silently duplicate/drop frames.")
        if any(f.shape[:2]!=(controls.height,controls.width) for f in generated): raise RuntimeError("Backend frame dimensions differ from source; configure explicit resize/output dimensions before compositing.")
        if stage_cache.enabled:
            cache_dir.mkdir(parents=True,exist_ok=True)
            cached_values={"frames":np.stack(generated)}
            if masks is not None:cached_values["masks"]=np.stack(masks)
            np.savez_compressed(cached_file,**cached_values)
            (cache_dir/"complete.json").write_text(json.dumps({"stage":"generated","backend":backend.name,"frames":len(generated)},indent=2),encoding="utf-8")
    if len(generated)!=len(frames): raise RuntimeError(f"Backend returned {len(generated)} frames for a {len(frames)}-frame source. Refusing to silently duplicate/drop frames.")
    if any(f.shape[:2]!=(controls.height,controls.width) for f in generated): raise RuntimeError("Backend frame dimensions differ from source; configure explicit resize/output dimensions before compositing.")
    if config.get("temporal",{}).get("enabled"): generated=stabilize_generated(generated,float(config["temporal"].get("strength",.2)),controls.scene_cuts)
    if config.get("compositing",{}).get("enabled"):
        if masks is None:
            if controls.operation in {Operation.ENVIRONMENT_RECAST,Operation.FULL_RECAST,Operation.STYLE_RECAST}:
                masks=[np.full((controls.height,controls.width),255,np.uint8) for _ in frames]
            else: raise ValueError("Selective replacement needs an edit mask sequence; provide --mask. Full-scene recast operations can omit it.")
        preserve=None
        if controls.preservation_masks:
            preserve=_read_mask(controls.preservation_masks[0],len(frames),(controls.height,controls.width))
        occ=config.get("occlusion",{})
        if occ.get("enabled"):
            checkpoint=config.get("depth",{}).get("checkpoint")
            foreground_path=occ.get("foreground_mask")
            if not checkpoint or not foreground_path:
                raise ValueError("Depth-ordered occlusion handling requires a local depth.checkpoint and occlusion.foreground_mask sequence")
            from .analysis import depth_sequence
            from .occlusion import visibility_mask
            depth_maps=depth_sequence(frames,"depth_anything",checkpoint)
            foreground_masks=_read_mask(foreground_path,len(frames),(controls.height,controls.width))
            visible_masks=[]
            for mask,depth,foreground in zip(masks,depth_maps,foreground_masks):
                selected=depth[mask>0]
                target_depth=float(np.median(selected)) if selected.size else float(np.median(depth))
                target_depth_map=np.full(depth.shape,target_depth,dtype=np.float32)
                visible_masks.append(visibility_mask(mask,target_depth_map,depth,foreground,near_is_high=True))
            masks=visible_masks
        generated=[composite(src,gen,mask,preserve_mask=(preserve[i] if preserve else None),feather_px=int(config["compositing"].get("feather_px",8)),color_match_enabled=bool(config["compositing"].get("color_match",False))) for i,(src,gen,mask) in enumerate(zip(frames,generated,masks))]
    identity_result=None
    if config.get("identity",{}).get("enabled"):
        ref=next((r for r in controls.references.references if r.category.upper()=="CHARACTER"),None)
        if ref is None:raise ValueError("Identity evaluation requires a CHARACTER reference image")
        from .identity import LocalInsightFace
        evaluator=LocalInsightFace(config["identity"].get("model_name","buffalo_l"),config["identity"].get("root","models/insightface"))
        identity_result=evaluator.evaluate_reference(ref.path,generated,float(config["identity"].get("threshold",.55)))
        (work/"identity_evaluation.json").write_text(json.dumps(identity_result,indent=2),encoding="utf-8")
    result=write_video(generated,out,controls.fps,audio_from=source if preserve_audio else None,codec=codec)
    controls.save(work/"controls.json")
    (work/"run.json").write_text(json.dumps({"source":str(Path(source).resolve()),"output":str(result),"backend":backend.name,"frames":len(generated),"fps":controls.fps,"shots":json.loads((work/"shots.json").read_text(encoding="utf-8")) if (work/"shots.json").is_file() else [],"identity_evaluation":identity_result},indent=2),encoding="utf-8")
    return result
