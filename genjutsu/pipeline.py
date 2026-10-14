"""High-level transformation orchestration with explicit model/backend capability checks."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from PIL import Image
from .analysis import scene_cuts, estimate_camera
from .backends import create_backend
from .compositing import composite
from .config import load_config
from .controls import ControlPackage, Operation
from .prompt import compile_prompt
from .temporal import stabilize_generated
from .video import probe_video, read_frames, write_video, timing_kwargs
from .frame_store import FrameStore, ImageSequence
from .routing import route_config

def analyze_video(path: str|Path, *, frame_directory=None):
    frames,info=read_frames(path,disk_backed=True,directory=frame_directory)
    cuts=scene_cuts(frames)
    camera=estimate_camera(frames)
    # A cross-cut affine estimate is meaningless; retain its position for
    # indexing but explicitly mark it invalid in the whole-clip package.
    for cut in cuts:
        camera["trajectory"][cut-1]={"dx":0.,"dy":0.,"rotation":0.,"zoom":1.,"valid":False,"scene_cut":True}
    return ControlPackage(source_video=info.path,fps=info.fps,width=info.width,height=info.height,frame_count=len(frames),duration=info.duration,audio_metadata=info.audio_metadata or {},scene_cuts=cuts,camera_motion=camera,
        fps_rational=info.fps_rational,time_base=info.time_base,frame_pts=info.frame_pts,frame_durations=info.frame_durations,timestamps=info.timestamps,
        frame_indices=list(range(len(frames))),source_frame_indices=list(range(len(frames))),source_start_time=info.start_time,is_vfr=info.is_vfr),frames

def _read_mask(path: str|Path, expected: int, shape: tuple[int,int]) -> list[np.ndarray]:
    from .ui import VIDEO_SUFFIXES
    from .video import iter_frames
    def video_masks(p):
        info=probe_video(p)
        if info.frame_count!=expected:raise ValueError(f"Mask sequence has {info.frame_count} frames but source has {expected}")
        if (info.height,info.width)!=shape:raise ValueError(f"Mask video dimensions must match source {shape[::-1]}")
        return FrameStore.from_frames(np.asarray(Image.fromarray(frame).convert("L")) for frame in iter_frames(p))
    if isinstance(path,(list,tuple)):
        files=[Path(item) for item in path]
        if len(files)==1 and files[0].suffix.lower() in VIDEO_SUFFIXES:
            return video_masks(files[0])
        if any(p.suffix.lower() in VIDEO_SUFFIXES for p in files):
            raise ValueError("Provide one mask video or a PNG sequence, not mixed mask uploads")
    else:
        p=Path(path)
        if p.is_dir():files=sorted(p.glob("*.png"))
        elif p.suffix.lower() in VIDEO_SUFFIXES:
            return video_masks(p)
        else:files=[p]
    if not files: raise FileNotFoundError(f"No mask PNGs at {path}")
    return ImageSequence(files,expected,shape)

def transform_video(source: str|Path, output: str|Path, operation: str|Operation, prompt: str,
                    references: list[str]|None=None, edit_mask: str|Path|None=None,
                    config_path: str|Path|None="config.yaml", negative_prompt: str="", seed: int=0,
                    codec: str="h264", preserve_audio: bool=True, **settings) -> Path:
    config=load_config(config_path)
    override_config=settings.pop("runtime_config",{})
    for key,value in override_config.items():
        config[key]={**config.get(key,{}),**value} if isinstance(value,dict) else value
    controls,frames=analyze_video(source)
    if settings.get("backend_override"):
        config.setdefault("backend",{})["type"]=settings["backend_override"]
    controls.operation=Operation(operation); controls.prompt=prompt; controls.negative_prompt=negative_prompt
    controls.settings.update(settings); controls.settings["seed"]=int(seed)
    default_category={Operation.CHARACTER_SWAP:"CHARACTER",Operation.MOTION_TRANSFER:"CHARACTER",
        Operation.PRODUCT_SWAP:"PRODUCT",Operation.OUTFIT_SWAP:"OUTFIT",Operation.ENVIRONMENT_RECAST:"LOCATION",Operation.STYLE_RECAST:"STYLE"}.get(controls.operation,"OBJECT")
    for path in references or []:
        from .references import ReferenceManager
        # Generic reference classification is made explicit in settings or defaults to category OBJECT.
        manager=ReferenceManager()
        row=path if isinstance(path,dict) else {"path":str(path)}
        manager.add(row["path"],row.get("category",settings.get("reference_category",default_category)),row.get("weight",settings.get("reference_weight",1.0)),row.get("metadata"))
        controls.references.references.extend(manager.package.references)
    masks=_read_mask(edit_mask,len(frames),(controls.height,controls.width)) if edit_mask else None
    controls.prompt=compile_prompt(controls)
    if config["processing"].get("chunk_frames",81)<1: raise ValueError("processing.chunk_frames must be positive")
    config=route_config(config,controls.operation,controls)
    backend=create_backend(config)
    controls.settings["wan_animate_preprocess"]=config.get("backend",{}).get("wan_animate_preprocess",{})
    out=Path(output).resolve(); work=out.parent/(out.stem+".work"); work.mkdir(parents=True,exist_ok=True)
    selection={**config.get("selection",{}),**settings.get("selection",{})}
    if "selection_prompt" not in controls.settings:
        controls.settings["selection_prompt"]=selection.get("text_prompt")

    def prepare_shot(local,shot_frames,selected_masks,directory):
        if selected_masks is not None:return selected_masks
        if local.operation in {Operation.FULL_RECAST,Operation.STYLE_RECAST}:
            return FrameStore.from_frames(np.full((local.height,local.width),255,np.uint8) for _ in range(len(shot_frames)))
        if backend.name=="wan22-diffusers" and getattr(backend,"mode",None)=="animate":
            if local.settings.get('selection_prompt','').strip().lower()=='face':
                from .selection import select_face_and_track
                current,metadata=select_face_and_track(shot_frames,selection.get('sam2_config'),selection.get('sam2_checkpoint'),
                    config.get('identity',{}).get('root','models/insightface'),device=selection.get('device','cuda'),chunk_frames=int(selection.get('chunk_frames',32)))
                local.settings['selection_results']=metadata
                return current
            # The official replacement preprocessor performs person detection,
            # SAM2 tracking and all four model-native conditioning videos.
            return None
        text=local.settings.get("selection_prompt")
        if text:
            from .selection import select_and_track
            current,metadata=select_and_track(shot_frames,text,selection.get("detector_model"),selection.get("sam2_config"),selection.get("sam2_checkpoint"),int(selection.get("frame_index",0)),selection.get("device","cuda"),float(selection.get("box_threshold",.30)),float(selection.get("text_threshold",.25)),int(selection.get("detection_index",0)),int(selection.get("chunk_frames",32)))
            local.settings["selection_results"]=metadata
            if local.operation==Operation.ENVIRONMENT_RECAST:
                # Target selection identifies the subject to PRESERVE here.
                return FrameStore.from_frames(255-mask for mask in current)
            return current
        if local.operation==Operation.ENVIRONMENT_RECAST:
            raise ValueError("Environment recast requires subject selection or an explicit background edit mask; it cannot silently regenerate the subject")
        return None

    from .shots import run_shots
    cache_cfg=config.get("cache",{})
    identity_cfg=config.get("identity",{})
    evaluator=None
    if identity_cfg.get("enabled") or identity_cfg.get('evaluate_only'):
        from .identity import LocalInsightFace
        evaluator=LocalInsightFace(identity_cfg.get("model_name","buffalo_l"),identity_cfg.get("root","models/insightface"))
        from .cache import content_identity
        controls.settings['identity_acceptance']={**identity_cfg,'model_contents':content_identity(evaluator.root/'models'/identity_cfg.get('model_name','buffalo_l'))}
    def accept_shot(local,source_frames,candidate,local_masks,directory):
        from .identity import correct_identity
        accepted,_=correct_identity(backend,local,source_frames,candidate,directory,evaluator,local_masks,
            float(identity_cfg.get("threshold",.55)),int(identity_cfg.get("max_correction_attempts",2)))
        return accepted
    generated,shot_states,shot_masks=run_shots(frames,controls.scene_cuts,masks,controls,controls.references,backend,work,
        controls.settings.get("shots",{}),int(config.get("processing",{}).get("shot_retries",1)),
        cache_dir=Path(cache_cfg.get("directory",".genjutsu-cache"))/"generated",
        cache_enabled=bool(cache_cfg.get("enabled",True)),prepare_shot=prepare_shot,
        validate_output=accept_shot if identity_cfg.get('enabled') else None)
    if masks is None and shot_masks is not None:masks=shot_masks
    if len(generated)!=len(frames):raise RuntimeError("Backend frame count differs from source")
    if any(f.shape[:2]!=(controls.height,controls.width) for f in generated):raise RuntimeError("Backend dimensions differ from source")
    if config.get("temporal",{}).get("enabled"): generated=stabilize_generated(generated,float(config["temporal"].get("strength",.2)),controls.scene_cuts)
    if config.get("compositing",{}).get("enabled"):
        if masks is None:
            if controls.operation in {Operation.ENVIRONMENT_RECAST,Operation.FULL_RECAST,Operation.STYLE_RECAST}:
                masks=FrameStore.from_frames(np.full((controls.height,controls.width),255,np.uint8) for _ in range(len(frames)))
            else: raise ValueError("Selective replacement needs an edit mask sequence; provide --mask. Full-scene recast operations can omit it.")
        preserve=None
        if controls.preservation_masks:
            preserve=_read_mask(controls.preservation_masks,len(frames),(controls.height,controls.width))
        occ=config.get("occlusion",{})
        if occ.get("enabled"):
            checkpoint=config.get("depth",{}).get("checkpoint")
            foreground_path=occ.get("foreground_mask")
            if not checkpoint:
                raise ValueError("Depth-ordered occlusion handling requires a local depth.checkpoint")
            from .analysis import depth_sequence
            from .occlusion import visibility_mask
            depth_maps=depth_sequence(frames,"depth_anything",checkpoint,disk_backed=True)
            if foreground_path:
                foreground_masks=_read_mask(foreground_path,len(frames),(controls.height,controls.width))
            else:
                from .selection import select_and_track
                foreground_masks,_=select_and_track(frames,occ.get('foreground_prompt','hand'),selection.get('detector_model'),
                    selection.get('sam2_config'),selection.get('sam2_checkpoint'),device=selection.get('device','cuda'))
            visible_masks=FrameStore()
            for mask,depth,foreground in zip(masks,depth_maps,foreground_masks):
                selected=depth[mask>0]
                target_depth=float(np.median(selected)) if selected.size else float(np.median(depth))
                target_depth_map=np.full(depth.shape,target_depth,dtype=np.float32)
                visible_masks.append(visibility_mask(mask,target_depth_map,depth,foreground,near_is_high=True))
            masks=visible_masks.finish()
        generated=FrameStore.from_frames(composite(src,gen,mask,preserve_mask=(preserve[i] if preserve is not None else np.where(mask==0,255,0).astype(np.uint8)),feather_px=int(config["compositing"].get("feather_px",8)),color_match_enabled=bool(config["compositing"].get("color_match",False))) for i,(src,gen,mask) in enumerate(zip(frames,generated,masks)))
    identity_result=None
    if config.get("identity",{}).get("enabled") or config.get('identity',{}).get('evaluate_only'):
        ref=next((r for r in controls.references.references if r.category.upper()=="CHARACTER"),None)
        if ref is None:raise ValueError("Identity evaluation requires a CHARACTER reference image")
        identity_result=evaluator.evaluate_reference([r for r in controls.references.references if r.category.upper()=='CHARACTER'],generated,float(config["identity"].get("threshold",.55)),masks)
        (work/"identity_evaluation.json").write_text(json.dumps(identity_result,indent=2),encoding="utf-8")
        if not identity_result['passes_threshold'] and identity_cfg.get('enabled'):
            raise RuntimeError("Final identity acceptance failed after temporal repair/compositing; output is not approved. See identity_evaluation.json")
    if config.get('evaluation',{}).get('enabled'):
        from .evaluation import evaluate_transformation
        metrics=evaluate_transformation(frames,generated,masks,controls.scene_cuts,identity_result)
        (work/'evaluation.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    restoration=config.get('restoration',{})
    if restoration.get('enabled'):
        from .restoration import restore_faces
        generated=restore_faces(generated,restoration.get('model_path',''))
        if evaluator and identity_cfg.get('enabled'):
            restored_identity=evaluator.evaluate_reference([r for r in controls.references.references if r.category.upper()=='CHARACTER'],generated,float(identity_cfg.get('threshold',.55)),masks)
            if not restored_identity['passes_threshold']:raise RuntimeError('Face restoration violated the final identity gate')
            identity_result=restored_identity
    upscale=config.get('upscale',{})
    if upscale.get('enabled'):
        method=upscale.get('model','realesrgan');scale=int(upscale.get('scale',2))
        if method=='realesrgan':
            from .restoration import realesrgan
            generated=realesrgan(generated,upscale.get('model_path'),scale,disk_backed=True)
        elif method=='lanczos':
            import cv2
            generated=FrameStore.from_frames(cv2.resize(frame,None,fx=scale,fy=scale,interpolation=cv2.INTER_LANCZOS4) for frame in generated)
        else:raise ValueError(f'Unsupported upscale method: {method}')
    result=write_video(generated,out,controls.fps_rational,audio_from=source if preserve_audio else None,codec=codec,**timing_kwargs(controls))
    controls.save(work/"controls.json")
    (work/"run.json").write_text(json.dumps({"source":str(Path(source).resolve()),"output":str(result),"quality":settings.get('quality','source'),"backend":backend.name,"frames":len(generated),"fps":controls.fps,"fps_rational":controls.fps_rational,"is_vfr":controls.is_vfr,"timing_policy":"preserve_source_pts","frame_storage":"disk_backed","shots":json.loads((work/"shots.json").read_text(encoding="utf-8")) if (work/"shots.json").is_file() else [],"identity_evaluation":identity_result},indent=2),encoding="utf-8")
    return result
