#!/usr/bin/env python
"""Executable, local-only RunPod validation suites for real adapters and inference."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

SUITES=("selection","wan-preprocess","wan-animate","wan-vace","wan-i2v","depth","pose","raft","identity","realesrgan","multishot-e2e")


def require(path: Path, label: str):
    if not path.exists():raise FileNotFoundError(f"Required {label} is missing: {path}")
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",choices=[*SUITES,"all"],required=True)
    parser.add_argument("--assets-dir",type=Path,required=True,help="Validation bundle: source.mp4, reference.png, and mask/*.png")
    parser.add_argument("--models-dir",type=Path,required=True,help="Directory of already-downloaded model weights and config manifests")
    parser.add_argument("--run-inference",action="store_true",help="Required for generative or learned-model GPU suites; never downloads weights")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--dtype",default="bfloat16")
    parser.add_argument("--steps",type=int,default=20)
    args=parser.parse_args()
    assets=args.assets_dir.resolve(); models=args.models_dir.resolve()
    source=require(assets/"source.mp4","representative source video")
    reference=require(assets/"reference.png","reference image")
    manifest_path=require(assets/"manifest.json","validation manifest")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    model_manifest_path=models/"manifest.json"
    model_manifest=json.loads(model_manifest_path.read_text(encoding="utf-8")) if model_manifest_path.is_file() else {}
    from genjutsu.video import read_frames,probe_video
    frames,video_info=read_frames(source,disk_backed=True)
    if not frames or video_info.fps<=0:raise RuntimeError("source.mp4 did not decode to a valid constant-rate video")
    if args.suite in ("wan-i2v","all"):
        from genjutsu.analysis import scene_cuts
        if scene_cuts(frames):raise SystemExit("Wan I2V suite requires a single-shot clip with no detected cuts so its 4k+1 frame contract remains intact")
        if (len(frames)-1)%4!=0:raise SystemExit(f"Wan I2V requires 4k+1 frames; source has {len(frames)}")
    if args.suite in ("wan-animate","wan-vace","wan-i2v","multishot-e2e","all") and not args.run_inference:
        raise SystemExit("This suite runs real model inference. Pass --run-inference explicitly after placing licensed model weights locally.")
    suites=SUITES if args.suite=="all" else (args.suite,)
    results=[]
    for suite in suites:
        print(f"RUN {suite}",flush=True)
        if suite=="selection":
            from genjutsu.analysis import scene_cuts
            from genjutsu.selection import select_and_track
            target=str(manifest.get("target_prompt","person"))
            from genjutsu.frame_store import FrameStore
            masks=FrameStore(); details=[]; boundaries=[0]+scene_cuts(frames)+[len(frames)]
            for start,end in zip(boundaries,boundaries[1:]):
                current,detail=select_and_track(frames[start:end],target,str(require(Path(model_manifest.get("grounding_dino",models/"grounding-dino")),"Grounding DINO files")),
                    str(require(Path(model_manifest.get("sam2_config",models/"sam2.yaml")),"SAM 2 config")),
                    str(require(Path(model_manifest.get("sam2_checkpoint",models/"sam2_hiera_large.pt")),"SAM 2 checkpoint")),
                    int(manifest.get("selection_frame",0)),args.device,detection_index=int(manifest.get("detection_index",0)),chunk_frames=int(manifest.get("selection_chunk_frames",32)))
                masks.extend(current);details.append(detail)
            if len(masks)!=len(frames) or not all(np.any(m) for m in masks):raise RuntimeError("SAM 2 result has empty or missing shot masks")
            results.append({"suite":suite,"shots":len(details),"tracked_masks":len(masks),"details":details})
        elif suite=="wan-preprocess":
            from genjutsu.wan_preprocess import preprocess_wan_animate,validate_wan_conditioning
            prep=require(Path(model_manifest.get("wan_preprocess_checkpoint",models/"wan22-preprocess")),"Wan Animate preprocessing checkpoint directory")
            repo=require(Path(model_manifest.get("wan_repo",models/"Wan2.2")),"Wan 2.2 source checkout")
            outputs=preprocess_wan_animate(source,reference,assets/"runpod-wan-preprocess",repo,prep,-1,tuple(manifest.get("resolution_area",[1280,720])))
            validation=validate_wan_conditioning(outputs,video_info.fps,len(frames));results.append({"suite":suite,"outputs":outputs,"contract":validation})
        elif suite in ("wan-vace","wan-animate","wan-i2v","multishot-e2e"):
            from genjutsu.pipeline import transform_video
            job_source=source;job_frames=frames;job_info=video_info
            if suite=="multishot-e2e" and args.suite=="all" and (assets/"source_multishot.mp4").is_file():
                job_source=assets/"source_multishot.mp4";job_frames,job_info=read_frames(job_source)
            if suite=="multishot-e2e":
                from genjutsu.analysis import scene_cuts
                if not scene_cuts(job_frames):raise RuntimeError("Multi-shot suite needs a source with at least one detected scene cut")
            mask_path=assets/"mask_multishot" if suite=="multishot-e2e" and (assets/"mask_multishot").is_dir() else assets/"mask"
            mask_dir=None if suite in ("wan-animate","wan-i2v") else require(mask_path,"per-frame edit mask PNG directory")
            workflow={"backend":{"type":"wan22" if suite in ("wan-animate","wan-i2v") else "wan_vace","mode":"i2v" if suite=="wan-i2v" else "animate","device":args.device,"dtype":args.dtype,"steps":args.steps,"cpu_offload":True,"allow_download":False},
                "processing":{"shot_retries":0,"chunk_frames":81,"overlap_frames":9},
                "temporal":{"enabled":True,"strength":.15},"compositing":{"enabled":True,"feather_px":4,"color_match":True},
                "cache":{"enabled":False}}
            model_key="wan_animate_model" if suite=="wan-animate" else ("wan_i2v_model" if suite=="wan-i2v" else "wan_vace_model")
            model_default="Wan2.2-Animate-14B-Diffusers" if suite=="wan-animate" else ("Wan2.1-I2V-14B-480P-Diffusers" if suite=="wan-i2v" else "Wan2.1-VACE-1.3B-diffusers")
            model_path=model_manifest.get(model_key)
            workflow["backend"]["model_id"]=str(require(Path(model_path or models/model_default),"local Diffusers model directory"))
            if suite=="wan-animate":
                workflow["backend"]["mode"]="animate"
                workflow["backend"]["wan_animate_preprocess"]={"wan_repo":str(require(Path(model_manifest.get("wan_repo",models/"Wan2.2")),"Wan checkout")),
                    "checkpoint_dir":str(require(Path(model_manifest.get("wan_preprocess_checkpoint",models/"wan22-preprocess")),"Wan preprocessing checkpoints")),
                    "fps":-1,"resolution_area":manifest.get("resolution_area",[1280,720])}
            import yaml
            config_path=assets/f"runpod-{suite}.yaml";config_path.write_text(yaml.safe_dump(workflow),encoding="utf-8")
            output=assets/f"runpod-{suite}.mp4"
            operation="full_recast" if suite=="wan-i2v" else ("character_swap" if suite not in ("wan-vace",) else str(manifest.get("operation","object_swap")))
            transform_settings={"reference_category":str(manifest.get("reference_category","CHARACTER" if suite not in ("wan-vace","multishot-e2e") else "OBJECT"))}
            if suite=="multishot-e2e":transform_settings["shots"]=manifest.get("shots",{})
            result=transform_video(job_source,output,operation,
                str(manifest.get("prompt","A coherent high quality transformation following the supplied reference.")),[str(reference)],str(mask_dir) if mask_dir else None,
                config_path=config_path,seed=int(manifest.get("seed",1234)),**transform_settings)
            output_info=probe_video(result)
            from fractions import Fraction
            tolerance=max(float(Fraction(job_info.time_base)),float(Fraction(output_info.time_base)))
            if not np.allclose(output_info.timestamps,job_info.timestamps,atol=tolerance,rtol=0) or abs(output_info.duration-job_info.duration)>tolerance+.001:raise RuntimeError("Generated output presentation timestamps or endpoint differ from source")
            if output_info.frame_count!=job_info.frame_count or abs(output_info.fps-job_info.fps)>.05:raise RuntimeError(f"Generated output timing mismatch: source={job_info.frame_count}@{job_info.fps}, output={output_info.frame_count}@{output_info.fps}")
            if job_info.has_audio and not output_info.has_audio:raise RuntimeError("Generated output lost the source audio stream")
            results.append({"suite":suite,"output":str(result),"frames":output_info.frame_count,"fps":output_info.fps,"audio_preserved":output_info.has_audio})
        elif suite=="depth":
            from genjutsu.analysis import depth_sequence
            model=require(Path(model_manifest.get("depth_model",models/"depth-anything")),"Depth Anything local model directory")
            values=depth_sequence(frames,"depth_anything",str(model))
            if len(values)!=len(frames) or any(not np.isfinite(d).all() for d in values):raise RuntimeError("Depth Anything result length or numeric validation failed")
            results.append({"suite":suite,"frames":len(values),"shape":list(values[0].shape)})
        elif suite=="pose":
            from genjutsu.analysis import pose_sequence
            values=pose_sequence(frames,"mediapipe")
            visible=sum(1 for item in values if item.get("body"))
            if visible==0:raise RuntimeError("MediaPipe detected no bodies in the representative source")
            results.append({"suite":suite,"frames":len(values),"body_detected_frames":visible})
        elif suite=="raft":
            from genjutsu.analysis import raft_flow
            checkpoint=require(Path(model_manifest.get("raft_checkpoint",models/"raft_large.pth")),"Torchvision RAFT state_dict")
            forward,backward=raft_flow(frames,str(checkpoint))
            if len(forward)!=len(frames)-1 or any(not np.isfinite(flow).all() for flow in forward+backward):raise RuntimeError("RAFT flow output failed numeric/length validation")
            results.append({"suite":suite,"pairs":len(forward),"shape":list(forward[0].shape)})
        elif suite=="identity":
            from genjutsu.identity import LocalInsightFace
            result=LocalInsightFace(root=str(require(models/"insightface","InsightFace model root"))).evaluate_reference(reference,frames,float(manifest.get("identity_threshold",.55)))
            if result["detected_frames"]==0:raise RuntimeError("InsightFace detected no faces in the validation source")
            results.append({"suite":suite,**result})
        elif suite=="realesrgan":
            from genjutsu.restoration import realesrgan
            weights=require(Path(model_manifest.get("realesrgan_weights",models/"RealESRGAN_x4plus.pth")),"Real-ESRGAN x4plus weights")
            output=realesrgan(frames[:min(3,len(frames))],str(weights),2)
            if output[0].shape[1]!=frames[0].shape[1]*2 or output[0].shape[0]!=frames[0].shape[0]*2:raise RuntimeError("Real-ESRGAN output scale is not 2x")
            results.append({"suite":suite,"frames":len(output),"scale":2})
    print(json.dumps({"status":"passed","gpu":args.device,"results":results},indent=2))


if __name__=="__main__":
    try:main()
    except Exception as exc:
        print(f"VALIDATION FAILED: {type(exc).__name__}: {exc}",file=sys.stderr);raise SystemExit(1)
