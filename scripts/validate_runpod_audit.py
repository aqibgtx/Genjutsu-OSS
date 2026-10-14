#!/usr/bin/env python
"""Real Wan VACE timing/cache regression acceptance; never downloads weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import threading
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True,help="Local Wan VACE config, allow_download=false")
    parser.add_argument("--revision-b-config",type=Path,help="Optional second real local checkpoint for revision/load invalidation")
    parser.add_argument("--run-inference",action="store_true")
    parser.add_argument("--max-host-mib",type=float,default=16384)
    args=parser.parse_args()
    if not args.run_inference:
        raise SystemExit("Pass --run-inference to authorize these real GPU generation jobs")
    from genjutsu.cache import backend_signature
    from genjutsu.backends.vace import WanVACEBackend
    from genjutsu.config import load_config
    from genjutsu.pipeline import analyze_video,transform_video
    from genjutsu.video import probe_video
    from fractions import Fraction
    import psutil
    import torch
    import yaml
    if not torch.cuda.is_available():raise RuntimeError("Real CUDA inference is required for this acceptance test")
    assets=args.assets_dir.resolve();output=assets/"audit-results";output.mkdir(parents=True,exist_ok=True)
    for name in ("source_vfr.mp4","source_ntsc.mp4","reference.png","reference_b.png","mask_vfr","mask_ntsc"):
        if not (assets/name).exists():raise FileNotFoundError(assets/name)
    config=load_config(args.config)
    if config["backend"]["type"] not in {"wan_vace","vace"}:raise ValueError("Audit requires the real Wan VACE adapter")
    if not Path(config["backend"]["model_id"]).is_dir():raise ValueError("Set backend.model_id to an existing local Diffusers model directory")
    config["backend"]["allow_download"]=False
    config["cache"]={"enabled":True,"directory":str(output/"cache")}
    config["processing"]["shot_retries"]=0
    config_path=output/"config.yaml";config_path.write_text(yaml.safe_dump(config),encoding="utf-8")
    reference=output/"reference_work.png";shutil.copyfile(assets/"reference.png",reference)
    baseline=psutil.Process().memory_info().rss;peak=[baseline];stop=threading.Event()
    def sample():
        while not stop.wait(.05):peak[0]=max(peak[0],psutil.Process().memory_info().rss)
    thread=threading.Thread(target=sample);thread.start()
    torch.cuda.reset_peak_memory_stats()
    results=[]

    def run(label,source,mask,settings=None,cfg=config_path):
        target=output/f"{label}.mp4"
        result=transform_video(source,target,"object_swap","Replace the selected subject using the reference image.",[str(reference)],str(mask),
            config_path=cfg,seed=7821,reference_category="OBJECT",shots=settings or {})
        info=probe_video(source);actual=probe_video(result)
        if actual.frame_count!=info.frame_count or (actual.width,actual.height)!=(info.width,info.height):raise RuntimeError("Output frame count/geometry differs from source")
        tolerance=max(float(Fraction(info.time_base)),float(Fraction(actual.time_base)))
        if not np.allclose(actual.timestamps,info.timestamps,atol=tolerance,rtol=0):raise RuntimeError("Output presentation timestamps changed")
        if abs(actual.duration-info.duration)>tolerance+.001:raise RuntimeError("Output endpoint changed")
        if not actual.has_audio:raise RuntimeError("Output lost audio")
        if abs(float(actual.audio_metadata.get("start_time",0))-(float(info.audio_metadata.get("start_time",0))-info.start_time))>.025:raise RuntimeError("Audio offset changed by more than one AAC packet")
        state=json.loads(target.with_suffix(".work").joinpath("shots.json").read_text())
        results.append({"label":label,"output":str(result),"shots":state,"frames":actual.frame_count,"timestamps_preserved":True})
        return state

    try:
        for name in ("vfr","ntsc"):
            source=assets/f"source_{name}.mp4";mask=assets/f"mask_{name}"
            controls,frames=analyze_video(source)
            if len(controls.scene_cuts)<2 or not probe_video(source).has_audio:raise RuntimeError("Each source requires three detected shots and audio")
            if name=="vfr" and not controls.is_vfr:raise RuntimeError("source_vfr.mp4 must have genuinely variable intervals")
            if name=="ntsc" and Fraction(controls.fps_rational)!=Fraction(30000,1001):raise RuntimeError("source_ntsc.mp4 must be exactly 30000/1001 FPS")
            first=run(name,source,mask)
            if any(s["status"]!="complete" for s in first):raise RuntimeError("Fresh audit unexpectedly reused an existing cache; remove audit-results before running")
            second=run(name,source,mask)
            if any(s["status"]!="cached" for s in second):raise RuntimeError("Unchanged inference did not reuse all shot caches")
            changed=run(name,source,mask,{"1":{"settings":{"seed":7822}}})
            if [s["status"] for s in changed]!=["cached","complete"]+["cached"]*(len(changed)-2):raise RuntimeError("Changing one shot seed did not invalidate exactly that shot")
        shutil.copyfile(assets/"reference_b.png",reference)
        changed=run("vfr",assets/"source_vfr.mp4",assets/"mask_vfr")
        if any(s["status"]!="complete" for s in changed):raise RuntimeError("Reference bytes changed at the same path but a shot reused stale output")
        if args.revision_b_config:
            alternate=load_config(args.revision_b_config)
            alternate["backend"]["allow_download"]=False
            alternate["cache"]=config["cache"]
            signature_a=backend_signature(WanVACEBackend(config["backend"]))
            signature_b=backend_signature(WanVACEBackend(alternate["backend"]))
            if not signature_b["cache_safe"] or signature_a["model"]["content"]==signature_b["model"]["content"]:raise RuntimeError("Revision B must contain different real local checkpoint bytes")
            alternate_path=output/"revision_b.yaml";alternate_path.write_text(yaml.safe_dump(alternate),encoding="utf-8")
            changed=run("revision_b",assets/"source_vfr.mp4",assets/"mask_vfr",cfg=alternate_path)
            if any(s["status"]!="complete" for s in changed):raise RuntimeError("Changed model checkpoint reused a stale shot")
    finally:
        stop.set();thread.join()
    host=(peak[0]-baseline)/2**20
    if host>args.max_host_mib:raise RuntimeError(f"Host RSS growth {host:.1f} MiB exceeds {args.max_host_mib} MiB")
    report={"status":"passed","real_inference":True,"gpu":torch.cuda.get_device_name(),"gpu_peak_mib":torch.cuda.max_memory_allocated()/2**20,
        "host_peak_growth_mib":host,"revision_b_validated":bool(args.revision_b_config),"results":results}
    (output/"validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
