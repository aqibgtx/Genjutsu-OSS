from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from .pipeline import analyze_video, transform_video
from .controls import Operation
from .project import create_project, inspect_project, restore_project
from .config import load_config
from .backends import create_backend
from .video import probe_video

def build_parser():
    p=argparse.ArgumentParser(prog="genjutsu",description="Local video transformation pipeline. Generation requires a real configured model backend.")
    p.add_argument("--version",action="version",version="genjutsu-oss 0.1.0"); sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("analyze",help="Read source video and emit measured metadata"); a.add_argument("source"); a.add_argument("--output",default="controls.json")
    t=sub.add_parser("transform",help="Run a configured real model backend and composite the result"); t.add_argument("source"); t.add_argument("output"); t.add_argument("--operation",choices=[o.value for o in Operation],required=True); t.add_argument("--prompt",required=True); t.add_argument("--negative-prompt",default=""); t.add_argument("--reference",action="append",default=[]); t.add_argument("--mask"); t.add_argument("--config",default="config.yaml"); t.add_argument("--seed",type=int,default=0); t.add_argument("--codec",choices=["h264","h265","prores","vp9"],default="h264"); t.add_argument("--no-audio",action="store_true")
    c=sub.add_parser("check",help="Check local media/runtime and backend configuration"); c.add_argument("--config",default="config.yaml"); c.add_argument("--video")
    pr=sub.add_parser("project",help="Create, inspect, or restore a .genjutsu archive"); ps=pr.add_subparsers(dest="project_command",required=True); pc=ps.add_parser("create"); pc.add_argument("source"); pc.add_argument("output"); pc.add_argument("--include-source",action="store_true"); pi=ps.add_parser("inspect"); pi.add_argument("path"); pre=ps.add_parser("restore"); pre.add_argument("path"); pre.add_argument("directory")
    ui=sub.add_parser("ui",help="Start the optional local Gradio interface"); ui.add_argument("--config",default="config.yaml"); ui.add_argument("--server-name",default="127.0.0.1"); ui.add_argument("--port",type=int,default=7860)
    api=sub.add_parser("api",help="Start the optional FastAPI developer API"); api.add_argument("--host",default="127.0.0.1"); api.add_argument("--port",type=int,default=8000)
    return p

def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        if args.command=="analyze":
            controls,_=analyze_video(args.source); controls.save(args.output); print(json.dumps({"controls":args.output,"frames":controls.frame_count,"fps":controls.fps,"resolution":[controls.width,controls.height],"scene_cuts":controls.scene_cuts,"camera_method":controls.camera_motion.get("method")},indent=2))
        elif args.command=="transform": print(transform_video(args.source,args.output,args.operation,args.prompt,args.reference,args.mask,args.config,args.negative_prompt,args.seed,args.codec,not args.no_audio))
        elif args.command=="check":
            result={"ffmpeg":_which("ffmpeg"),"ffprobe":_which("ffprobe"),"config":load_config(args.config),"backend":None,"video":None}
            try: result["backend"]=create_backend(load_config(args.config)).name
            except Exception as e: result["backend_error"]=str(e)
            if args.video:
                v=probe_video(args.video); result["video"]=v.__dict__
            print(json.dumps(result,indent=2))
        elif args.command=="project":
            if args.project_command=="inspect": print(json.dumps(inspect_project(args.path),indent=2))
            elif args.project_command=="restore": print(restore_project(args.path,args.directory))
            else:
                controls,_=analyze_video(args.source); print(create_project(args.output,controls,args.include_source))
        elif args.command=="ui":
            from .ui import launch; launch(args.config,args.server_name,args.port)
        elif args.command=="api":
            try: import uvicorn
            except ImportError as e: raise RuntimeError("Install genjutsu-oss[ui] or install uvicorn and fastapi to use the developer API") from e
            uvicorn.run("genjutsu.api:app",host=args.host,port=args.port)
    except Exception as e:
        print(f"genjutsu: {e}",file=sys.stderr); return 2
    return 0

def _which(name):
    import shutil
    return shutil.which(name)

if __name__=="__main__": raise SystemExit(main())
