"""One-step dependency checks and ComfyUI extension install; never downloads model weights."""
from __future__ import annotations
import argparse, importlib.util, shutil, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(); p.add_argument("--comfyui-path"); p.add_argument("--check",action="store_true"); p.add_argument("--install-deps",action="store_true"); args=p.parse_args()
    if args.install_deps: subprocess.run([sys.executable,"-m","pip","install","-e",str(ROOT)+"[video,ui,api]"],check=True)
    missing=[x for x in ("ffmpeg","ffprobe") if shutil.which(x) is None]
    modules={m:importlib.util.find_spec(m) is not None for m in ("numpy","PIL","yaml","requests","cv2","torch","diffusers","transformers","sam2","mediapipe")}
    print("Python:",sys.version.split()[0]); print("FFmpeg:","available" if not missing else "missing: "+", ".join(missing)); print("Modules:",modules)
    if args.comfyui_path:
        comfy=Path(args.comfyui_path).expanduser().resolve()
        if not (comfy/"main.py").exists(): raise SystemExit(f"ComfyUI main.py not found under {comfy}")
        target=comfy/"custom_nodes"/"ComfyUI-Genjutsu"
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(ROOT/"ComfyUI-Genjutsu",target,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
        shutil.copytree(ROOT/"genjutsu",target/"genjutsu",dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
        shutil.copytree(ROOT/"workflows",target/"workflows",dirs_exist_ok=True)
        print("Installed:",target)
    return 2 if missing else 0
if __name__=="__main__": raise SystemExit(main())
