"""ComfyUI custom-node entry point for Genjutsu-OSS."""
from pathlib import Path
import sys
root=str(Path(__file__).resolve().parent.parent)
if root not in sys.path: sys.path.insert(0,root)
custom_root=str(Path(__file__).resolve().parent)
if custom_root not in sys.path: sys.path.insert(0,custom_root)
from genjutsu_comfy_nodes.core import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY="./web"
