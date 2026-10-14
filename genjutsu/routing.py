"""Operation-aware routing between real backends with explicit capabilities."""
from __future__ import annotations
from copy import deepcopy
from .controls import Operation


def route_config(config: dict, operation: Operation, controls=None) -> dict:
    routed=deepcopy(config); backend=routed.setdefault("backend",{}); name=str(backend.get("type","comfyui")).lower()
    if name not in ("auto","automatic"):
        return routed
    if operation in (Operation.CHARACTER_SWAP,Operation.MOTION_TRANSFER):
        if backend.get("use_wan_animate_for_character") is False:
            raise ValueError("Automatic character/motion routing requires Wan Animate; generic VACE is not an identity-preserving fallback. Select an explicit backend to perform a different experiment.")
        preprocess=backend.get("wan_animate_preprocess",{})
        if not preprocess.get("wan_repo") or not preprocess.get("checkpoint_dir"):
            raise ValueError("Automatic Wan Animate routing requires backend.wan_animate_preprocess.wan_repo and checkpoint_dir")
        backend["type"]="wan22"; backend["mode"]="animate"
        backend["model_id"]=backend.get("animate_model_id","Wan-AI/Wan2.2-Animate-14B-Diffusers")
    else:
        backend["type"]="wan_vace"
        backend["model_id"]=backend.get("vace_model_id",backend.get("model_id","Wan-AI/Wan2.1-VACE-1.3B-diffusers"))
    return routed
