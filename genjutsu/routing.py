"""Operation-aware routing between real backends with explicit capabilities."""
from __future__ import annotations
from copy import deepcopy
from .controls import Operation


def route_config(config: dict, operation: Operation, controls=None) -> dict:
    routed=deepcopy(config); backend=routed.setdefault("backend",{}); name=str(backend.get("type","comfyui")).lower()
    if name not in ("auto","automatic"):
        return routed
    # VACE performs source-video masked editing across object, product, clothing,
    # character and environment categories. Wan Animate is chosen only when the
    # caller explicitly opts into its character-specific pipeline.
    animate_enabled=bool(backend.get("use_wan_animate_for_character",False))
    if operation in (Operation.CHARACTER_SWAP,Operation.MOTION_TRANSFER) and animate_enabled:
        preprocess=backend.get("wan_animate_preprocess",{})
        if not preprocess.get("wan_repo") or not preprocess.get("checkpoint_dir"):
            raise ValueError("Automatic Wan Animate routing requires backend.wan_animate_preprocess.wan_repo and checkpoint_dir")
        backend["type"]="wan22"; backend["mode"]="animate"
    else:
        backend["type"]="wan_vace"
    return routed
