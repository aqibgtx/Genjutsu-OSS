"""Reference-free and reference-based video quality measurements."""
from __future__ import annotations
import numpy as np
from .temporal import flicker_score, flow_consistency

def compare_videos(source: list[np.ndarray], output: list[np.ndarray], edit_masks: list[np.ndarray]|None=None) -> dict[str,float|None]:
    if len(source)!=len(output): raise ValueError("Video lengths differ")
    if not source:return {}
    temporal=flow_consistency(output)
    if edit_masks:
        sims=[]
        for s,o,m in zip(source,output,edit_masks):
            keep=m==0
            if keep.any(): sims.append(float(1-np.mean(np.abs(s[keep].astype(np.float32)-o[keep].astype(np.float32)))/255))
        background=float(np.mean(sims)) if sims else None
    else: background=None
    return {"background_similarity":background,"temporal_consistency":temporal,"flicker_score":flicker_score(output),"motion_similarity":None,"camera_similarity":None,"identity_similarity":None,"object_similarity":None,"prompt_alignment":None,"perceptual_quality":None}
