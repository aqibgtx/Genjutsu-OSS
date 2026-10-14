"""Mask propagation and point tracking using measured optical flow."""
from __future__ import annotations
import numpy as np
from .analysis import optical_flow

def propagate_mask_sequence(frames: list[np.ndarray], initial_mask: np.ndarray, start_index: int = 0) -> list[np.ndarray]:
    if not frames: return []
    if initial_mask.shape[:2]!=frames[0].shape[:2]: raise ValueError("Initial mask dimensions must match frames")
    cv2=_cv2(); fw,bw=optical_flow(frames); masks=[None]*len(frames); masks[start_index]=(initial_mask>0).astype(np.uint8)*255
    for i in range(start_index,len(frames)-1): masks[i+1]=_warp(masks[i],fw[i],cv2)
    for i in range(start_index-1,-1,-1): masks[i]=_warp(masks[i+1],bw[i],cv2)
    return [m if m is not None else np.zeros_like(initial_mask) for m in masks]

def _warp(mask,flow,cv2):
    h,w=mask.shape[:2]; x,y=np.meshgrid(np.arange(w),np.arange(h)); return cv2.remap(mask.astype(np.uint8), (x-flow[:,:,0]).astype(np.float32),(y-flow[:,:,1]).astype(np.float32),cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)

def _cv2():
    try: import cv2; return cv2
    except ImportError as e: raise RuntimeError("OpenCV required; install genjutsu-oss[video].") from e
