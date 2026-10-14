"""Alpha compositing, mask boundaries, and local luminance/color matching."""
from __future__ import annotations
import numpy as np

def soft_mask(mask: np.ndarray, feather_px: int=8, dilation_px: int=0) -> np.ndarray:
    cv2=_cv2(); m=(mask>0).astype(np.uint8)*255
    if dilation_px>0: m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilation_px+1,2*dilation_px+1)))
    if feather_px>0: m=cv2.GaussianBlur(m,(0,0),feather_px/2)
    return m.astype(np.float32)/255

def match_color(source: np.ndarray, generated: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Match local lighting from a narrow preserved ring around the edit."""
    cv2=_cv2(); src=source.astype(np.float32); gen=generated.astype(np.float32); edit=mask>0
    if not edit.any(): return generated
    radius=max(3,min(source.shape[:2])//64)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
    ring=(cv2.dilate(edit.astype(np.uint8),kernel)>0)&~edit
    # Fall back for full-frame recasts, where no preserved ring exists.
    if ring.sum()<max(16,edit.sum()//100):return generated
    a=src[ring]; b=gen[ring]
    # Robust clipped moments avoid a few saturated pixels driving the transfer.
    alo,ahi=np.percentile(a,[5,95],axis=0); blo,bhi=np.percentile(b,[5,95],axis=0)
    amean=np.mean(np.clip(a,alo,ahi),axis=0); bmean=np.mean(np.clip(b,blo,bhi),axis=0)
    astd=np.std(np.clip(a,alo,ahi),axis=0); bstd=np.std(np.clip(b,blo,bhi),axis=0)
    gain=np.clip(astd/(bstd+1e-5),.5,2.)
    gain=np.where((astd<5)|(bstd<5),1.,gain)
    out=(gen-bmean)*gain+amean
    return np.clip(out,0,255).astype(np.uint8)

def composite(original: np.ndarray, generated: np.ndarray, edit_mask: np.ndarray, preserve_mask: np.ndarray|None=None, feather_px: int=8, color_match_enabled: bool=False) -> np.ndarray:
    if original.shape!=generated.shape: raise ValueError("Original and generated frames must have identical dimensions")
    m=soft_mask(edit_mask,feather_px)
    if preserve_mask is not None: m*=1-(preserve_mask>0).astype(np.float32)
    gen=match_color(original,generated,edit_mask) if color_match_enabled else generated
    return np.clip(original.astype(np.float32)*(1-m[...,None])+gen.astype(np.float32)*m[...,None],0,255).astype(np.uint8)

def _cv2():
    try: import cv2; return cv2
    except ImportError as e: raise RuntimeError("OpenCV required; install genjutsu-oss[video].") from e
