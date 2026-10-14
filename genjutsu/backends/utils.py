from __future__ import annotations
import numpy as np

def as_rgb8(frame) -> np.ndarray:
    """Normalize PIL or Diffusers NumPy frames (often float in [0,1]) to RGB uint8."""
    if hasattr(frame,"convert"): frame=frame.convert("RGB")
    arr=np.asarray(frame)
    if arr.ndim!=3 or arr.shape[2]<3: raise ValueError(f"Expected RGB frame, got shape {arr.shape}")
    arr=arr[:,:,:3]
    if np.issubdtype(arr.dtype,np.floating):
        maximum=float(np.nanmax(arr)) if arr.size else 0.0
        arr=np.clip(arr*(255.0 if maximum<=1.0 else 1.0),0,255)
    return np.ascontiguousarray(arr,dtype=np.uint8)
