"""Flow-warped temporal diagnostics and conservative smoothing."""
from __future__ import annotations
import numpy as np
from .analysis import optical_flow

def flicker_score(frames: list[np.ndarray]) -> float:
    if len(frames)<2: return 0.0
    return float(np.mean([np.mean(np.abs(a.astype(np.float32)-b.astype(np.float32)))/255 for a,b in zip(frames,frames[1:])]))

def flow_consistency(frames: list[np.ndarray]) -> float:
    if len(frames)<2:return 1.0
    import cv2
    fw,_=optical_flow(frames); vals=[]
    for a,b,f in zip(frames,frames[1:],fw):
        h,w=f.shape[:2]; x,y=np.meshgrid(np.arange(w),np.arange(h)); warp=cv2.remap(a,(x-f[:,:,0]).astype(np.float32),(y-f[:,:,1]).astype(np.float32),cv2.INTER_LINEAR)
        vals.append(float(np.mean(np.abs(warp.astype(np.float32)-b.astype(np.float32)))/255))
    return max(0.,1.-float(np.mean(vals)))

def stabilize_generated(frames: list[np.ndarray], strength: float=.2, scene_cuts: list[int]|None=None) -> list[np.ndarray]:
    """Flow-warp temporally stable pixels, rejecting occlusions and resetting at cuts."""
    if not frames or strength<=0:return frames
    cv2=_cv2(); out=[frames[0]]; cuts=set(scene_cuts or [])
    for i in range(1,len(frames)):
        if i in cuts:
            out.append(frames[i]);continue
        forward,back=optical_flow([out[-1],frames[i]])
        h,w=back[0].shape[:2]; x,y=np.meshgrid(np.arange(w),np.arange(h))
        warped=cv2.remap(out[-1],(x+back[0][:,:,0]).astype(np.float32),(y+back[0][:,:,1]).astype(np.float32),cv2.INTER_LINEAR)
        # Forward/backward flow disagreement marks disocclusions and unreliable motion.
        sampled=cv2.remap(forward[0],(x+back[0][:,:,0]).astype(np.float32),(y+back[0][:,:,1]).astype(np.float32),cv2.INTER_LINEAR)
        error=np.linalg.norm(sampled+back[0],axis=2)
        magnitude=np.linalg.norm(back[0],axis=2)
        reliable=error < (1.0+0.05*magnitude)
        alpha=(reliable.astype(np.float32)*float(np.clip(strength,0,1)))[...,None]
        out.append(np.clip(frames[i].astype(np.float32)*(1-alpha)+warped.astype(np.float32)*alpha,0,255).astype(np.uint8))
    return out

def _cv2():
    try: import cv2; return cv2
    except ImportError as e: raise RuntimeError("OpenCV required; install genjutsu-oss[video].") from e
