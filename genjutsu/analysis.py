"""Classical, local video analysis. Learned model hooks are separate and fail clearly when absent."""
from __future__ import annotations
import numpy as np

def _cv2():
    try: import cv2; return cv2
    except ImportError as e: raise RuntimeError("OpenCV is required for video analysis. Install genjutsu-oss[video].") from e

def scene_cuts(frames: list[np.ndarray], threshold: float = .42) -> list[int]:
    cv2 = _cv2(); cuts = []
    for i in range(1, len(frames)):
        a = cv2.cvtColor(cv2.resize(frames[i-1], (64, 64)), cv2.COLOR_RGB2HSV)
        b = cv2.cvtColor(cv2.resize(frames[i], (64, 64)), cv2.COLOR_RGB2HSV)
        ha = cv2.calcHist([a], [0, 1], None, [32, 32], [0, 180, 0, 256]); hb = cv2.calcHist([b], [0, 1], None, [32, 32], [0, 180, 0, 256])
        if cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA) >= threshold: cuts.append(i)
    return cuts

def optical_flow(frames: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Farneback CPU flow in pixels/frame; this is a real classical baseline, not RAFT."""
    cv2 = _cv2(); fw=[]; bw=[]
    for a,b in zip(frames, frames[1:]):
        ga=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY); gb=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY)
        fw.append(cv2.calcOpticalFlowFarneback(ga,gb,None,.5,3,15,3,5,1.2,0))
        bw.append(cv2.calcOpticalFlowFarneback(gb,ga,None,.5,3,15,3,5,1.2,0))
    return fw,bw

def raft_flow(frames: list[np.ndarray], checkpoint: str, iterations: int=20, device: str|None=None) -> tuple[list[np.ndarray],list[np.ndarray]]:
    """Torchvision RAFT inference from a user-provided local state_dict; never fetches weights."""
    from pathlib import Path
    if not Path(checkpoint).is_file(): raise FileNotFoundError(checkpoint)
    try:
        import torch
        from torchvision.models.optical_flow import raft_large
    except ImportError as e: raise RuntimeError("RAFT requires a compatible PyTorch and Torchvision installation.") from e
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    model=raft_large(weights=None,progress=False)
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if "state_dict" in state:state=state["state_dict"]
    model.load_state_dict(state,strict=True); model.eval().to(device)
    def infer(a,b):
        h,w=a.shape[:2]; nh=((h+7)//8)*8; nw=((w+7)//8)*8
        aa=cv2.resize(a,(nw,nh)); bb=cv2.resize(b,(nw,nh))
        x=torch.from_numpy(aa).permute(2,0,1).float()[None].to(device); y=torch.from_numpy(bb).permute(2,0,1).float()[None].to(device)
        x=x/127.5-1; y=y/127.5-1
        with torch.inference_mode(): flow=model(x,y,num_flow_updates=iterations)[-1][0].permute(1,2,0).cpu().numpy()
        return cv2.resize(flow,(w,h))*np.array([w/nw,h/nh],np.float32)
    cv2=_cv2(); fwd=[]; back=[]
    for a,b in zip(frames,frames[1:]): fwd.append(infer(a,b)); back.append(infer(b,a))
    return fwd,back

def optical_flow_with_backend(frames,model="farneback",checkpoint=None,iterations=20):
    if model.lower() in ("opencv","farneback"): return optical_flow(frames)
    if model.lower()=="raft":
        if not checkpoint: raise ValueError("RAFT requires a local checkpoint path")
        return raft_flow(frames,checkpoint,iterations)
    raise ValueError(f"Unknown optical-flow model: {model}")

def estimate_camera(frames: list[np.ndarray]) -> dict:
    """Estimate frame-to-frame global translation, rotation and scale via robust affine fit."""
    cv2=_cv2(); trajectory=[]
    for a,b in zip(frames,frames[1:]):
        g1=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY); g2=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY)
        p=cv2.goodFeaturesToTrack(g1, maxCorners=600, qualityLevel=.01, minDistance=8)
        if p is None: trajectory.append({"dx":0.,"dy":0.,"rotation":0.,"zoom":1.}); continue
        q,st,_=cv2.calcOpticalFlowPyrLK(g1,g2,p,None)
        good=st.ravel()==1
        if good.sum()<3: trajectory.append({"dx":0.,"dy":0.,"rotation":0.,"zoom":1.}); continue
        m,_=cv2.estimateAffinePartial2D(p[good],q[good],method=cv2.RANSAC,ransacReprojThreshold=3)
        if m is None: m=np.eye(2,3)
        trajectory.append({"dx":float(m[0,2]),"dy":float(m[1,2]),"rotation":float(np.arctan2(m[1,0],m[0,0])),"zoom":float(np.hypot(m[0,0],m[1,0]))})
    return {"method":"opencv_ransac_affine","trajectory":trajectory}

def detect_objects(frames: list[np.ndarray], max_frames: int = 1) -> dict:
    """Returns explicit scene metadata; object-class detection needs an installed detector."""
    if not frames: return {"objects": [], "method": "none"}
    return {"objects": [], "method": "not_configured", "note": "Install/configure an object detector such as Grounding DINO; no labels are fabricated."}

def grounded_detect(frames: list[np.ndarray], prompt: str, checkpoint: str, box_threshold:float=.3, text_threshold:float=.25) -> list[dict]:
    """Open-vocabulary Grounding DINO via Transformers using a local model directory."""
    from pathlib import Path
    if not Path(checkpoint).is_dir(): raise FileNotFoundError(f"Grounding DINO model directory missing: {checkpoint}")
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except ImportError as e: raise RuntimeError("Install PyTorch and Transformers with Grounding DINO support.") from e
    processor=AutoProcessor.from_pretrained(checkpoint,local_files_only=True); model=AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint,local_files_only=True).eval()
    device="cuda" if torch.cuda.is_available() else "cpu"; model.to(device); output=[]
    from PIL import Image
    for idx,frame in enumerate(frames):
        caption=prompt.strip()
        if not caption.endswith('.'):caption+='.'
        inputs=processor(images=Image.fromarray(frame),text=caption,return_tensors="pt").to(device)
        with torch.inference_mode(): result=model(**inputs)
        from inspect import signature
        post=processor.post_process_grounded_object_detection
        threshold_key='threshold' if 'threshold' in signature(post).parameters else 'box_threshold'
        parsed=post(result,inputs.input_ids,**{threshold_key:float(box_threshold)},text_threshold=float(text_threshold),target_sizes=[(frame.shape[0],frame.shape[1])])[0]
        labels=parsed.get('text_labels',parsed.get('labels',[]))
        output.append({"frame":idx,"detections":[{"label":str(label),"score":float(score),"box":[float(x) for x in box]} for box,score,label in zip(parsed["boxes"],parsed["scores"],labels)]})
    return output

def depth_sequence(frames: list[np.ndarray], model: str = "none", checkpoint: str | None = None, *, disk_backed=False) -> list[np.ndarray]:
    if model in ("none", "disabled"): raise RuntimeError("Learned depth is not configured. Set model=depth_anything and provide a local Transformers model directory.")
    if model not in ("depth_anything","depth-anything"): raise RuntimeError(f"Unsupported depth adapter {model}; supported local baseline is Transformers Depth Anything.")
    if not checkpoint: raise RuntimeError("Depth Anything weights must already exist locally; set depth.checkpoint. This adapter never downloads weights.")
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        from PIL import Image
    except ImportError as e: raise RuntimeError("Install PyTorch and Transformers to use Depth Anything.") from e
    processor=AutoImageProcessor.from_pretrained(checkpoint,local_files_only=True); model_obj=AutoModelForDepthEstimation.from_pretrained(checkpoint,local_files_only=True).eval()
    from .frame_store import FrameStore
    device="cuda" if torch.cuda.is_available() else "cpu"; model_obj.to(device); outs=FrameStore() if disk_backed else []
    with torch.inference_mode():
        for frame in frames:
            inputs=processor(images=Image.fromarray(frame),return_tensors="pt").to(device); pred=model_obj(**inputs).predicted_depth
            depth=torch.nn.functional.interpolate(pred.unsqueeze(1),size=frame.shape[:2],mode="bicubic",align_corners=False).squeeze().cpu().numpy().astype(np.float32)
            outs.append(depth)
    if disk_backed:
        return normalize_depth_store(outs.finish())
    # Normalize with shared clip percentiles, not independent per-frame normalization.
    vals=np.concatenate([d.ravel() for d in outs]); lo,hi=np.percentile(vals,[2,98]); scale=max(float(hi-lo),1e-6)
    return [np.clip((d-lo)/scale,0,1).astype(np.float32) for d in outs]


def normalize_depth_store(maps):
    """Shared clip percentiles from a bounded 4096-bin histogram, then disk output."""
    from .frame_store import FrameStore
    if not maps:
        return FrameStore().finish()
    lo=min(float(np.min(d)) for d in maps)
    hi=max(float(np.max(d)) for d in maps)
    if hi<=lo:
        return FrameStore.from_frames(np.zeros_like(d,dtype=np.float32) for d in maps)
    histogram=np.zeros(4096,np.int64)
    for depth in maps:
        histogram+=np.histogram(depth,bins=4096,range=(lo,hi))[0]
    cdf=np.cumsum(histogram)
    quantiles=[lo+(int(np.searchsorted(cdf,cdf[-1]*q))+.5)*(hi-lo)/4096 for q in (.02,.98)]
    low,high=quantiles;scale=max(high-low,1e-6)
    return FrameStore.from_frames(np.clip((d-low)/scale,0,1).astype(np.float32) for d in maps)

def pose_sequence(frames: list[np.ndarray], model: str = "none", checkpoint: str | None = None) -> list[dict]:
    if model in ("none", "disabled"): raise RuntimeError("Pose extraction is not configured. Set pose.model=mediapipe or install/configure DWPose.")
    if model!="mediapipe": raise RuntimeError(f"Unsupported pose adapter {model}; use the optional MediaPipe Holistic adapter or configure DWPose externally.")
    try: import mediapipe as mp
    except ImportError as e: raise RuntimeError("Install mediapipe to use the Holistic body/face/hand pose adapter.") from e
    outputs=[]
    try:
        with mp.solutions.holistic.Holistic(static_image_mode=False,model_complexity=1,enable_segmentation=True) as holistic:
            for frame in frames:
                r=holistic.process(frame)
                def coords(lm): return None if lm is None else [{"x":float(p.x),"y":float(p.y),"z":float(p.z),"visibility":float(getattr(p,"visibility",1.0))} for p in lm.landmark]
                outputs.append({"body":coords(r.pose_landmarks),"face":coords(r.face_landmarks),"left_hand":coords(r.left_hand_landmarks),"right_hand":coords(r.right_hand_landmarks)})
    except AttributeError as e:
        if "MessageFactory" in str(e) or "FieldDescriptor" in str(e):
            raise RuntimeError("MediaPipe Holistic is incompatible with the installed protobuf runtime. Install the optional pose dependencies, which constrain protobuf to <5.") from e
        raise
    return outputs
