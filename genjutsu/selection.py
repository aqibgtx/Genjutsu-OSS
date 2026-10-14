"""Open-vocabulary target selection followed by official SAM 2 video propagation."""
from __future__ import annotations
import numpy as np


def select_and_track(frames: list[np.ndarray], text_prompt: str, detector_dir: str,
                     sam2_config: str, sam2_checkpoint: str, frame_index: int = 0,
                     device: str = "cuda", box_threshold: float = .30,
                     text_threshold: float = .25, detection_index: int = 0, chunk_frames: int = 32):
    """Detect a prompted object at a selected frame, initialize SAM 2 with its box,
    and propagate the resulting object mask through that shot.

    Returns ``(masks, selection_metadata)``. Ambiguous/empty detections fail closed.
    """
    if not frames:
        raise ValueError("Cannot select a target from an empty shot")
    if not text_prompt or not text_prompt.strip():
        raise ValueError("Enter a target description, such as 'person' or 'red bicycle'")
    if not 0 <= frame_index < len(frames):
        raise IndexError("Selection frame index lies outside this shot")
    if not detector_dir or not sam2_config or not sam2_checkpoint:
        raise ValueError("Automatic selection requires local Grounding DINO model directory plus SAM 2 config and checkpoint. Configure selection.detector_model, selection.sam2_config and selection.sam2_checkpoint.")
    from .analysis import grounded_detect
    detections = grounded_detect([frames[frame_index]], text_prompt, detector_dir,box_threshold,text_threshold)[0]["detections"]
    detections = [d for d in detections if d["score"] >= min(box_threshold, 1.)]
    if not detections:
        raise RuntimeError(f"Grounding DINO found no '{text_prompt}' target in shot frame {frame_index}")
    detections.sort(key=lambda item: item["score"], reverse=True)
    if detection_index < 0 or detection_index >= len(detections):
        raise IndexError(f"Detection {detection_index} unavailable; found {len(detections)} candidates")
    selected = detections[detection_index]
    box = tuple(int(round(v)) for v in selected["box"])
    from .segmentation import SAM2Segmenter
    masks = SAM2Segmenter(sam2_config, sam2_checkpoint, device).segment_video(frames, frame_index, box,chunk_frames=chunk_frames)
    if len(masks) != len(frames) or not any(np.any(mask) for mask in masks):
        raise RuntimeError("SAM 2 did not return a non-empty propagated mask sequence")
    return masks, {"prompt": text_prompt, "frame_index": frame_index, "detection": selected,
                   "tracker": "facebookresearch/sam2 video predictor", "mask_frames": len(masks),"chunk_frames":chunk_frames}

def select_face_and_track(frames,sam2_config,sam2_checkpoint,identity_root,frame_index=0,device='cuda',chunk_frames=32):
    """Use real face detection to seed SAM2; avoid editing body/background."""
    from .identity import LocalInsightFace
    from .segmentation import SAM2Segmenter
    if not frames or not 0<=frame_index<len(frames):raise ValueError('Face selection frame is outside the shot')
    detector=LocalInsightFace(root=identity_root,providers=['CPUExecutionProvider'])
    faces=detector.embeddings([frames[frame_index]])[0]
    if not faces:raise RuntimeError('No face detected in source seed frame')
    selected=max(faces,key=lambda face:float((face.bbox[2]-face.bbox[0])*(face.bbox[3]-face.bbox[1])))
    h,w=frames[frame_index].shape[:2]
    x1,y1,x2,y2=selected.bbox
    box=(max(0,int(x1)),max(0,int(y1)),min(w,int(np.ceil(x2))),min(h,int(np.ceil(y2))))
    tracked=SAM2Segmenter(sam2_config,sam2_checkpoint,device).segment_video(frames,frame_index,box,chunk_frames=chunk_frames)
    # A face-sized box can still make SAM2 select the whole person. Restrict
    # each propagated mask to the measured, reidentified face region.
    from .frame_store import FrameStore
    from .identity import cosine_similarity
    import cv2
    masks=FrameStore()
    for index,frame in enumerate(frames):
        candidates=detector.embeddings([frame])[0]
        candidates=[face for face in candidates if cosine_similarity(selected.normed_embedding,face.normed_embedding)>=.25]
        region=np.zeros((h,w),np.uint8)
        if candidates:
            face=max(candidates,key=lambda face:cosine_similarity(selected.normed_embedding,face.normed_embedding))
            a,b,c,d=face.bbox
            a,b,c,d=max(0,int(a)),max(0,int(b)),min(w,int(np.ceil(c))),min(h,int(np.ceil(d)))
            cv2.ellipse(region,((a+c)//2,(b+d)//2),(max(1,(c-a)//2),max(1,(d-b)//2)),0,0,360,255,-1)
        masks.append(np.minimum(tracked[index],region))
    masks=masks.finish()
    if not any(np.any(mask) for mask in masks):raise RuntimeError('SAM2 returned empty face masks')
    return masks,{'prompt':'face','box':list(box),'frame_index':frame_index,'method':'local face detector -> official SAM2','mask_frames':len(masks)}
