"""Mask utilities and SAM 2 adapter. Segmentation is never guessed from a text prompt."""
from __future__ import annotations
from pathlib import Path
import numpy as np

def mask_from_bbox(shape: tuple[int, int], bbox: tuple[int,int,int,int]) -> np.ndarray:
    h,w=shape; x1,y1,x2,y2=map(int,bbox)
    if not (0<=x1<x2<=w and 0<=y1<y2<=h): raise ValueError("bbox must be x1,y1,x2,y2 inside frame")
    out=np.zeros((h,w),np.uint8); out[y1:y2,x1:x2]=255; return out

def clean_mask(mask: np.ndarray, kernel: int = 3) -> np.ndarray:
    cv2=_cv2(); m=(mask>0).astype(np.uint8)*255
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel|1,kernel|1))
    return cv2.morphologyEx(cv2.morphologyEx(m,cv2.MORPH_CLOSE,k),cv2.MORPH_OPEN,k)

def _cv2():
    try: import cv2; return cv2
    except ImportError as e: raise RuntimeError("OpenCV required; install genjutsu-oss[video].") from e

class SAM2Segmenter:
    """Uses the official facebook/sam2 builder API with a local checkpoint and config."""
    def __init__(self, config: str, checkpoint: str, device: str = "cuda"):
        try: from sam2.build_sam import build_sam2_video_predictor
        except ImportError as e: raise RuntimeError("Install the official SAM 2 package (facebookresearch/sam2) in the runtime environment.") from e
        if not Path(checkpoint).is_file(): raise FileNotFoundError(f"SAM 2 checkpoint not found: {checkpoint}")
        self.predictor=build_sam2_video_predictor(config,checkpoint,device=device)

    def segment_video(self, frames, frame_index: int, bbox: tuple[int,int,int,int], object_id: int = 1, chunk_frames: int = 32):
        """Bound SAM's host/state memory with overlapping, mask-seeded windows.

        The seed window propagates in both directions. Adjacent windows overlap
        one frame and receive the previous real SAM mask through add_new_mask.
        Fully lost targets at a window boundary require user reinitialization.
        """
        import tempfile, cv2
        from .frame_store import FrameStore
        if not frames or not 0<=frame_index<len(frames):raise ValueError("SAM seed frame is outside the shot")
        if chunk_frames<2:raise ValueError("SAM chunk_frames must be at least 2")
        zero=np.zeros(frames[0].shape[:2],np.uint8)
        masks=FrameStore.from_frames(zero for _ in range(len(frames)))
        def window(start,end,seed,box=None,mask=None,directions=(False,)):
            with tempfile.TemporaryDirectory(prefix="genjutsu_sam2_") as td:
                for i in range(start,end):
                    cv2.imwrite(str(Path(td)/f"{i-start:06d}.jpg"),cv2.cvtColor(frames[i],cv2.COLOR_RGB2BGR))
                state=self.predictor.init_state(video_path=td,offload_video_to_cpu=True,offload_state_to_cpu=True)
                if box is not None:
                    self.predictor.add_new_points_or_box(inference_state=state,frame_idx=seed,obj_id=object_id,box=np.array(box,dtype=np.float32))
                else:
                    if not np.any(mask):raise RuntimeError("SAM target lost at chunk boundary; reinitialize the target or shorten selection.chunk_frames")
                    self.predictor.add_new_mask(inference_state=state,frame_idx=seed,obj_id=object_id,mask=mask>0)
                for reverse in directions:
                    for idx,ids,logits in self.predictor.propagate_in_video(state,start_frame_idx=seed,reverse=reverse):
                        for j,obj in enumerate(ids):
                            if obj==object_id:masks.set(start+idx,(logits[j]>0).squeeze().cpu().numpy().astype(np.uint8)*255)
                del state
        start=max(0,frame_index-chunk_frames//2);end=min(len(frames),start+chunk_frames)
        window(start,end,frame_index-start,box=bbox,directions=(False,True))
        anchor_start,anchor_end=start,end
        while end<len(frames):
            start=end-1;end=min(len(frames),start+chunk_frames)
            window(start,end,0,mask=masks[start])
        start=anchor_start
        while start>0:
            end=start+1;start=max(0,end-chunk_frames)
            window(start,end,end-start-1,mask=masks[end-1],directions=(True,))
        return masks.finish()

def save_masks(masks: list[np.ndarray], directory: str | Path) -> list[str]:
    from PIL import Image
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True); out=[]
    for i,m in enumerate(masks):
        p=directory/f"{i:08d}.png"; Image.fromarray((m>0).astype(np.uint8)*255).save(p); out.append(str(p))
    return out
