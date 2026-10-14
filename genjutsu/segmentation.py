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

    def segment_video(self, frames: list[np.ndarray], frame_index: int, bbox: tuple[int,int,int,int], object_id: int = 1) -> list[np.ndarray]:
        """Seed a video predictor with a box prompt and propagate its predicted masks."""
        import tempfile, cv2
        with tempfile.TemporaryDirectory(prefix="genjutsu_sam2_") as td:
            for i,f in enumerate(frames): cv2.imwrite(str(Path(td)/f"{i:06d}.jpg"),cv2.cvtColor(f,cv2.COLOR_RGB2BGR))
            state=self.predictor.init_state(video_path=td)
            self.predictor.add_new_points_or_box(inference_state=state,frame_idx=frame_index,obj_id=object_id,box=np.array(bbox,dtype=np.float32))
            masks=[np.zeros(frames[0].shape[:2],np.uint8) for _ in frames]
            for idx,ids,logits in self.predictor.propagate_in_video(state):
                for j,obj in enumerate(ids):
                    if obj==object_id: masks[idx]=(logits[j]>0).squeeze().cpu().numpy().astype(np.uint8)*255
            return masks

def save_masks(masks: list[np.ndarray], directory: str | Path) -> list[str]:
    from PIL import Image
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True); out=[]
    for i,m in enumerate(masks):
        p=directory/f"{i:08d}.png"; Image.fromarray((m>0).astype(np.uint8)*255).save(p); out.append(str(p))
    return out
