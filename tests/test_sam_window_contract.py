"""Window management contract only; a test double does not validate SAM inference."""
from pathlib import Path
import numpy as np
from PIL import Image

from genjutsu.frame_store import FrameStore
from genjutsu.segmentation import SAM2Segmenter


class TensorContract:
    def __init__(self,array):self.array=array
    def __gt__(self,value):return TensorContract(self.array>value)
    def squeeze(self):return TensorContract(self.array.squeeze())
    def cpu(self):return self
    def numpy(self):return self.array


class PredictorContract:
    def __init__(self):self.windows=[];self.mask_seeds=0;self.reverse_calls=0
    def init_state(self,video_path,offload_video_to_cpu,offload_state_to_cpu):
        paths=sorted(Path(video_path).glob("*.jpg"));self.windows.append(len(paths))
        assert offload_video_to_cpu and offload_state_to_cpu
        with Image.open(paths[0]) as image:shape=(image.height,image.width)
        return {"count":len(paths),"shape":shape}
    def add_new_points_or_box(self,inference_state,frame_idx,obj_id,box):
        inference_state["id"]=obj_id
    def add_new_mask(self,inference_state,frame_idx,obj_id,mask):
        assert mask.dtype==bool and mask.any()
        self.mask_seeds+=1;inference_state["id"]=obj_id
    def propagate_in_video(self,inference_state,start_frame_idx,reverse):
        self.reverse_calls+=int(reverse)
        indices=range(start_frame_idx,-1,-1) if reverse else range(start_frame_idx,inference_state["count"])
        for index in indices:
            yield index,[inference_state["id"]],[TensorContract(np.ones(inference_state["shape"],np.float32))]


def test_sam_windows_bound_state_reseed_masks_and_cover_before_seed():
    segmenter=object.__new__(SAM2Segmenter);segmenter.predictor=PredictorContract()
    frames=FrameStore.from_frames(np.full((16,32,3),60,np.uint8) for _ in range(80))
    masks=segmenter.segment_video(frames,20,(2,2,12,12),chunk_frames=12)
    assert isinstance(masks,FrameStore) and len(masks)==80
    assert max(segmenter.predictor.windows)<=12
    assert segmenter.predictor.mask_seeds>=5 and segmenter.predictor.reverse_calls>=2
    assert all(mask.shape==(16,32) and mask.any() for mask in masks)
