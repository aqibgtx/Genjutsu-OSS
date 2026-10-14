import numpy as np
from genjutsu.backends.utils import as_rgb8

def test_diffusers_float_frames_convert_to_full_range_uint8():
    values=np.array([[[0.,.5,1.]]],dtype=np.float32)
    result=as_rgb8(values)
    assert result.dtype==np.uint8 and result.tolist()==[[[0,127,255]]]
