"""Numeric SDK mask compatibility tests; these do not simulate AI inference."""
import numpy as np
import pytest
from genjutsu.backends.vace import repair_reference_mask_padding

@pytest.mark.parametrize('frames,refs',[(5,0),(5,1),(5,2),(5,6),(5,13),(17,6),(81,6)])
def test_reference_mask_has_one_zero_latent_frame_per_reference(frames,refs):
    base=(frames+3)//4
    values=np.arange(64*base*2*4,dtype=np.float32).reshape(1,64,base,2,4)
    prefix=np.zeros((1,64,min(base,refs),2,4),np.float32)
    upstream=np.concatenate([prefix,values],axis=2)
    corrected=repair_reference_mask_padding(upstream,frames,refs)
    assert corrected.shape==(1,64,base+refs,2,4)
    assert np.array_equal(corrected[:,:,refs:],values)
    assert np.all(corrected[:,:,:refs]==0)
    assert corrected.dtype==upstream.dtype

def test_fixed_upstream_is_not_modified():
    correct=np.ones((1,64,8,2,4),np.float32)
    assert repair_reference_mask_padding(correct,5,6) is correct

def test_unknown_tensor_mismatch_is_not_hidden():
    with pytest.raises(ValueError,match='Unsupported'):
        repair_reference_mask_padding(np.zeros((1,64,3,2,4)),5,6)
