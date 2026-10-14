import numpy as np
from genjutsu.compositing import composite, soft_mask
from genjutsu.identity import cosine_similarity
from genjutsu.metrics import compare_videos
from genjutsu.motion import chunk_ranges, stitch_chunks, scene_chunk_ranges
from genjutsu.occlusion import visibility_mask
from genjutsu.segmentation import mask_from_bbox
from genjutsu.temporal import flicker_score
from genjutsu.temporal import stabilize_generated

def test_bbox_mask_and_compositor_preserve_outside():
    mask=mask_from_bbox((20,30),(5,4,15,14))
    assert mask.sum()==100*255
    source=np.zeros((20,30,3),np.uint8); generated=np.full_like(source,255)
    out=composite(source,generated,mask,feather_px=0)
    assert np.all(out[0,0]==0) and np.all(out[8,8]==255)

def test_soft_mask_stays_in_unit_interval():
    m=soft_mask(mask_from_bbox((32,32),(8,8,24,24)),4)
    assert m.min()>=0 and m.max()<=1 and 0<m[7,16]<1

def test_local_color_match_uses_preserved_ring_without_flattening_subject():
    source=np.full((32,32,3),40,np.uint8)
    generated=np.full_like(source,80);generated[10:22,10:22]=160
    mask=mask_from_bbox((32,32),(10,10,22,22))
    out=composite(source,generated,mask,feather_px=0,color_match_enabled=True)
    assert np.all(out[0,0]==40)
    assert np.all((out[16,16]>=118)&(out[16,16]<=122))

def test_flow_chunk_ranges_and_overlap_stitch():
    ranges=chunk_ranges(100,41,5)
    assert ranges==[(0,41),(36,77),(72,100)]
    chunks=[list(range(a,b)) for a,b in ranges]
    assert len(stitch_chunks(chunks,5))==100
    shot_ranges=scene_chunk_ranges(100,41,5,[50])
    assert all(not (start<50<end) for start,end,_ in shot_ranges)
    assert sum(end-start-trim for start,end,trim in shot_ranges)==100

def test_occlusion_mask_hides_farther_edit_layer():
    edit=np.full((3,3),255,np.uint8); edit_depth=np.full((3,3),.8,np.float32)
    foreground_depth=np.full((3,3),.2,np.float32); foreground=np.zeros((3,3),np.uint8); foreground[1,1]=255
    out=visibility_mask(edit,edit_depth,foreground_depth,foreground)
    assert out[1,1]==0 and out[0,0]==255

def test_identity_and_temporal_metrics_are_numeric():
    assert cosine_similarity([1,0],[1,0])==1
    assert flicker_score([np.zeros((2,2,3),np.uint8)]*2)==0
    scores=compare_videos([np.zeros((2,2,3),np.uint8)]*2,[np.zeros((2,2,3),np.uint8)]*2)
    assert scores["temporal_consistency"]>0.99
    assert scores["identity_similarity"] is None

def test_temporal_stabilization_resets_at_shot_cuts():
    first=np.zeros((16,16,3),np.uint8);second=np.full_like(first,255)
    result=stabilize_generated([first,second],strength=1.,scene_cuts=[1])
    assert np.array_equal(result[1],second)
