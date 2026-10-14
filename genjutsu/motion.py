from .analysis import optical_flow, estimate_camera

def chunk_ranges(frame_count: int, chunk_size: int, overlap: int) -> list[tuple[int,int]]:
    if chunk_size<1 or overlap<0 or overlap>=chunk_size: raise ValueError("Require chunk_size > overlap >= 0")
    step=chunk_size-overlap; ranges=[]; start=0
    while start<frame_count:
        end=min(start+chunk_size,frame_count); ranges.append((start,end))
        if end==frame_count: break
        start+=step
    return ranges

def stitch_chunks(chunks: list[list], overlap: int) -> list:
    result=[]
    for i,chunk in enumerate(chunks): result.extend(chunk if i==0 else chunk[min(overlap,len(chunk)):])
    return result

def scene_chunk_ranges(frame_count:int,chunk_size:int,overlap:int,cuts:list[int]|None=None)->list[tuple[int,int,int]]:
    """Return absolute (start,end,trim_start) ranges with no chunk crossing a detected shot cut."""
    boundaries=[0]+sorted({int(c) for c in (cuts or []) if 0<int(c)<frame_count})+[frame_count]
    result=[]
    for shot_start,shot_end in zip(boundaries,boundaries[1:]):
        local=chunk_ranges(shot_end-shot_start,chunk_size,overlap)
        for index,(a,b) in enumerate(local):result.append((shot_start+a,shot_start+b,0 if index==0 else overlap))
    return result
