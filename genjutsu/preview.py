"""Cheap real media previews. Resize before model inference, preserving PTS."""
from pathlib import Path
from fractions import Fraction
import json
from PIL import Image
import numpy as np
from .video import probe_video, iter_frames, write_video

DRAFT = {"width": 144, "height": 256, "frames": 5, "steps": 1}

def prepare_preview(source, directory, *, width=144, height=256, frames=5):
    if min(width,height)<16 or width%16 or height%16 or frames<1:
        raise ValueError("Preview dimensions must be positive multiples of 16; frame limit must be positive")
    source=Path(source).resolve(); directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    info=probe_video(source); count=min(frames,info.frame_count)
    # Fit to the requested bounding box, preserving aspect ratio to within one
    # 16-pixel model grid cell. No upscaling is performed.
    scale=min(width/info.width,height/info.height,1.)
    size=(max(16,int(info.width*scale)//16*16),max(16,int(info.height*scale)//16*16))
    output=directory/"draft_source.mp4"
    tb=Fraction(info.time_base)
    from .frame_store import FrameStore
    sequence=FrameStore.from_frames(np.asarray(Image.fromarray(frame).resize(size,Image.Resampling.LANCZOS)) for frame in iter_frames(source,limit=count))
    write_video(sequence,output,info.fps_rational,audio_from=source if info.has_audio else None,
                timestamps=info.timestamps[:count],frame_durations=[float(d*tb) for d in info.frame_durations[:count]],
                time_base=info.time_base,audio_start_time=info.start_time)
    actual=probe_video(output)
    if actual.frame_count!=count:raise RuntimeError("Draft extraction changed the selected frame count")
    (directory/"preview.json").write_text(json.dumps({"source":str(source),"preview":str(output),"size":list(size),
        "frame_count":count,"source_timestamps":info.timestamps[:count],"purpose":"lowest-cost wiring preview, not identity-quality acceptance"},indent=2))
    return output
