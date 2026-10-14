"""CPU media fixtures; the test backend performs grayscale conversion, never AI."""
from copy import deepcopy
import numpy as np
import pytest
import yaml

from genjutsu.frame_store import FrameStore


class CPUGrayscaleBackend:
    name = "test-only-cpu-grayscale"

    def __init__(self):
        self.calls = []
        self.fail_once = set()
        self.failed = set()

    def generate(self, controls, frames, output_dir):
        self.calls.append(deepcopy(controls))
        index = controls.settings["shot_index"]
        if index in self.fail_once and index not in self.failed:
            self.failed.add(index)
            raise RuntimeError("test-only transient error")
        import cv2
        return FrameStore.from_frames(cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB) for frame in frames)


@pytest.fixture
def cpu_backend():
    return CPUGrayscaleBackend()


@pytest.fixture
def local_config(tmp_path):
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump({"backend": {"type": "comfyui"}, "temporal": {"enabled": True, "strength": .15},
        "compositing": {"enabled": True, "feather_px": 0}, "cache": {"enabled": True, "directory": str(tmp_path/"cache")}}))
    return path


@pytest.fixture
def multishot_source(tmp_path):
    """Independent FFmpeg-generated VFR fixture: three shots, AAC with offset."""
    import subprocess
    from genjutsu.video import write_video
    silent = tmp_path/"cfr.mp4"
    frames=[]
    for index in range(18):
        frame=np.zeros((48,64,3),np.uint8)
        frame[:]=[(30,40,150),(20,140,40),(160,30,20)][index//6]
        x=5+2*(index%6)
        frame[15:30,x:x+12]=[225,220,30]
        if index%6>=3:frame[15:30,x+5:x+12]=[220,20,20]
        frames.append(frame)
    write_video(frames,silent,"20/1")
    source=tmp_path/"vfr-audio.mp4"
    # Alternating 50/100 ms spacing, independent of our writer's PTS support.
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(silent),"-itsoffset","0.2","-f","lavfi","-i","sine=frequency=880:sample_rate=48000:duration=1.5",
        "-map","0:v:0","-map","1:a:0","-vf","settb=1/1000,setpts='(N+floor(N/2))*50'","-fps_mode","passthrough","-enc_time_base","1/1000",
        "-c:v","libx264","-c:a","aac",str(source)],check=True,capture_output=True)
    return source
