"""Measure native + Python RSS in isolated workers on real 1080p media."""
import json
import subprocess
import sys


WORKER = r'''
import json, sys, threading, time
import psutil
from genjutsu.video import read_frames, write_video, timing_kwargs
from genjutsu.frame_store import FrameStore, FrameView
process=psutil.Process()
baseline=process.memory_info().rss
peak=[baseline]
stop=threading.Event()
def sample():
    while not stop.wait(.005):
        peak[0]=max(peak[0],process.memory_info().rss)
thread=threading.Thread(target=sample);thread.start()
try:
    frames,info=read_frames(sys.argv[1],disk_backed=True)
    assert isinstance(frames,FrameStore) and isinstance(frames[:],FrameView)
    for frame in frames:
        assert frame.shape==(1080,1920,3)
    write_video(frames,sys.argv[2],info.fps_rational,**timing_kwargs(info))
finally:
    stop.set();thread.join()
print(json.dumps({"frames":len(frames),"peak_delta_mib":(peak[0]-baseline)/2**20}))
'''


def test_1080p_decode_slice_and_encode_rss_is_bounded(tmp_path):
    measurements=[]
    for count in [16,96]:
        source=tmp_path/f"input{count}.mp4"
        subprocess.run(["ffmpeg","-y","-v","error","-f","lavfi","-i","color=c=blue:s=1920x1080:r=24","-frames:v",str(count),
            "-c:v","libx264","-preset","ultrafast","-threads","1",str(source)],check=True,capture_output=True)
        result=subprocess.run([sys.executable,"-c",WORKER,str(source),str(tmp_path/f"output{count}.mp4")],check=True,capture_output=True,text=True)
        measured=json.loads(result.stdout.strip().splitlines()[-1]);measurements.append(measured)
        assert measured["frames"]==count
        assert measured["peak_delta_mib"]<230,measurements
    # Eager 96-frame RGB alone is ~570 MiB. Adding 80 frames must not add that.
    assert measurements[1]["peak_delta_mib"]-measurements[0]["peak_delta_mib"]<64,measurements
    print("1080p RSS measurements:",json.dumps(measurements))
