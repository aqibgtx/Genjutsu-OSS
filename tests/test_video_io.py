import numpy as np
import subprocess
from genjutsu.analysis import estimate_camera, optical_flow, scene_cuts
from genjutsu.video import probe_video, read_frames, write_video

def test_ffmpeg_round_trip_preserves_fps_and_frame_count(tmp_path):
    frames=[]
    for i in range(9):
        image=np.zeros((32,48,3),np.uint8); image[:,i:i+8]=[220,40,20]; frames.append(image)
    source=tmp_path/"source.mp4"; write_video(frames,source,12,codec="h264")
    info=probe_video(source); decoded,again=read_frames(source)
    assert info.frame_count==9 and abs(info.fps-12)<.01
    assert len(decoded)==9 and decoded[0].shape==(32,48,3) and again.has_audio is False

def test_video_understanding_returns_measured_outputs(tmp_path):
    frames=[np.full((32,32,3),10,np.uint8) for _ in range(3)]
    assert scene_cuts(frames)==[]
    forward,backward=optical_flow(frames)
    assert len(forward)==len(backward)==2 and np.isfinite(forward[0]).all()
    assert len(estimate_camera(frames)["trajectory"])==2

def test_supported_output_codecs(tmp_path):
    frames=[np.full((32,48,3),value,np.uint8) for value in (20,80,160,220,120)]
    for codec,extension in (("h265","mp4"),("prores","mov"),("vp9","webm")):
        output=tmp_path/f"codec.{extension}"
        write_video(frames,output,12,codec=codec)
        info=probe_video(output)
        assert info.frame_count==len(frames) and info.codec


def test_repeated_tiny_h265_exports_avoid_native_heap_crash(tmp_path):
    frames=[np.full((32,48,3),value,np.uint8) for value in (20,80,160,220,120)]
    for index in range(8):
        output=tmp_path/f"tiny{index}.mp4"
        write_video(frames,output,12,codec="h265")
        assert probe_video(output).frame_count==5

def test_audio_track_is_preserved_and_reencoded(tmp_path):
    frames=[np.full((32,48,3),80,np.uint8) for _ in range(9)]
    silent=tmp_path/"silent.mp4"; write_video(frames,silent,12)
    tone=tmp_path/"tone.wav"
    subprocess.run(["ffmpeg","-y","-v","error","-f","lavfi","-i","sine=frequency=440:duration=1","-c:a","pcm_s16le",str(tone)],check=True,capture_output=True)
    source=tmp_path/"with_audio.mp4"
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(silent),"-i",str(tone),"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-shortest",str(source)],check=True,capture_output=True)
    output=tmp_path/"preserved.mp4"; write_video(frames,output,12,audio_from=source)
    info=probe_video(output)
    assert info.has_audio and info.audio_metadata["codec_name"]=="aac"
