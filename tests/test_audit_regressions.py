"""Regression tests for uploads, shot isolation, content identity, and real media."""
from copy import deepcopy
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest

from genjutsu.cache import backend_signature, content_identity, generation_key, model_signature
from genjutsu.controls import ControlPackage, Reference
from genjutsu.frame_store import FrameStore, FrameView
from genjutsu.pipeline import _read_mask, analyze_video, transform_video
from genjutsu.shots import run_shots, slice_controls, shot_ranges
from genjutsu.ui import mask_upload, upload_path
from genjutsu.video import iter_frames, probe_video, read_frames, timing_kwargs, write_video


@pytest.mark.parametrize("wrapper",[lambda p:p, lambda p:Path(p), lambda p:{"path":p}, lambda p:{"name":p},lambda p:SimpleNamespace(path=p),lambda p:SimpleNamespace(name=p)])
def test_upload_representations_and_single_mask(wrapper,tmp_path):
    image=tmp_path/"mask.png";Image.fromarray(np.full((8,8),255,np.uint8)).save(image)
    assert upload_path(wrapper(str(image)))==str(image)
    assert mask_upload([wrapper(str(image))])==str(image)
    masks=_read_mask(mask_upload([wrapper(str(image))]),3,(8,8))
    assert len(masks)==3 and all(np.all(m==255) for m in masks)


def test_multiple_masks_sorted_by_filename_not_upload_directory(tmp_path):
    paths=[]
    for directory,name,value in [("z","frame1.png",1),("a","frame10.png",10),("x","frame2.png",2)]:
        folder=tmp_path/directory;folder.mkdir();path=folder/name
        Image.fromarray(np.full((8,8),value,np.uint8)).save(path);paths.append(str(path))
    masks=_read_mask(mask_upload(paths),3,(8,8))
    assert [int(m[0,0]) for m in masks]==[1,2,10]
    assert mask_upload(None) is None
    with pytest.raises(ValueError,match="one mask video"):
        mask_upload([paths[0],str(tmp_path/"mask.mp4")])
    with pytest.raises(ValueError,match="one mask video"):
        mask_upload([str(tmp_path/"mask.mp4"),str(tmp_path/"mask2.mp4")])
    with pytest.raises(ValueError,match="missing a file path"):
        upload_path({})


def test_mask_video_upload_checks_count_and_dimensions(tmp_path):
    path=tmp_path/"mask.mp4"
    write_video([np.full((16,16,3),v,np.uint8) for v in [0,255,0]],path,12)
    masks=_read_mask(mask_upload([{"path":str(path)}]),3,(16,16))
    assert isinstance(masks,FrameStore) and int(masks[1].mean())>250
    with pytest.raises(ValueError,match="source has 4"):_read_mask(path,4,(16,16))
    with pytest.raises(ValueError,match="dimensions"):_read_mask(path,3,(8,8))


def test_shot_temporal_controls_are_local_and_do_not_cross_cuts(tmp_path):
    count=8
    paths=[]
    for i in range(count):
        p=tmp_path/f"{i:08d}.png";Image.fromarray(np.full((8,8),i,np.uint8)).save(p);paths.append(str(p))
    c=ControlPackage("source",fps=8,width=8,height=8,frame_count=count,duration=1.0,time_base="1/1000",
        frame_pts=[0,50,150,200,400,500,800,900],frame_durations=[50,100,50,200,100,300,100,100],is_vfr=True,
        source_start_time=1.25,scene_cuts=[4],object_masks=[str(tmp_path)],subject_masks=paths,preservation_masks=paths,depth_sequence=paths,
        pose_sequence=[{"pose":i} for i in range(count)],camera_motion={"method":"measured","trajectory":[{"dx":i} for i in range(count-1)]},
        tracking_data={"observations":[{"frame":i,"timestamp":c_time,"box":[i,1,2,3]} for i,c_time in enumerate([0,.05,.15,.2,.4,.5,.8,.9])],"by_frame":{str(i):[i] for i in range(count)}},
        optical_flow={"forward":[f"fw{i}" for i in range(count-1)],"backward":[f"bw{i}" for i in range(count-1)],"method":"measured"})
    original=deepcopy(c.to_dict());first=slice_controls(c,0,4);last=slice_controls(c,4,8)
    assert first.duration==pytest.approx(.4) and last.duration==pytest.approx(.6)
    assert last.frame_indices==[0,1,2,3] and last.source_frame_indices==[4,5,6,7]
    assert last.timestamps==pytest.approx([0,.1,.4,.5]) and last.frame_pts==[0,100,400,500]
    assert last.source_start_time==pytest.approx(1.65)
    assert last.camera_motion["trajectory"]==[{"dx":4},{"dx":5},{"dx":6}]
    assert last.optical_flow["forward"]==["fw4","fw5","fw6"]
    assert last.pose_sequence==[{"pose":i} for i in range(4,8)]
    assert last.subject_masks==last.preservation_masks==last.depth_sequence==paths[4:]
    assert last.object_masks==paths[4:]
    assert last.tracking_data["observations"][0]["frame"]==0
    assert last.tracking_data["observations"][0]["timestamp"]==0
    assert last.tracking_data["observations"][0]["source_frame_index"]==4
    assert last.tracking_data["by_frame"]=={str(i):[i+4] for i in range(4)}
    assert not last.scene_cuts and c.to_dict()==original
    assert shot_ranges(0,[])==[]


@pytest.mark.parametrize("fps",["24000/1001","30000/1001","60000/1001"])
def test_rational_rate_round_trip_without_float_drift(tmp_path,fps):
    path=tmp_path/"fractional.mp4"
    write_video([np.full((16,32,3),i,np.uint8) for i in range(31)],path,fps)
    info=probe_video(path)
    assert Fraction(info.fps_rational)==Fraction(fps)
    assert not info.is_vfr and info.frame_count==31
    assert info.duration==pytest.approx(float(31/Fraction(fps)),abs=float(Fraction(info.time_base)))


def test_independently_generated_vfr_round_trip_preserves_pts_and_audio(multishot_source,tmp_path):
    info=probe_video(multishot_source)
    assert info.is_vfr and len(set(np.diff(info.timestamps)))>1
    frames,_=read_frames(multishot_source,disk_backed=True)
    target=tmp_path/"roundtrip.mp4"
    write_video(frames,target,info.fps_rational,audio_from=multishot_source,**timing_kwargs(info))
    actual=probe_video(target)
    assert actual.is_vfr and actual.frame_count==info.frame_count
    assert actual.timestamps==pytest.approx(info.timestamps,abs=float(Fraction(info.time_base)))
    assert actual.duration==pytest.approx(info.duration,abs=.001)
    assert abs(float(actual.audio_metadata["start_time"])-float(info.audio_metadata["start_time"]))<.025
    assert len(list(iter_frames(target,start=4,limit=3)))==3
    assert list(iter_frames(target,limit=0))==[]
    def samples(path):
        decoded=subprocess.run(["ffmpeg","-v","error","-copyts","-i",str(path),"-map","0:a:0","-af","aresample=async=1:first_pts=0","-ac","1","-ar","48000","-f","f32le","-"],check=True,capture_output=True)
        return np.frombuffer(decoded.stdout,np.float32)
    a,b=samples(multishot_source),samples(target)
    assert np.mean(np.abs(a[:7000]))<.001 and np.mean(np.abs(b[:7000]))<.001
    n=1 << (len(a)+len(b)-1).bit_length()
    correlation=np.fft.irfft(np.fft.rfft(a,n)*np.conj(np.fft.rfft(b,n)),n)
    lag=int(np.argmax(correlation));lag=lag if lag<n//2 else lag-n
    assert abs(lag)<=1024 # AAC packet tolerance (21.3 ms at 48 kHz), no cumulative drift.


def test_nonzero_source_video_start_preserves_relative_audio_offset(multishot_source,tmp_path):
    shifted=tmp_path/"offset.mp4"
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(multishot_source),"-c","copy","-output_ts_offset","2",str(shifted)],check=True,capture_output=True)
    frames,info=read_frames(shifted,disk_backed=True)
    assert info.start_time>=1.99
    target=tmp_path/"normalized.mp4"
    write_video(frames,target,info.fps_rational,audio_from=shifted,**timing_kwargs(info))
    actual=probe_video(target)
    assert actual.timestamps==pytest.approx(info.timestamps,abs=.001)
    assert actual.duration==pytest.approx(info.duration,abs=.001)
    assert float(actual.audio_metadata["start_time"])==pytest.approx(float(info.audio_metadata["start_time"])-info.start_time,abs=.025)


@pytest.mark.parametrize("codec,extension",[("h265","mp4"),("prores","mov"),("vp9","webm")])
def test_vfr_export_codecs_do_not_drop_or_duplicate_frames(multishot_source,tmp_path,codec,extension):
    frames,info=read_frames(multishot_source,disk_backed=True)
    target=tmp_path/f"vfr.{extension}"
    write_video(frames,target,info.fps_rational,audio_from=multishot_source,codec=codec,**timing_kwargs(info))
    actual=probe_video(target)
    tolerance=max(float(Fraction(info.time_base)),float(Fraction(actual.time_base)))
    assert actual.frame_count==info.frame_count and actual.has_audio
    assert actual.timestamps==pytest.approx(info.timestamps,abs=tolerance)
    assert actual.duration==pytest.approx(info.duration,abs=tolerance+.001)


def test_content_hash_invalidates_same_size_same_mtime_model_and_reference(tmp_path,cpu_backend):
    c=ControlPackage("unused",fps=12,width=8,height=8,frame_count=1)
    model=tmp_path/"model";model.mkdir();weights=model/"weights.bin";weights.write_bytes(b"revision-one")
    ref=tmp_path/"ref.png";Image.fromarray(np.zeros((8,8,3),np.uint8)).save(ref)
    c.references.references=[Reference(str(ref),"OBJECT",.5)]
    cpu_backend.model_id=str(model);cpu_backend.revision="rev1"
    frames=[np.zeros((8,8,3),np.uint8)]
    signature=backend_signature(cpu_backend);first=generation_key(c,frames,None,signature)
    stat=weights.stat();weights.write_bytes(b"revision-two");os.utime(weights,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    second=generation_key(c,frames,None,backend_signature(cpu_backend));assert second!=first
    original_ref=ref.read_bytes();before=ref.stat();ref.write_bytes(original_ref[:-1]+bytes([original_ref[-1]^1]));os.utime(ref,ns=(before.st_atime_ns,before.st_mtime_ns))
    assert generation_key(c,frames,None,signature)!=first
    changed=deepcopy(signature);changed["software"]["packages"]["diffusers"]="another-version"
    assert generation_key(c,frames,None,changed)!=generation_key(c,frames,None,signature)
    other=deepcopy(c);other.temporal_settings["strength"]+=.1
    assert generation_key(other,frames,None,signature)!=generation_key(c,frames,None,signature)
    other=deepcopy(c);other.identity_strength+=.1
    assert generation_key(other,frames,None,signature)!=generation_key(c,frames,None,signature)
    assert model_signature(str(tmp_path/"missing"))["cache_safe"] is False


def test_multishot_retry_cache_resume_and_real_cpu_processing(multishot_source,tmp_path,cpu_backend):
    c,frames=analyze_video(multishot_source)
    assert isinstance(frames,FrameStore) and isinstance(frames[2:5],FrameView)
    masks=FrameStore.from_frames(np.full((c.height,c.width),255,np.uint8) for _ in frames)
    assert c.scene_cuts==[6,12]
    cpu_backend.fail_once={1}
    out,states,_=run_shots(frames,c.scene_cuts,masks,c,c.references,cpu_backend,tmp_path/"work",{"1":{"prompt":"second shot","settings":{"seed":44}}},retries=1)
    assert [s["attempts"] for s in states]==[1,2,1] and len(out)==len(frames)
    assert all(call.frame_count==6 and len(call.timestamps)==6 for call in cpu_backend.calls)
    assert cpu_backend.calls[1].prompt=="second shot"
    assert out[8][20,20,0]==out[8][20,20,1]
    calls=len(cpu_backend.calls)
    _,cached,_=run_shots(frames,c.scene_cuts,masks,c,c.references,cpu_backend,tmp_path/"work",{"1":{"prompt":"second shot","settings":{"seed":44}}},retries=1)
    assert all(s["status"]=="cached" for s in cached) and len(cpu_backend.calls)==calls
    # A missing frame invalidates only that shot and is regenerated.
    (Path(cached[1]["output_path"])/"00000002.png").unlink()
    _,recovered,_=run_shots(frames,c.scene_cuts,masks,c,c.references,cpu_backend,tmp_path/"work",{"1":{"prompt":"second shot","settings":{"seed":44}}},retries=1)
    assert [s["status"] for s in recovered]==["cached","complete","cached"]


def test_high_level_vfr_multishot_pipeline_composites_preserves_audio_and_caches(multishot_source,tmp_path,cpu_backend,local_config,monkeypatch):
    monkeypatch.setattr("genjutsu.pipeline.create_backend",lambda config:cpu_backend)
    mask=tmp_path/"mask.png";m=np.zeros((48,64),np.uint8);m[12:34,3:30]=255;Image.fromarray(m).save(mask)
    result=transform_video(multishot_source,tmp_path/"result.mp4","object_swap","gray",edit_mask=[str(mask)],config_path=local_config,shots={"2":{"prompt":"last shot"}})
    info=probe_video(multishot_source);actual=probe_video(result)
    assert actual.timestamps==pytest.approx(info.timestamps,abs=.001)
    assert actual.has_audio and actual.duration==pytest.approx(info.duration,abs=.001)
    original=list(iter_frames(multishot_source));decoded=list(iter_frames(result))
    outside=m==0
    assert np.mean([np.abs(a[outside].astype(float)-b[outside]).mean() for a,b in zip(original,decoded)])<5
    assert len(cpu_backend.calls)==3 and cpu_backend.calls[2].prompt=="last shot"
    transform_video(multishot_source,result,"object_swap","gray",edit_mask=[str(mask)],config_path=local_config,shots={"2":{"prompt":"last shot"}})
    assert len(cpu_backend.calls)==3
    transform_video(multishot_source,result,"object_swap","gray",edit_mask=[str(mask)],config_path=local_config,shots={"2":{"prompt":"different last shot"}})
    assert len(cpu_backend.calls)==4 and cpu_backend.calls[-1].settings["shot_index"]==2
    run=json.loads((tmp_path/"result.work"/"run.json").read_text())
    assert run["timing_policy"]=="preserve_source_pts" and run["frame_storage"]=="disk_backed"


def test_shared_depth_normalization_uses_bounded_histogram(tmp_path):
    from genjutsu.analysis import normalize_depth_store
    arrays=[np.linspace(i,i+1,128,dtype=np.float32).reshape(8,16) for i in range(3)]
    maps=FrameStore.from_frames(arrays)
    normalized=normalize_depth_store(maps)
    values=np.concatenate([a.ravel() for a in arrays]);lo,hi=np.percentile(values,[2,98])
    assert isinstance(normalized,FrameStore)
    for a,b in zip(arrays,normalized):
        assert b==pytest.approx(np.clip((a-lo)/(hi-lo),0,1),abs=.005)


def test_wan_animate_rejects_vfr_before_model_import_or_load():
    from genjutsu.backends.wan import Wan22Backend
    c=ControlPackage("source",is_vfr=True)
    with pytest.raises(ValueError,match="CFR"):
        Wan22Backend({"mode":"animate"}).generate(c,[np.zeros((16,16,3),np.uint8)],"unused")


def test_file_backed_flow_controls_slice_without_cross_shot_transition(tmp_path):
    forward=np.arange(7*4*6*2,dtype=np.float32).reshape(7,4,6,2)
    path=tmp_path/"forward.npy";np.save(path,forward)
    c=ControlPackage("source",fps=12,frame_count=8,optical_flow={"forward":str(path),"backward":str(path)})
    first=slice_controls(c,0,4,artifact_dir=tmp_path/"shot0")
    last=slice_controls(c,4,8,artifact_dir=tmp_path/"shot1")
    assert np.array_equal(np.load(first.optical_flow["forward"]),forward[:3])
    assert np.array_equal(np.load(last.optical_flow["forward"]),forward[4:7])
    assert first.optical_flow["forward"]!=last.optical_flow["forward"]
    assert c.optical_flow["forward"]==str(path)


def test_unresolved_models_and_unverified_comfy_server_do_not_reuse_cache(tmp_path,cpu_backend):
    from genjutsu.backends.comfyui import ComfyUIBackend
    assert backend_signature(ComfyUIBackend({}))["cache_safe"] is False
    cpu_backend.model_id=str(tmp_path/"not-present")
    c=ControlPackage("source",fps=12,width=16,height=16,frame_count=2)
    frames=FrameStore.from_frames(np.zeros((16,16,3),np.uint8) for _ in range(2))
    for _ in range(2):
        _,states,_=run_shots(frames,[],None,c,c.references,cpu_backend,tmp_path/"work")
        assert states[0]["status"]=="complete"
    assert len(cpu_backend.calls)==2
