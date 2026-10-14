import subprocess
import numpy as np
import pytest

from genjutsu.analysis import scene_cuts
from genjutsu.occlusion import visibility_mask
from genjutsu.routing import route_config
from genjutsu.segmentation import mask_from_bbox
from genjutsu.shots import shot_ranges
from genjutsu.tracking import propagate_mask_sequence
from genjutsu.video import probe_video, read_frames, write_video
from genjutsu.wan_preprocess import validate_wan_conditioning


@pytest.mark.parametrize("fps",[12,24])
def test_synthetic_multishot_video_audio_motion_masks_and_occlusion(tmp_path,fps):
    frames=[]
    backgrounds=[(20,35,160),(30,150,40),(160,35,25)]
    for i in range(12):
        shot=i//4; image=np.zeros((48,64,3),np.uint8);image[:]=backgrounds[shot]
        x=5+(i%4)*4
        image[16:31,x:x+12]=[230,220,30]  # moving selected object
        if i%4>=2:image[16:31,x+5:x+12]=[230,20,20]  # a moving foreground occluder
        frames.append(image)
    silent=tmp_path/f"silent_{fps}.mp4";write_video(frames,silent,fps)
    tone=tmp_path/f"tone_{fps}.wav"
    subprocess.run(["ffmpeg","-y","-v","error","-f","lavfi","-i","sine=frequency=440:duration=1","-c:a","pcm_s16le",str(tone)],check=True,capture_output=True)
    source=tmp_path/f"source_{fps}.mp4"
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(silent),"-i",str(tone),"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-shortest",str(source)],check=True,capture_output=True)
    decoded,info=read_frames(source)
    cuts=scene_cuts(decoded)
    assert info.has_audio and info.audio_metadata["codec_name"]=="aac"
    assert len(decoded)==12 and abs(info.fps-fps)<.01
    assert len(cuts)==2 and shot_ranges(len(decoded),cuts)==[(0,4),(4,8),(8,12)]

    moving=decoded[:4]
    initial=mask_from_bbox(moving[0].shape[:2],(5,16,17,31))
    tracked=propagate_mask_sequence(moving,initial)
    assert len(tracked)==4 and all(mask.shape==(48,64) for mask in tracked)
    centers=[np.where(mask>0)[1].mean() for mask in tracked]
    assert centers[-1]>centers[0]

    target=mask_from_bbox((48,64),(5,16,25,31))
    foreground=mask_from_bbox((48,64),(15,16,25,31))
    visible=visibility_mask(target,np.full((48,64),.8,np.float32),np.full((48,64),.3,np.float32),foreground)
    assert np.all(visible[16:31,15:25]==0)
    assert np.all(visible[16:31,5:15]==255)
    inverse_depth_visible=visibility_mask(target,np.full((48,64),.2,np.float32),np.full((48,64),.8,np.float32),foreground,near_is_high=True)
    assert np.all(inverse_depth_visible[16:31,15:25]==0)

    output=tmp_path/f"output_{fps}.mp4";write_video(decoded,output,fps,audio_from=source)
    result=probe_video(output)
    assert result.has_audio and result.frame_count==12 and abs(result.fps-fps)<.01


def test_automatic_operation_router_uses_real_compatible_backend():
    base={"backend":{"type":"auto"}}
    assert route_config(base,__import__("genjutsu.controls",fromlist=["Operation"]).Operation.PRODUCT_SWAP)["backend"]["type"]=="wan_vace"
    animate={"backend":{"type":"auto","use_wan_animate_for_character":True,
        "wan_animate_preprocess":{"wan_repo":"/local/Wan2.2","checkpoint_dir":"/local/preprocess"}}}
    from genjutsu.controls import Operation
    assert route_config(animate,Operation.CHARACTER_SWAP)["backend"]["type"]=="wan22"
    with pytest.raises(ValueError,match="requires"):
        route_config({"backend":{"type":"auto","use_wan_animate_for_character":True}},Operation.MOTION_TRANSFER)


def test_wan_animate_replacement_contract_video_shapes_and_mask_convention(tmp_path):
    names={"pose_video":"src_pose.mp4","face_video":"src_face.mp4","background_video":"src_bg.mp4","mask_video":"src_mask.mp4"}
    for key,name in names.items():
        clips=[]
        for index in range(5):
            frame=np.zeros((512,512,3),np.uint8) if key=="face_video" else np.zeros((32,48,3),np.uint8)
            if key=="mask_video":frame[:,:24]=255 if index<3 else 0
            elif key=="pose_video":frame[8:24,20+index:24+index]=255
            elif key=="face_video":frame[8:24,15:31]=120
            else:frame[:]=70
            clips.append(frame)
        write_video(clips,tmp_path/name,12)
    result=validate_wan_conditioning({key:tmp_path/name for key,name in names.items()},expected_fps=12,expected_frames=5)
    assert result["mask_convention"]=={"white":"generate/replace","black":"preserve/condition","channels":"grayscale RGB"}
    assert result["face_video"]["size"]==[512,512] and result["pose_video"]["size"]==[48,32]
    with pytest.raises(ValueError,match="expected source FPS"):
        validate_wan_conditioning({key:tmp_path/name for key,name in names.items()},expected_fps=24)
