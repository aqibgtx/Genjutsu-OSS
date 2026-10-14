"""Real CPU media/preflight regression tests; no generative output simulation."""
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from genjutsu.preview import prepare_preview
from genjutsu.video import probe_video
from genjutsu.references import split_face_sheet
from genjutsu.routing import route_config
from genjutsu.controls import Operation
from genjutsu.workflow import preflight,resolve_input

def test_draft_preview_decodes_only_five_frames_and_preserves_vfr_audio(multishot_source,tmp_path):
    source=probe_video(multishot_source)
    result=prepare_preview(multishot_source,tmp_path/'draft')
    preview=probe_video(result)
    assert preview.frame_count==5
    assert preview.width<=144 and preview.height<=256
    assert preview.has_audio
    assert np.allclose(preview.timestamps,source.timestamps[:5],atol=.001)
    assert abs(preview.duration-sum(source.frame_durations[:5])*__import__('fractions').Fraction(source.time_base))<.001
    assert json.loads((tmp_path/'draft/preview.json').read_text())['frame_count']==5

def test_preview_rejects_invalid_model_geometry_before_io(tmp_path):
    with pytest.raises(ValueError,match='multiples of 16'):
        prepare_preview(tmp_path/'missing.mp4',tmp_path,width=143)

def test_character_auto_never_silently_uses_vace():
    for operation in (Operation.CHARACTER_SWAP,Operation.MOTION_TRANSFER):
        with pytest.raises(ValueError,match='requires'):
            route_config({'backend':{'type':'auto'}},operation)
        cfg={'backend':{'type':'auto','model_id':'existing-vace','animate_model_id':'local-animate',
             'wan_animate_preprocess':{'wan_repo':'local-source','checkpoint_dir':'local-controls'}}}
        actual=route_config(cfg,operation)
        assert actual['backend']['model_id']=='local-animate'
        assert actual['backend']['type']=='wan22'
        assert cfg['backend']['type']=='auto'
        cfg['backend']['use_wan_animate_for_character']=False
        with pytest.raises(ValueError,match='fallback'):route_config(cfg,operation)

def test_face_sheet_split_trims_labels_and_keeps_six_distinct_photos(tmp_path):
    sheet=Image.new('RGB',(300,220),'white')
    for index in range(6):
        tile=Image.new('RGB',(98,94),(30+index*25,60,90))
        sheet.paste(tile,((index%3)*100,(index//3)*110))
    source=tmp_path/'sheet.png';sheet.save(source)
    paths=split_face_sheet(source,tmp_path/'views')
    assert len(paths)==6
    for index,path in enumerate(paths):
        with Image.open(path) as image:
            assert image.height==94 and image.width==98
            assert image.getpixel((20,20))==(30+index*25,60,90)

def test_preflight_collects_missing_local_assets_without_download(tmp_path):
    config={'backend':{'type':'auto','animate_model_id':str(tmp_path/'missing'),
        'wan_animate_preprocess':{'wan_repo':str(tmp_path/'code'),'checkpoint_dir':str(tmp_path/'weights')}}}
    with pytest.raises(RuntimeError) as error:preflight(config,'character_swap')
    assert 'local model directory' in str(error.value)
    assert 'Animate preprocessing asset' in str(error.value)

def test_missing_uploaded_assets_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError,match='Uploaded asset'):
        resolve_input(str(tmp_path/'missing.png'))

def test_graphical_workflows_are_connected_draft_transform_graphs():
    root=Path(__file__).resolve().parents[1]/'workflows'
    for path in (root/'gui').glob('*.json'):
        graph=json.loads(path.read_text());api=json.loads((root/path.name).read_text())
        assert graph['nodes'][0]['type']=='GenjutsuTransform'
        assert len(graph['links'])==2
        assert graph['nodes'][0]['widgets_values'][6]=='draft'
        assert graph['nodes'][0]['widgets_values'][8]=='fixed'
        assert api['1']['inputs']['quality']=='draft'

def test_identity_missing_bundle_cannot_trigger_implicit_download(tmp_path):
    from genjutsu.identity import LocalInsightFace
    with pytest.raises(FileNotFoundError,match='automatic downloads are disabled'):
        LocalInsightFace(root=tmp_path)

def test_full_recast_cannot_silently_become_full_white_generation(tmp_path):
    from genjutsu.workflow import run_full_recast
    with pytest.raises(ValueError,match='categorized references'):
        run_full_recast('source',['reference'],'',tmp_path/'output.mp4',{},1234,'object')
    with pytest.raises(ValueError,match='Character and Environment'):
        run_full_recast('source',[{'path':'reference','category':'CHARACTER'}],'',tmp_path/'output.mp4',{},1234,'object')

def test_evaluation_measures_real_pixels_without_fabricated_identity():
    from genjutsu.evaluation import evaluate_transformation
    frames=[np.full((48,64,3),50,np.uint8) for _ in range(3)]
    masks=[np.zeros((48,64),np.uint8) for _ in range(3)]
    report=evaluate_transformation(frames,frames,masks)
    assert report['background_preservation']['mean_absolute_error_8bit']==0
    assert report['motion']['mean_flow_endpoint_difference_pixels']==0
    assert report['identity']['status']=='not_measured'
    assert report['perceptual_quality']['status']=='not_measured'

def test_animate_reference_keeps_aspect_ratio_and_segment_matches_small_clip(tmp_path):
    from genjutsu.backends.wan import prepare_animate_reference,animate_segment_length
    path=tmp_path/'reference.png'
    Image.new('RGB',(200,100),(255,0,0)).save(path)
    image=np.asarray(prepare_animate_reference(path,(144,256)))
    assert image.shape==(256,144,3)
    occupied=np.all(image==[255,0,0],axis=2)
    rows=np.flatnonzero(occupied.any(axis=1))
    assert len(rows)==72
    assert np.all(image[0]==127)
    assert animate_segment_length(77,5)==5
    assert animate_segment_length(77,8)==5
    assert animate_segment_length(77,81)==77
    with pytest.raises(ValueError):animate_segment_length(1,5)

def test_rejected_identity_retains_actual_candidate_without_regenerating(tmp_path):
    # Unit contract only: this deliberately rejecting evaluator is not evidence
    # of face recognition or generative quality.
    from genjutsu.identity import correct_identity,IdentityAcceptanceError
    from genjutsu.controls import ControlPackage,Reference,ReferencePackage
    from genjutsu.frame_store import FrameStore
    class Reject:
        def evaluate_reference(self,*args):
            return {'passes_threshold':False,'minimum':0.,'mean':0.,'scores':[0.]}
    class NoGeneration:
        def generate(self,*args):raise AssertionError('Zero correction budget must not run inference')
    frames=[np.full((24,32,3),73,np.uint8)]
    controls=ControlPackage(source_video='unit-contract',references=ReferencePackage([Reference('ref',category='CHARACTER')]))
    with pytest.raises(IdentityAcceptanceError):
        correct_identity(NoGeneration(),controls,frames,frames,tmp_path,Reject(),max_attempts=0)
    retained=FrameStore.open(tmp_path/'rejected_candidate')
    assert np.array_equal(retained[0],frames[0])
    assert json.loads((tmp_path/'identity_correction.json').read_text())['passes_threshold'] is False

def test_source_quality_preprocessing_does_not_upscale_small_identity_clip():
    from genjutsu.wan_preprocess import conditioning_area
    assert conditioning_area((1280,720),144,256)==(144,256)
    assert conditioning_area((1280,720),1920,1080)==(1280,720)

def test_invalid_numeric_inference_cannot_be_encoded_as_black_video():
    from genjutsu.backends.utils import as_rgb8
    for invalid in (np.nan,np.inf,-np.inf):
        pixels=np.zeros((24,32,3),np.float32);pixels[0,0,0]=invalid
        with pytest.raises(ValueError,match='non-finite'):as_rgb8(pixels)
    assert np.all(as_rgb8(np.full((24,32,3),.5,np.float32))==127)
