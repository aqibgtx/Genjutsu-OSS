"""Measured CPU diagnostics; absent learned metrics are explicitly unavailable."""
import numpy as np
from .analysis import optical_flow,estimate_camera

def evaluate_transformation(source,generated,masks=None,scene_cuts=None,identity=None):
    if len(source)!=len(generated) or not source:raise ValueError('Evaluation needs aligned, nonempty source/output sequences')
    cuts=set(scene_cuts or [])
    preserve_error=0.;preserve_pixels=0;flow_errors=[];temporal_errors=[]
    import cv2
    for index,(a,b) in enumerate(zip(source,generated)):
        if a.shape!=b.shape:raise ValueError('Evaluation geometry differs')
        if masks is not None:
            protected=masks[index]==0
            preserve_error+=float(np.abs(a.astype(np.float32)-b.astype(np.float32))[protected].sum())
            preserve_pixels+=int(protected.sum())*3
        if index and index not in cuts:
            src_forward,_=optical_flow([source[index-1],a])
            gen_forward,gen_backward=optical_flow([generated[index-1],b])
            flow_errors.append(float(np.linalg.norm(src_forward[0]-gen_forward[0],axis=2).mean()))
            h,w=b.shape[:2];x,y=np.meshgrid(np.arange(w),np.arange(h))
            previous=cv2.remap(generated[index-1],(x+gen_backward[0][:,:,0]).astype(np.float32),
                (y+gen_backward[0][:,:,1]).astype(np.float32),cv2.INTER_LINEAR)
            temporal_errors.append(float(np.abs(previous.astype(np.float32)-b.astype(np.float32)).mean()/255))
    src_camera=estimate_camera(source)['trajectory'];gen_camera=estimate_camera(generated)['trajectory']
    camera_errors=[float(np.hypot(a['dx']-b['dx'],a['dy']-b['dy'])) for index,(a,b) in enumerate(zip(src_camera,gen_camera)) if index+1 not in cuts]
    return {'identity':identity or {'status':'not_measured','reason':'No local identity evaluation configured'},
        'background_preservation':{'mean_absolute_error_8bit':preserve_error/preserve_pixels if preserve_pixels else None,'measured_channel_samples':preserve_pixels},
        'motion':{'mean_flow_endpoint_difference_pixels':float(np.mean(flow_errors)) if flow_errors else None,'method':'Farneback source/output diagnostic, not semantic motion acceptance'},
        'temporal':{'mean_flow_warp_rgb_error':float(np.mean(temporal_errors)) if temporal_errors else None},
        'camera':{'mean_affine_translation_difference_pixels':float(np.mean(camera_errors)) if camera_errors else None,'method':'2D RANSAC affine diagnostic'},
        'object_similarity':{'status':'not_measured','reason':'Requires a configured learned reference evaluator'},
        'prompt_alignment':{'status':'not_measured','reason':'Requires a configured text/image evaluator'},
        'perceptual_quality':{'status':'not_measured','reason':'Execution/media checks do not establish perceptual quality'}}
