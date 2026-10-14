from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from PIL import Image
from genjutsu.video import probe_video, read_frames, write_video, timing_kwargs
from genjutsu.analysis import scene_cuts, estimate_camera, optical_flow
from genjutsu.frame_store import FrameStore
from genjutsu.segmentation import SAM2Segmenter, mask_from_bbox, save_masks
from genjutsu.tracking import propagate_mask_sequence
from genjutsu.controls import ControlPackage, Operation, Reference
from genjutsu.prompt import compile_prompt

class GenjutsuVideoLoader:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"video_path":("STRING",{"default":""})}}
    RETURN_TYPES=("IMAGE","STRING","FLOAT","INT","INT","INT","FLOAT","STRING"); RETURN_NAMES=("frames","video_path","fps","width","height","frame_count","duration","audio_metadata_json"); FUNCTION="load"; CATEGORY="Genjutsu/Video"
    def load(self,video_path):
        frames,v=read_frames(video_path)
        import torch
        tensor=torch.from_numpy(np.stack(frames).astype(np.float32)/255.0)
        return (tensor,v.path,v.fps,v.width,v.height,v.frame_count,v.duration,json.dumps(v.audio_metadata or {}))

class GenjutsuSceneAnalyzer:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"video_path":("STRING",{"default":""})}}
    RETURN_TYPES=("STRING","STRING","STRING"); RETURN_NAMES=("scene_cuts_json","camera_motion_json","video_metadata_json"); FUNCTION="analyze"; CATEGORY="Genjutsu/Analysis"
    def analyze(self,video_path):
        frames,info=read_frames(video_path); cuts=scene_cuts(frames); cam=estimate_camera(frames)
        return (json.dumps(cuts),json.dumps(cam),json.dumps(info.__dict__))

class GenjutsuGroundingDetector:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"video_path":("STRING",{"default":""}),"text_prompt":("STRING",{"default":"person"}),"model_directory":("STRING",{"default":"models/grounding-dino"}),"output_json":("STRING",{"default":"output/detections.json"})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("detections_json",); FUNCTION="detect"; CATEGORY="Genjutsu/Video Understanding"
    def detect(self,video_path,text_prompt,model_directory,output_json):
        from genjutsu.analysis import grounded_detect
        result=grounded_detect(read_frames(video_path)[0],text_prompt,model_directory); p=Path(output_json); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(result,indent=2),encoding="utf-8"); return (str(p.resolve()),)

class GenjutsuSAM2Segmenter:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"video_path":("STRING",{"default":""}),"sam2_config":("STRING",{"default":""}),"checkpoint":("STRING",{"default":""}),"frame_index":("INT",{"default":0,"min":0}),"x1":("INT",{"default":0,"min":0}),"y1":("INT",{"default":0,"min":0}),"x2":("INT",{"default":100,"min":1}),"y2":("INT",{"default":100,"min":1}),"output_dir":("STRING",{"default":"output/masks"})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("mask_directory",); FUNCTION="segment"; CATEGORY="Genjutsu/Video Understanding"
    def segment(self,video_path,sam2_config,checkpoint,frame_index,x1,y1,x2,y2,output_dir):
        frames,_=read_frames(video_path); predictor=SAM2Segmenter(sam2_config,checkpoint)
        masks=predictor.segment_video(frames,frame_index,(x1,y1,x2,y2)); return (str(Path(save_masks(masks,output_dir)[0]).parent),)

class GenjutsuFlowTracker:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"video_path":("STRING",{"default":""}),"mask_path":("STRING",{"default":""}),"start_index":("INT",{"default":0,"min":0}),"output_dir":("STRING",{"default":"output/tracked_masks"})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("mask_directory",); FUNCTION="track"; CATEGORY="Genjutsu/Video Understanding"
    def track(self,video_path,mask_path,start_index,output_dir):
        frames,_=read_frames(video_path); mask=np.asarray(Image.open(mask_path).convert("L")); masks=propagate_mask_sequence(frames,mask,start_index)
        paths=save_masks(masks,output_dir); return (str(Path(paths[0]).parent),)

class GenjutsuOpticalFlow:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"video_path":("STRING",{"default":""}),"model":(["farneback","raft"],),"checkpoint":("STRING",{"default":""}),"output_dir":("STRING",{"default":"output/flow"})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("flow_directory",); FUNCTION="compute"; CATEGORY="Genjutsu/Video Understanding"
    def compute(self,video_path,model,checkpoint,output_dir):
        from genjutsu.analysis import optical_flow_with_backend
        frames,_=read_frames(video_path); fw,bw=optical_flow_with_backend(frames,model,checkpoint or None); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
        np.save(out/"forward.npy",np.stack(fw) if fw else np.empty((0,))); np.save(out/"backward.npy",np.stack(bw) if bw else np.empty((0,))); return (str(out),)

class GenjutsuCameraAnalyzer:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"video_path":("STRING",{"default":""})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("camera_motion_json",); FUNCTION="analyze"; CATEGORY="Genjutsu/Video Understanding"
    def analyze(self,video_path): return (json.dumps(estimate_camera(read_frames(video_path)[0])),)

class GenjutsuControlPackage:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"source_video":("STRING",{"default":""}),"operation":([o.value for o in Operation],),"prompt":("STRING",{"multiline":True}),"references_json":("STRING",{"default":"[]"}),"preserve_json":("STRING",{"default":"[\"motion\",\"camera\",\"composition\"]"}),"output_path":("STRING",{"default":"output/controls.json"})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("controls_json",); FUNCTION="build"; CATEGORY="Genjutsu/Controls"
    def build(self,source_video,operation,prompt,references_json,preserve_json,output_path):
        v=probe_video(source_video); refs=[Reference(**r) for r in json.loads(references_json)]
        c=ControlPackage(source_video=v.path,fps=v.fps,width=v.width,height=v.height,frame_count=v.frame_count,duration=v.duration,audio_metadata=v.audio_metadata or {},operation=Operation(operation),prompt=prompt)
        c.fps_rational=v.fps_rational;c.time_base=v.time_base;c.frame_pts=v.frame_pts;c.frame_durations=v.frame_durations;c.timestamps=v.timestamps;c.source_start_time=v.start_time;c.is_vfr=v.is_vfr
        c.references.references=refs; c.settings["preserve"]=json.loads(preserve_json); c.save(output_path); return (str(Path(output_path).resolve()),)

class GenjutsuPromptCompiler:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"controls_json":("STRING",{"default":""})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("compiled_prompt",); FUNCTION="compile"; CATEGORY="Genjutsu/Controls"
    def compile(self,controls_json):
        from genjutsu.controls import ControlPackage
        return (compile_prompt(ControlPackage.load(controls_json)),)

class GenjutsuWan22Generator:
    """Run actual Hugging Face Diffusers Wan 2.2 inference; requires local/downloadable model weights."""
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"source_video":("STRING",{"default":""}),"reference_image":("STRING",{"default":""}),"mode":(["animate","i2v"],),"prompt":("STRING",{"multiline":True}),"negative_prompt":("STRING",{"multiline":True}),"model_id":("STRING",{"default":"Wan-AI/Wan2.2-Animate-14B-Diffusers"}),"dtype":(["float16","bfloat16","float32"],),"pose_video":("STRING",{"default":""}),"face_video":("STRING",{"default":""}),"mask_video":("STRING",{"default":""}),"output_path":("STRING",{"default":"output/wan_generated.mp4"}),"seed":("INT",{"default":0,"min":0}),"steps":("INT",{"default":30,"min":1,"max":100})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("video_path",); FUNCTION="generate"; CATEGORY="Genjutsu/Generation"
    def generate(self,source_video,reference_image,mode,prompt,negative_prompt,model_id,dtype,pose_video,face_video,mask_video,output_path,seed,steps):
        from genjutsu.backends.wan import Wan22Backend
        frames,info=read_frames(source_video,disk_backed=True)
        c=ControlPackage(source_video=info.path,fps=info.fps,width=info.width,height=info.height,frame_count=info.frame_count,operation=Operation.CHARACTER_SWAP,prompt=prompt,negative_prompt=negative_prompt)
        c.fps_rational=info.fps_rational;c.time_base=info.time_base;c.frame_pts=info.frame_pts;c.frame_durations=info.frame_durations;c.timestamps=info.timestamps;c.source_start_time=info.start_time;c.is_vfr=info.is_vfr
        c.references.references=[Reference(str(Path(reference_image).resolve()),"CHARACTER",1.)]; c.settings.update({"pose_video":pose_video or None,"face_video":face_video or None,"mask_video":mask_video or None,"seed":seed})
        engine=Wan22Backend({"mode":mode,"model_id":model_id,"dtype":dtype,"steps":steps,"cpu_offload":True}); out_frames=engine.generate(c,frames,Path(output_path).parent)
        write_video(out_frames,output_path,info.fps_rational,audio_from=source_video,**timing_kwargs(info))
        return (str(Path(output_path).resolve()),)

class GenjutsuVideoOutput:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"video_path":("STRING",{"default":""})}}
    RETURN_TYPES=(); FUNCTION="output"; OUTPUT_NODE=True; CATEGORY="Genjutsu/Video"
    def output(self,video_path):
        if not Path(video_path).is_file(): raise FileNotFoundError(video_path)
        try:
            import folder_paths
            output_dir=Path(folder_paths.get_output_directory())
            src=Path(video_path).resolve(); dest=output_dir/src.name
            if src!=dest.resolve():
                import shutil
                shutil.copy2(src,dest)
            return {"ui":{"text":[str(dest)],"gifs":[{"filename":dest.name,"subfolder":"","type":"output"}]}}
        except ImportError:
            return {"ui":{"text":[str(Path(video_path).resolve())]}}

class GenjutsuWanVACEGenerator:
    """Real Wan VACE video-to-video transformation conditioned on a per-frame edit mask."""
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"source_video":("STRING",{"default":""}),"reference_image":("STRING",{"default":""}),"additional_reference_images_json":("STRING",{"default":"[]","multiline":True}),"edit_mask_directory":("STRING",{"default":""}),"prompt":("STRING",{"multiline":True}),"negative_prompt":("STRING",{"multiline":True}),"model_id":("STRING",{"default":"Wan-AI/Wan2.1-VACE-1.3B-diffusers"}),"dtype":(["float16","bfloat16","float32"],),"output_path":("STRING",{"default":"output/vace_generated.mp4"}),"seed":("INT",{"default":0,"min":0}),"steps":("INT",{"default":30,"min":1,"max":100}),"chunk_frames":("INT",{"default":81,"min":5}),"overlap_frames":("INT",{"default":9,"min":0})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("video_path",); FUNCTION="generate"; CATEGORY="Genjutsu/Generation"
    def generate(self,source_video,reference_image,additional_reference_images_json,edit_mask_directory,prompt,negative_prompt,model_id,dtype,output_path,seed,steps,chunk_frames,overlap_frames):
        from genjutsu.backends.vace import WanVACEBackend
        frames,info=read_frames(source_video,disk_backed=True); masks=sorted(Path(edit_mask_directory).glob("*.png"))
        if len(masks)!=len(frames): raise ValueError(f"Expected {len(frames)} masks, got {len(masks)}")
        c=ControlPackage(source_video=info.path,fps=info.fps,width=info.width,height=info.height,frame_count=info.frame_count,operation=Operation.OBJECT_SWAP,prompt=prompt,negative_prompt=negative_prompt)
        c.fps_rational=info.fps_rational;c.time_base=info.time_base;c.frame_pts=info.frame_pts;c.frame_durations=info.frame_durations;c.timestamps=info.timestamps;c.source_start_time=info.start_time;c.is_vfr=info.is_vfr
        from genjutsu.analysis import scene_cuts
        additional=json.loads(additional_reference_images_json or "[]")
        c.references.references=[Reference(str(Path(p).resolve()),"OBJECT",1.) for p in [reference_image,*additional]]; c.object_masks=[str(Path(edit_mask_directory).resolve())]; c.settings["seed"]=seed; c.scene_cuts=scene_cuts(frames)
        engine=WanVACEBackend({"model_id":model_id,"dtype":dtype,"steps":steps,"chunk_frames":chunk_frames,"overlap_frames":overlap_frames,"cpu_offload":True})
        out_frames=engine.generate(c,frames,Path(output_path).parent); write_video(out_frames,output_path,info.fps_rational,audio_from=source_video,**timing_kwargs(info))
        return (str(Path(output_path).resolve()),)

class GenjutsuPoseExtractor:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"video_path":("STRING",{"default":""}),"model":(["mediapipe"],),"output_json":("STRING",{"default":"output/pose.json"})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("pose_json",); FUNCTION="extract"; CATEGORY="Genjutsu/Video Understanding"
    def extract(self,video_path,model,output_json):
        from genjutsu.analysis import pose_sequence
        seq=pose_sequence(read_frames(video_path)[0],model); p=Path(output_json); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(seq),encoding="utf-8"); return (str(p.resolve()),)

class GenjutsuDepthExtractor:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"video_path":("STRING",{"default":""}),"checkpoint_directory":("STRING",{"default":""}),"output_directory":("STRING",{"default":"output/depth"})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("depth_directory",); FUNCTION="extract"; CATEGORY="Genjutsu/Video Understanding"
    def extract(self,video_path,checkpoint_directory,output_directory):
        from genjutsu.analysis import depth_sequence
        values=depth_sequence(read_frames(video_path)[0],"depth_anything",checkpoint_directory); out=Path(output_directory); out.mkdir(parents=True,exist_ok=True)
        for i,d in enumerate(values): Image.fromarray((d*65535).astype(np.uint16)).save(out/f"{i:08d}.png")
        return (str(out.resolve()),)

class GenjutsuReferenceManager:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"references_json":("STRING",{"default":"[]","multiline":True})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("reference_package_json",); FUNCTION="manage"; CATEGORY="Genjutsu/References"
    def manage(self,references_json):
        from genjutsu.references import ReferenceManager
        m=ReferenceManager(); rows=json.loads(references_json)
        for row in rows:
            ref=m.add(row["path"],row.get("category","OBJECT"),float(row.get("weight",1)))
            ref.metadata["classical_features"]=m.extract_classical_features(ref.path)
        return (json.dumps({"references":[r.__dict__ for r in m.package.references]},indent=2),)

class GenjutsuTemporalRepair:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"video_path":("STRING",{"default":""}),"strength":("FLOAT",{"default":0.2,"min":0,"max":1}),"output_path":("STRING",{"default":"output/temporal.mp4"})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("video_path",); FUNCTION="repair"; CATEGORY="Genjutsu/Post Processing"
    def repair(self,video_path,strength,output_path):
        from genjutsu.temporal import stabilize_generated
        frames,info=read_frames(video_path,disk_backed=True); repaired=stabilize_generated(frames,strength); write_video(repaired,output_path,info.fps_rational,audio_from=video_path,**timing_kwargs(info)); return (str(Path(output_path).resolve()),)

class GenjutsuCompositor:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"source_video":("STRING",{"default":""}),"generated_video":("STRING",{"default":""}),"edit_mask_directory":("STRING",{"default":""}),"output_path":("STRING",{"default":"output/composite.mp4"}),"feather_px":("INT",{"default":8,"min":0,"max":128})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("video_path",); FUNCTION="composite_video"; CATEGORY="Genjutsu/Post Processing"
    def composite_video(self,source_video,generated_video,edit_mask_directory,output_path,feather_px):
        from genjutsu.compositing import composite
        src,info=read_frames(source_video,disk_backed=True); gen,_=read_frames(generated_video,disk_backed=True); mask_files=sorted(Path(edit_mask_directory).glob("*.png"))
        if len(src)!=len(gen) or len(src)!=len(mask_files): raise ValueError("Source, generated video and mask sequences must have identical frame counts")
        out=FrameStore.from_frames(composite(a,b,np.asarray(Image.open(m).convert("L")),feather_px=feather_px) for a,b,m in zip(src,gen,mask_files))
        write_video(out,output_path,info.fps_rational,audio_from=source_video,**timing_kwargs(info)); return (str(Path(output_path).resolve()),)

class GenjutsuIdentityScore:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"reference_image":("STRING",{"default":""}),"video_path":("STRING",{"default":""}),"model_directory":("STRING",{"default":"models/insightface"}),"output_json":("STRING",{"default":"output/identity.json"})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("identity_scores_json",); FUNCTION="score"; CATEGORY="Genjutsu/Quality"
    def score(self,reference_image,video_path,model_directory,output_json):
        import cv2
        from genjutsu.identity import LocalInsightFace
        from genjutsu.video import read_frames
        engine=LocalInsightFace(root=model_directory); ref=np.asarray(Image.open(reference_image).convert("RGB")); faces=engine.embeddings([ref]);
        if not faces[0]: raise ValueError("No face detected in reference image")
        scores=engine.score(faces[0][0].normed_embedding,read_frames(video_path)[0]); path=Path(output_json); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps({"scores":scores,"threshold":.55},indent=2),encoding="utf-8"); return (str(path.resolve()),)

class GenjutsuUpscaler:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"video_path":("STRING",{"default":""}),"method":(["lanczos","realesrgan"],),"scale":("INT",{"default":2,"min":2,"max":4}),"model_path":("STRING",{"default":""}),"output_path":("STRING",{"default":"output/upscaled.mp4"})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("video_path",); FUNCTION="upscale"; CATEGORY="Genjutsu/Post Processing"
    def upscale(self,video_path,method,scale,model_path,output_path):
        from genjutsu.restoration import resize_frames,realesrgan
        frames,info=read_frames(video_path); out=resize_frames(frames,scale) if method=="lanczos" else realesrgan(frames,model_path,scale)
        write_video(out,output_path,info.fps_rational,audio_from=video_path,**timing_kwargs(info)); return (str(Path(output_path).resolve()),)

NODE_CLASS_MAPPINGS={
    "GenjutsuVideoLoader":GenjutsuVideoLoader,"GenjutsuSceneAnalyzer":GenjutsuSceneAnalyzer,"GenjutsuGroundingDetector":GenjutsuGroundingDetector,
    "GenjutsuSAM2Segmenter":GenjutsuSAM2Segmenter,"GenjutsuFlowTracker":GenjutsuFlowTracker,
    "GenjutsuOpticalFlow":GenjutsuOpticalFlow,"GenjutsuCameraAnalyzer":GenjutsuCameraAnalyzer,
    "GenjutsuControlPackage":GenjutsuControlPackage,"GenjutsuPromptCompiler":GenjutsuPromptCompiler,
    "GenjutsuWan22Generator":GenjutsuWan22Generator,"GenjutsuWanVACEGenerator":GenjutsuWanVACEGenerator,"GenjutsuVideoOutput":GenjutsuVideoOutput,
    "GenjutsuPoseExtractor":GenjutsuPoseExtractor,"GenjutsuDepthExtractor":GenjutsuDepthExtractor,"GenjutsuReferenceManager":GenjutsuReferenceManager,
    "GenjutsuTemporalRepair":GenjutsuTemporalRepair,"GenjutsuCompositor":GenjutsuCompositor,"GenjutsuIdentityScore":GenjutsuIdentityScore,"GenjutsuUpscaler":GenjutsuUpscaler,
}
NODE_DISPLAY_NAME_MAPPINGS={k:k.replace("Genjutsu","Genjutsu ").replace("Wan22","Wan 2.2 ") for k in NODE_CLASS_MAPPINGS}
