"""Real Diffusers Wan 2.2 inference adapters. Model weights are never downloaded implicitly by setup."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import threading
from .base import VideoBackend, BackendUnavailable
from ..controls import ControlPackage, Operation
from .utils import as_rgb8

def animate_segment_length(configured, frame_count):
    """Match the SDK's 4k+1 segment shape without padding a tiny test to 77."""
    segment=min(int(configured),max(5,int(frame_count)))
    if segment<5:raise ValueError('Animate segment_frame_length must be at least five')
    return max(5,((segment-1)//4)*4+1)

def prepare_animate_reference(path, size):
    """Preserve face geometry when the SDK resizes to conditioning dimensions."""
    from PIL import Image, ImageOps
    with Image.open(path) as original:
        return ImageOps.pad(original.convert('RGB'),size,method=Image.Resampling.LANCZOS,color=(127,127,127))

class Wan22Backend(VideoBackend):
    """Wan 2.2 Animate replacement or I2V generation via Hugging Face Diffusers API."""
    name="wan22-diffusers"
    _PIPELINES={}; _LOCK=threading.Lock()
    def __init__(self, settings: dict):
        self.settings=dict(settings)
        self.revision=settings.get("revision")
        self.model_id=settings.get("model_id","Wan-AI/Wan2.2-Animate-14B-Diffusers")
        self.mode=settings.get("mode","animate")
        self.dtype=settings.get("dtype","float16")
        self.device=settings.get("device","cuda")
        self.steps=int(settings.get("steps",30)); self.guidance=float(settings.get("guidance",1.0))
        self.offload=bool(settings.get("cpu_offload",True)); self._pipe=None
        self.allow_download=bool(settings.get("allow_download",False))
        self.use_source_first_frame=bool(settings.get("use_source_first_frame",False))
        self.pose_video=settings.get("pose_video"); self.face_video=settings.get("face_video"); self.mask_video=settings.get("mask_video"); self.background_video=settings.get("background_video")
    def _load(self):
        if not hasattr(self,"_cache_model_identity"):
            from ..cache import backend_signature
            backend_signature(self)
        if self._pipe is not None:return self._pipe
        try:
            import torch
            from diffusers import AutoencoderKLWan, WanAnimatePipeline, WanImageToVideoPipeline
        except ImportError as e: raise BackendUnavailable("Wan inference requires a CUDA-compatible PyTorch and Diffusers build. Install the optional wan dependencies; weights are not downloaded by the installer.") from e
        if self.device.startswith("cuda") and not torch.cuda.is_available(): raise BackendUnavailable("Configured Wan backend requests CUDA, but torch.cuda.is_available() is false.")
        dtype=getattr(torch,self.dtype,None)
        if dtype is None: raise ValueError(f"Unsupported torch dtype: {self.dtype}")
        key=(self.model_id,self.revision,getattr(self,"_cache_model_identity",None),self.mode,self.dtype,self.device,self.offload,self.allow_download)
        with self._LOCK:
            if key in self._PIPELINES:self._pipe=self._PIPELINES[key]; return self._pipe
        try:
            model=getattr(self,"_resolved_model_path",None) or self.model_id
            vae=AutoencoderKLWan.from_pretrained(model,revision=self.revision,subfolder='vae',torch_dtype=torch.float32,local_files_only=not self.allow_download)
            if self.mode=="animate": self._pipe=WanAnimatePipeline.from_pretrained(model,revision=self.revision,vae=vae,torch_dtype=dtype,local_files_only=not self.allow_download)
            elif self.mode=="i2v": self._pipe=WanImageToVideoPipeline.from_pretrained(model,revision=self.revision,vae=vae,torch_dtype=dtype,local_files_only=not self.allow_download)
            else: raise ValueError("mode must be animate or i2v")
            if self.offload: self._pipe.enable_model_cpu_offload()
            else: self._pipe.to(self.device)
            self._PIPELINES[key]=self._pipe
        except Exception as e:
            raise BackendUnavailable(f"Could not load Wan model {self.model_id}: {e}. Confirm model access, Diffusers version, and local weight availability.") from e
        return self._pipe
    def generate(self, controls: ControlPackage, source_frames: list, output_dir: str|Path) -> list:
        if not source_frames: raise ValueError("Source video has no frames")
        if self.mode=="animate" and controls.is_vfr:
            raise ValueError("Wan Animate requires explicitly prepared CFR footage; VFR timing is supported by Wan VACE")
        if len(source_frames)>int(self.settings.get("max_shot_frames",256)):
            raise ValueError("Wan Animate/I2V shot exceeds backend.max_shot_frames (default 256). Split the source into shorter shots or use chunked Wan VACE; increasing this limit requires GPU/host memory validation.")
        try:
            import torch
            from PIL import Image
            from diffusers.utils import load_video
        except ImportError as e: raise BackendUnavailable("Install Diffusers and Pillow for Wan inference.") from e
        pipe=self._load(); out_dir=Path(output_dir); out_dir.mkdir(parents=True,exist_ok=True)
        ref=next((r for r in controls.references.references if r.category.upper() in {"CHARACTER","OBJECT","PRODUCT","OUTFIT"}),None)
        if self.mode=="animate":
            if not ref: raise ValueError("Wan Animate requires a character reference image")
            pose=controls.settings.get("pose_video",self.pose_video); face=controls.settings.get("face_video",self.face_video)
            mask=controls.settings.get("mask_video",self.mask_video); background=controls.settings.get("background_video",self.background_video)
            if not (pose and face and mask and background):
                prep=controls.settings.get("wan_animate_preprocess",{})
                if not prep.get("wan_repo") or not prep.get("checkpoint_dir"):
                    raise BackendUnavailable("Wan Animate replacement needs pose, face, background and mask videos. Configure backend.wan_repo and backend.preprocess_checkpoint_dir to invoke the official Wan-Video preprocessing pipeline automatically.")
                from ..wan_preprocess import preprocess_wan_animate
                ref_path=ref.path
                generated=preprocess_wan_animate(controls.source_video,ref_path,Path(output_dir)/"wan_animate_conditioning",
                    prep["wan_repo"],prep["checkpoint_dir"],int(prep.get("fps",-1)),tuple(prep.get("resolution_area",(1280,720))))
                pose,face,mask,background=(generated[k] for k in ("pose_video","face_video","mask_video","background_video"))
            if not all(Path(p).is_file() for p in (pose,face,mask,background)):
                raise FileNotFoundError("Wan Animate conditioning artifact path is missing")
            from ..wan_preprocess import validate_wan_conditioning
            validate_wan_conditioning({"pose_video":pose,"face_video":face,"mask_video":mask,"background_video":background},controls.fps,len(source_frames))
            generator=torch.Generator(device=self.device).manual_seed(int(controls.settings.get("seed",0)))
            pose_frames=load_video(pose); face_frames=load_video(face)
            # The pipeline resizes its reference to conditioning dimensions.
            # Pad first so a landscape face crop is not stretched into a
            # portrait face, changing its geometry and identity.
            reference_image=prepare_animate_reference(ref.path,(pose_frames[0].width,pose_frames[0].height))
            kwargs={"image":reference_image,"pose_video":pose_frames,"face_video":face_frames,"prompt":controls.prompt or "", "negative_prompt":controls.negative_prompt or "", "guidance_scale":self.guidance,"num_inference_steps":self.steps,"mode":"replace","generator":generator,"output_type":"np"}
            kwargs["height"],kwargs["width"]=pose_frames[0].height,pose_frames[0].width
            if "segment_frame_length" in __import__("inspect").signature(pipe.__call__).parameters:
                kwargs["segment_frame_length"]=animate_segment_length(controls.settings.get("segment_frame_length",77),len(source_frames))
            kwargs["background_video"]=load_video(background)
            kwargs["mask_video"]=load_video(mask)
            result=pipe(**kwargs).frames[0]
        elif self.mode=="i2v":
            use_source=bool(controls.settings.get("use_source_first_frame",self.use_source_first_frame))
            if not use_source and not ref: raise ValueError("Wan I2V requires a reference image unless backend.use_source_first_frame is enabled")
            image=Image.fromarray(source_frames[0]) if use_source else Image.open(ref.path).convert("RGB")
            generator=torch.Generator(device=self.device).manual_seed(int(controls.settings.get("seed",0)))
            result=pipe(image=image,prompt=controls.prompt or "",negative_prompt=controls.negative_prompt or "",num_frames=len(source_frames),guidance_scale=self.guidance,num_inference_steps=self.steps,generator=generator).frames[0]
        else: raise ValueError(self.mode)
        frames=[as_rgb8(f) for f in result]
        if frames and frames[0].shape[:2] != source_frames[0].shape[:2]:
            import cv2
            frames=[cv2.resize(f,(source_frames[0].shape[1],source_frames[0].shape[0]),interpolation=cv2.INTER_LANCZOS4) for f in frames]
        if len(frames) != len(source_frames):
            raise RuntimeError(f"Wan returned {len(frames)} frames for a {len(source_frames)}-frame source. Configure official preprocessing to preserve source FPS/frame count.")
        return frames
