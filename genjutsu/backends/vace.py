"""Wan VACE video editing adapter (official Diffusers API; current documented checkpoints are Wan 2.1)."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import threading
from PIL import Image
from .base import VideoBackend, BackendUnavailable
from ..controls import ControlPackage
from ..motion import scene_chunk_ranges
from .utils import as_rgb8

class WanVACEBackend(VideoBackend):
    name="wan-vace-diffusers"
    _PIPELINES={}; _LOCK=threading.Lock()
    def __init__(self,settings: dict):
        self.settings=dict(settings)
        self.revision=settings.get("revision")
        self.model_id=settings.get("model_id","Wan-AI/Wan2.1-VACE-1.3B-diffusers")
        self.device=settings.get("device","cuda"); self.dtype_name=settings.get("dtype","bfloat16")
        self.steps=int(settings.get("steps",30)); self.guidance=float(settings.get("guidance",5)); self.offload=bool(settings.get("cpu_offload",True))
        self.allow_download=bool(settings.get("allow_download",False))
        self.cache_chunks=bool(settings.get("cache_chunks",True))
        self.chunk_size=int(settings.get("chunk_frames",81)); self.overlap=int(settings.get("overlap_frames",9)); self._pipe=None
    def _load(self):
        if self._pipe is not None:return self._pipe
        try:
            import torch
            from diffusers import AutoencoderKLWan, WanVACEPipeline
        except ImportError as e: raise BackendUnavailable("Wan VACE needs a CUDA-compatible PyTorch and Diffusers build with WanVACEPipeline. Install optional Wan dependencies; model weights are not downloaded by setup.") from e
        if self.device.startswith("cuda") and not torch.cuda.is_available(): raise BackendUnavailable("Wan VACE is configured for CUDA, but CUDA is unavailable")
        dtype=getattr(torch,self.dtype_name,None)
        if dtype is None: raise ValueError(f"Unknown torch dtype {self.dtype_name}")
        key=(self.model_id,self.revision,getattr(self,"_cache_model_identity",None),self.dtype_name,self.device,self.offload,self.allow_download)
        with self._LOCK:
            if key in self._PIPELINES:self._pipe=self._PIPELINES[key]; return self._pipe
        try:
            model=getattr(self,"_resolved_model_path",None) or self.model_id
            vae=AutoencoderKLWan.from_pretrained(model,revision=self.revision,subfolder="vae",torch_dtype=torch.float32,local_files_only=not self.allow_download)
            self._pipe=WanVACEPipeline.from_pretrained(model,revision=self.revision,vae=vae,torch_dtype=dtype,local_files_only=not self.allow_download)
            if self.offload:self._pipe.enable_model_cpu_offload()
            else:self._pipe.to(self.device)
            self._PIPELINES[key]=self._pipe
        except Exception as e: raise BackendUnavailable(f"Could not load Wan VACE model {self.model_id}: {e}") from e
        return self._pipe
    def generate(self,controls: ControlPackage,source_frames: list,output_dir: str|Path)->list:
        if not source_frames: raise ValueError("No source frames provided")
        if not controls.object_masks and not controls.subject_masks:
            raise BackendUnavailable("Wan VACE requires an edit mask for region editing. Set the mask sequence on ControlPackage; only full-scene recast may use a full-white mask.")
        mask_paths=controls.object_masks or controls.subject_masks
        masks=_load_masks(mask_paths,len(source_frames),(controls.height,controls.width))
        try:
            import torch
        except ImportError as e: raise BackendUnavailable("Diffusers and PyTorch are required for Wan VACE") from e
        ref_paths=[r.path for r in controls.references.references]
        refs=[Image.open(p).convert("RGB") for p in ref_paths]
        # Wan's video VAE expects a 4k+1 frame count. Chunks are padded by repeating the final
        # source frame and trimmed after inference, so output duration and audio timing stay exact.
        ranges=scene_chunk_ranges(len(source_frames),self.chunk_size,self.overlap,controls.scene_cuts)
        cache_root=Path(output_dir)/"vace_chunks"; pipe=None
        from ..frame_store import FrameStore
        from ..cache import backend_signature, generation_key
        from copy import deepcopy
        signature=controls.settings.get("generation_signature") or backend_signature(self)
        can_cache=self.cache_chunks and signature["cache_safe"]
        output=FrameStore()
        for start,end,trim_start in ranges:
            src=list(source_frames[start:end]); msk=list(masks[start:end]); actual=len(src)
            wanted=actual if actual%4==1 else actual+(1-actual)%4
            if wanted<5:wanted=5
            while len(src)<wanted:src.append(src[-1]); msk.append(msk[-1])
            h=controls.height//16*16; w=controls.width//16*16
            if h<16 or w<16: raise ValueError("Wan VACE requires frame dimensions of at least 16 pixels")
            chunk_controls=deepcopy(controls)
            chunk_controls.settings["chunk_range"]=[start,end,trim_start]
            chunk_controls.settings["padded_size"]=[h,w,wanted]
            key=generation_key(chunk_controls,src,msk,signature)
            cached=cache_root/key
            if can_cache and (cached/"complete.json").is_file():
                try:
                    current=FrameStore.open(cached)
                    if len(current)==actual and current.shape==(controls.height,controls.width,3):
                        output.extend(current[trim_start:]); continue
                except (OSError,ValueError,KeyError): pass
            video=[Image.fromarray(f).resize((w,h)) for f in src]
            # VACE convention from Diffusers: black is preserved/conditioned, white is generated.
            mask=[Image.fromarray(m).convert("L").resize((w,h),Image.Resampling.NEAREST) for m in msk]
            if pipe is None:
                from diffusers.utils import load_video
                pipe=self._load()
            generator=torch.Generator(device=self.device).manual_seed(int(controls.settings.get("seed",0))+start)
            result=pipe(video=video,mask=mask,reference_images=refs or None,prompt=controls.prompt or "",negative_prompt=controls.negative_prompt or "",height=h,width=w,num_frames=wanted,num_inference_steps=self.steps,guidance_scale=self.guidance,generator=generator).frames[0]
            current=[as_rgb8(f) for f in result[:actual]]
            import cv2
            current=[cv2.resize(f,(controls.width,controls.height),interpolation=cv2.INTER_LANCZOS4) for f in current]
            if len(current)!=actual:raise RuntimeError("Wan VACE returned too few frames")
            if can_cache:
                cached.mkdir(parents=True,exist_ok=True)
                (cached/"complete.json").unlink(missing_ok=True)
                FrameStore.from_frames(current,cached)
                (cached/"complete.json").write_text(json.dumps({"key":key}),encoding="utf-8")
            output.extend(current[trim_start:])
        return output.finish()

def _load_masks(paths:list[str],count:int,shape:tuple[int,int])->list[np.ndarray]:
    p=Path(paths[0]); files=sorted(p.glob("*.png")) if p.is_dir() else [Path(x) for x in paths]
    from ..frame_store import ImageSequence
    return ImageSequence(files,count,shape)
