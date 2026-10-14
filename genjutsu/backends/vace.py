"""Wan VACE video editing adapter (official Diffusers API; current documented checkpoints are Wan 2.1)."""
from __future__ import annotations
from pathlib import Path
import hashlib, json
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
        key=(self.model_id,self.dtype_name,self.device,self.offload,self.allow_download)
        with self._LOCK:
            if key in self._PIPELINES:self._pipe=self._PIPELINES[key]; return self._pipe
        try:
            vae=AutoencoderKLWan.from_pretrained(self.model_id,subfolder="vae",torch_dtype=torch.float32,local_files_only=not self.allow_download)
            self._pipe=WanVACEPipeline.from_pretrained(self.model_id,vae=vae,torch_dtype=dtype,local_files_only=not self.allow_download)
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
        chunks=[]
        for start,end,trim_start in ranges:
            src=source_frames[start:end]; msk=masks[start:end]; actual=len(src)
            wanted=actual if actual%4==1 else actual+(1-actual)%4
            if wanted<5:wanted=5
            while len(src)<wanted:src.append(src[-1]); msk.append(msk[-1])
            h=controls.height//16*16; w=controls.width//16*16
            if h<16 or w<16: raise ValueError("Wan VACE requires frame dimensions of at least 16 pixels")
            digest=hashlib.sha256()
            for frame,mask_frame in zip(src,msk): digest.update(frame.tobytes()); digest.update(mask_frame.tobytes())
            references=[(r.path,Path(r.path).stat().st_size,Path(r.path).stat().st_mtime_ns,r.weight) for r in controls.references.references]
            fingerprint={"source_chunk_sha256":digest.hexdigest(),"range":[start,end],"operation":controls.operation.value,"prompt":controls.prompt,"negative_prompt":controls.negative_prompt,"references":references,"model":self.model_id,"steps":self.steps,"guidance":self.guidance,"seed":int(controls.settings.get("seed",0))+start,"size":[h,w]}
            key=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest(); cached=cache_root/f"{key}.npz"
            if self.cache_chunks and cached.is_file() and (cache_root/f"{key}.json").is_file():
                try:
                    with np.load(cached) as archive: current=[f.copy() for f in archive["frames"]]
                    if len(current)==actual:
                        chunks.append(current[trim_start:]); continue
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
            if self.cache_chunks:
                cache_root.mkdir(parents=True,exist_ok=True); np.savez_compressed(cached,frames=np.stack(current))
                (cache_root/f"{key}.json").write_text(json.dumps(fingerprint,indent=2),encoding="utf-8")
            chunks.append(current[trim_start:])
        return [frame for chunk in chunks for frame in chunk]

def _load_masks(paths:list[str],count:int,shape:tuple[int,int])->list[np.ndarray]:
    from PIL import Image
    p=Path(paths[0]); files=sorted(p.glob("*.png")) if p.is_dir() else [Path(x) for x in paths]
    if len(files)==1:files*=count
    if len(files)!=count:raise ValueError(f"Expected {count} mask frames, got {len(files)}")
    out=[]
    for f in files:
        m=np.asarray(Image.open(f).convert("L"))
        if m.shape!=shape:raise ValueError(f"Mask dimension mismatch for {f}")
        out.append(m)
    return out
