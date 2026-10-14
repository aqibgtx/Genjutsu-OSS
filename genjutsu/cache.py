from __future__ import annotations
from pathlib import Path
import hashlib, json, os
import platform
import subprocess
from importlib import metadata


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_identity(path):
    """Hash content, including config/tokenizer files, without loading weights."""
    path = Path(path)
    if path.is_file():
        return {"sha256": file_digest(path)}
    if path.is_dir():
        return {str(p.relative_to(path)): file_digest(p) for p in sorted(path.rglob("*")) if p.is_file() and not ({".git","__pycache__"}&set(p.relative_to(path).parts))}
    raise FileNotFoundError(path)


def software_signature():
    packages = {}
    for name in ("genjutsu-oss", "numpy", "Pillow", "av", "opencv-python", "torch", "torchvision", "diffusers", "transformers", "accelerate", "huggingface-hub", "safetensors", "ftfy", "sam2", "SAM-2", "insightface", "onnxruntime", "onnxruntime-gpu", "decord", "moviepy", "hydra-core", "peft"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    # Editable source changes must invalidate results even before a version bump.
    root = Path(__file__).parent
    source = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        source.update(str(path.relative_to(root)).encode())
        source.update(bytes.fromhex(file_digest(path)))
    try:
        ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        ffmpeg = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages, "source_sha256": source.hexdigest(), "ffmpeg": ffmpeg}


def model_signature(model_id, revision=None):
    """Resolve only local checkpoints/HF snapshots; never request downloads.

    Unresolved floating remote IDs cannot safely reuse a generation cache.
    """
    path = Path(model_id)
    if path.exists():
        resolved = path.resolve()
        return {"requested": str(model_id), "revision": revision, "resolved": str(resolved), "content": content_identity(resolved), "cache_safe": True}
    try:
        from huggingface_hub import snapshot_download
        resolved = Path(snapshot_download(str(model_id), revision=revision, local_files_only=True))
        return {"requested": str(model_id), "revision": revision, "resolved":str(resolved), "commit": resolved.name, "content": content_identity(resolved), "cache_safe": True}
    except Exception:
        # Hub versions use different exception families for an offline miss.
        return {"requested": str(model_id), "revision": revision, "cache_safe": False}


def backend_signature(backend):
    settings = dict(getattr(backend, "settings", {}))
    keys = ("model_id", "revision", "mode", "dtype", "dtype_name", "device", "steps", "guidance", "offload", "chunk_size", "overlap", "url", "workflow", "node_map", "use_source_first_frame", "pose_video", "face_video", "mask_video", "background_video")
    values = {key: getattr(backend, key) for key in keys if hasattr(backend, key)}
    result = {"name": backend.name, "settings": settings, "values": values, "software": software_signature(), "cache_safe": True}
    if hasattr(backend, "model_id"):
        result["model"] = model_signature(backend.model_id, getattr(backend, "revision", None))
        result["cache_safe"] = result["model"]["cache_safe"]
        identity=StageCache.key("loaded-model",result["model"])
        if getattr(backend,"_cache_model_identity",identity)!=identity:
            backend._pipe=None
        backend._cache_model_identity=identity
        backend._resolved_model_path=result["model"].get("resolved")
    if values.get("workflow") and Path(values["workflow"]).is_file():
        result["workflow"] = content_identity(values["workflow"])
    if backend.name == "comfyui-http":
        # Checkpoint filenames do not establish the remote checkpoint revision.
        result["cache_safe"] = bool(settings.get("model_fingerprint") and settings.get("software_fingerprint"))
    for key in ("pose_video", "face_video", "mask_video", "background_video"):
        if values.get(key) and Path(values[key]).is_file():
            result[key] = content_identity(values[key])
    return result


def generation_key(controls, frames, masks, signature):
    digest = hashlib.sha256()
    for sequence in (frames, masks or []):
        for frame in sequence:
            digest.update(str((frame.shape, str(frame.dtype))).encode())
            digest.update(frame.tobytes())
    data = controls.to_dict()
    data.pop("source_video", None)
    # References and external control artifacts are keyed by actual bytes.
    data["references"]["references"] = [{**r.__dict__, "content": content_identity(r.path)} for r in controls.references.references]
    for name in ("subject_masks", "object_masks", "preservation_masks", "depth_sequence"):
        data[name] = [content_identity(path) for path in getattr(controls, name)]
    settings = dict(data["settings"])
    settings.pop("generation_signature", None)
    for name in ("pose_video", "face_video", "background_video", "mask_video"):
        if settings.get(name):
            settings[name] = content_identity(settings[name])
    data["settings"] = settings
    def artifact_contents(value):
        if isinstance(value,dict):return {key:artifact_contents(item) for key,item in value.items()}
        if isinstance(value,list):return [artifact_contents(item) for item in value]
        if isinstance(value,(str,Path)) and str(value).lower().endswith((".npy",".png",".json")) and Path(value).is_file():return content_identity(value)
        return value
    for name in ("tracking_data","optical_flow","camera_motion","pose_sequence"):
        data[name]=artifact_contents(data[name])
    return StageCache.key("generated-v2", {"controls": data, "frames_and_masks": digest.hexdigest(), "backend": signature})

class StageCache:
    def __init__(self,root: str|Path,enabled: bool=True): self.root=Path(root); self.enabled=enabled
    @staticmethod
    def key(stage: str, inputs: dict) -> str:
        raw=json.dumps({"stage":stage,"inputs":inputs},sort_keys=True,default=str).encode(); return hashlib.sha256(raw).hexdigest()
    def path(self,stage: str,key: str) -> Path: return self.root/stage/key
    def get_or_create(self,stage: str,inputs: dict,writer):
        key=self.key(stage,inputs); p=self.path(stage,key)
        if self.enabled and (p/"complete.json").exists(): return p,True
        p.mkdir(parents=True,exist_ok=True); writer(p)
        if self.enabled: (p/"complete.json").write_text(json.dumps({"key":key,"stage":stage}),encoding="utf-8")
        return p,False
