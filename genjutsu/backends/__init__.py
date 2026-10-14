from .base import VideoBackend, BackendUnavailable
from .wan import Wan22Backend
from .comfyui import ComfyUIBackend
from .vace import WanVACEBackend

def create_backend(config: dict) -> VideoBackend:
    settings=config.get("backend",{})
    name=settings.get("type","comfyui").lower()
    if name in ("wan22","wan_animate","wan-animate","diffusers"): return Wan22Backend(settings)
    if name in ("wan_vace","vace"): return WanVACEBackend(settings)
    if name=="comfyui": return ComfyUIBackend(settings)
    raise ValueError(f"Unknown backend {name}; available: comfyui, wan22")

__all__=["VideoBackend","BackendUnavailable","Wan22Backend","WanVACEBackend","ComfyUIBackend","create_backend"]
