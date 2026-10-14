from pathlib import Path
from typing import Any
import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "backend": {"type": "comfyui", "url": "http://127.0.0.1:8188", "workflow": None, "timeout": 3600},
    "segmentation": {"model": "opencv", "checkpoint": None, "device": "auto"},
    "tracking": {"model": "opencv"}, "depth": {"model": "none"}, "pose": {"model": "none"},
    "optical_flow": {"model": "opencv"}, "identity": {"enabled": False},
    "temporal": {"enabled": True, "strength": .4}, "compositing": {"enabled": True, "feather_px": 8},
    "upscale": {"enabled": False, "scale": 2}, "cache": {"directory": ".genjutsu-cache", "enabled": True},
    "processing": {"chunk_frames": 81, "overlap_frames": 9, "low_vram": False, "cpu_offload": True},
}

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = {k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULT_CONFIG.items()}
    if path and Path(path).exists():
        with Path(path).open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        for key, value in loaded.items():
            config[key] = {**config.get(key, {}), **value} if isinstance(value, dict) else value
    return config
