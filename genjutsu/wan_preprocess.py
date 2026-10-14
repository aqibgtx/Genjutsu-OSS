"""Adapter for Wan-Video's official Wan 2.2 Animate preprocessing pipeline.

This invokes the upstream preprocessing implementation and validates its media
artifacts. It never substitutes generic pose or segmentation output for the
model-specific conditioning videos.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import hashlib
import json
from typing import Any

from .video import probe_video

OUTPUTS = {
    "pose_video": "src_pose.mp4",
    "face_video": "src_face.mp4",
    "background_video": "src_bg.mp4",
    "mask_video": "src_mask.mp4",
}

def conditioning_area(resolution_area,width,height):
    """Keep the requested model area bounded by actual source pixels."""
    if int(resolution_area[0])*int(resolution_area[1])>width*height:return (width,height)
    return tuple(map(int,resolution_area))


def _sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def validate_wan_conditioning(artifacts: dict[str, str | Path], expected_fps: float | None = None,
                              expected_frames: int | None = None) -> dict[str, dict]:
    """Validate Wan's actual four-video replacement inputs before model invocation."""
    missing=[name for name in OUTPUTS if name not in artifacts]
    if missing:raise ValueError(f"Wan Animate replacement artifacts missing: {missing}")
    from .video import iter_frames
    import numpy as np
    report={}
    for key in OUTPUTS:
        path=Path(artifacts[key]); info=probe_video(path)
        if info.is_vfr:raise ValueError(f"{key} must be constant-rate Wan conditioning; received VFR video")
        if expected_fps is not None and abs(info.fps-expected_fps)>.05:
            raise ValueError(f"{key} is {info.fps} FPS; expected source FPS {expected_fps}")
        if expected_frames is not None and info.frame_count!=expected_frames:
            raise ValueError(f"{key} has {info.frame_count} frames; expected {expected_frames}")
        report[key]={"path":str(path),"fps":info.fps,"frames":info.frame_count,"size":[info.width,info.height]}
    counts={report[key]["frames"] for key in OUTPUTS}; fps={round(report[key]["fps"],3) for key in OUTPUTS}
    primary_sizes={tuple(report[key]["size"]) for key in ("pose_video","background_video","mask_video")}
    if len(counts)!=1 or len(fps)!=1 or len(primary_sizes)!=1:raise ValueError(f"Wan condition videos must have aligned frame count/FPS and pose/background/mask geometry: {report}")
    if tuple(report["face_video"]["size"])!=(512,512):raise ValueError(f"Wan Animate face conditioning must be cropped to 512x512, got {report['face_video']['size']}")
    white=black=False
    for sample in iter_frames(artifacts["mask_video"]):
        sample=sample.astype(np.int16)
        if np.max(np.abs(sample[:,:,0]-sample[:,:,1]))>2 or np.max(np.abs(sample[:,:,1]-sample[:,:,2]))>2:
            raise ValueError("Wan replacement mask video must be a grayscale RGB mask (same value in each channel)")
        white=white or bool(np.any(sample[:,:,0]>=240))
        black=black or bool(np.any(sample[:,:,0]<=15))
    if not white or not black:
        raise ValueError("Wan replacement mask must contain both white generation pixels and black preserved pixels")
    report["mask_convention"]={"white":"generate/replace","black":"preserve/condition","channels":"grayscale RGB"}
    return report


def preprocess_wan_animate(source: str | Path, reference: str | Path, output_dir: str | Path,
                           wan_repo: str | Path, checkpoint_dir: str | Path,
                           fps: int = -1, resolution_area: tuple[int, int] = (1280, 720),
                           timeout: int = 7200) -> dict[str, Any]:
    """Generate Wan-native pose, face, mask and background videos using upstream code."""
    source, reference = Path(source).resolve(), Path(reference).resolve()
    repo, ckpt = Path(wan_repo).resolve(), Path(checkpoint_dir).resolve()
    script = repo / "wan" / "modules" / "animate" / "preprocess" / "preprocess_data.py"
    required = [script, ckpt / "det" / "yolov10m.onnx", ckpt / "pose2d" / "vitpose_h_wholebody.onnx",
                ckpt / "sam2" / "sam2_hiera_large.pt"]
    for path in (source, reference, *required):
        if not path.exists():
            raise FileNotFoundError(f"Wan Animate preprocessing dependency is missing: {path}")
    # Upstream interprets this as a target area and can upscale even a tiny
    # source. Do not inflate a low-resolution acceptance clip to 720p.
    source_info=probe_video(source)
    resolution_area=conditioning_area(resolution_area,source_info.width,source_info.height)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    from .cache import content_identity,software_signature
    fingerprint={"source":_sha256_file(source),"reference":_sha256_file(reference),
        "preprocessor_source":content_identity(script.parent),"checkpoint_files":[(str(p.relative_to(ckpt)),content_identity(p)) for p in required[1:]],"software":software_signature(),
        "fps":fps,"resolution_area":list(resolution_area)}
    fingerprint["key"]=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
    existing={name:str(output/filename) for name,filename in OUTPUTS.items()}
    manifest=output/"preprocess_manifest.json"
    if all(Path(path).is_file() for path in existing.values()) and manifest.is_file():
        try:
            if json.loads(manifest.read_text(encoding="utf-8")).get("key")==fingerprint["key"]:
                validate_wan_conditioning(existing)
                return existing
        except (OSError,ValueError,KeyError,json.JSONDecodeError):pass
    cmd = [sys.executable, str(script), "--ckpt_path", str(ckpt), "--video_path", str(source),
           "--refer_path", str(reference), "--save_path", str(output), "--resolution_area",
           str(int(resolution_area[0])), str(int(resolution_area[1])), "--fps", str(int(fps)),
           "--replace_flag"]
    completed = subprocess.run(cmd, cwd=script.parent, capture_output=True, text=True,
                               timeout=timeout, check=False)
    if completed.returncode:
        raise RuntimeError("Official Wan Animate preprocessing failed (exit " + str(completed.returncode) + "):\n" +
                           completed.stdout[-4000:] + "\n" + completed.stderr[-8000:])
    artifacts = {name: str(output / filename) for name, filename in OUTPUTS.items()}
    infos = {}
    for key, path in artifacts.items():
        info = probe_video(path)
        if info.frame_count <= 0 or info.width <= 0 or info.height <= 0:
            raise RuntimeError(f"Wan preprocessing created an invalid {key}: {path}")
        infos[key] = {"fps": info.fps, "frames": info.frame_count, "width": info.width, "height": info.height}
    counts = {item["frames"] for item in infos.values()}
    if len(counts) != 1:
        raise RuntimeError(f"Wan Animate conditioning videos do not have aligned frame counts: {infos}")
    validate_wan_conditioning(artifacts)
    manifest.write_text(json.dumps(fingerprint|{"outputs":infos},indent=2),encoding="utf-8")
    return {**artifacts, "metadata": infos, "preprocessor": "Wan-Video/Wan2.2 official preprocess_data.py"}
