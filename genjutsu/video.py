"""FFmpeg-backed video I/O. Frames are RGB uint8 arrays; audio stays in the source during mux."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, shutil, subprocess
import numpy as np

@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration: float
    has_audio: bool
    codec: str | None = None
    audio_metadata: dict | None = None

def _run(args: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"Required executable not found: {args[0]}. Install FFmpeg and add it to PATH.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stderr.strip() or f"Command failed: {args[0]}") from e

def probe_video(path: str | Path) -> VideoInfo:
    path = Path(path).resolve()
    if not path.is_file(): raise FileNotFoundError(path)
    raw = _run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]).stdout
    info = json.loads(raw); streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video: raise ValueError(f"No video stream in {path}")
    rate = video.get("avg_frame_rate", "0/1").split("/")
    fps = float(rate[0]) / max(float(rate[1]), 1.0)
    duration = float(video.get("duration", info.get("format", {}).get("duration", 0)) or 0)
    count = int(video.get("nb_frames") or round(duration * fps))
    audio=next((s for s in streams if s.get("codec_type")=="audio"),None)
    audio_meta={} if audio is None else {k:audio.get(k) for k in ("codec_name","sample_rate","channels","channel_layout","bit_rate") if audio.get(k) is not None}
    return VideoInfo(str(path), fps, int(video["width"]), int(video["height"]), count, duration,
                     audio is not None, video.get("codec_name"),audio_meta)

def read_frames(path: str | Path, start: int = 0, limit: int | None = None) -> tuple[list[np.ndarray], VideoInfo]:
    info = probe_video(path)
    vf = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select='gte(n,{start})'", "-vsync", "0"]
    if limit is not None: vf += ["-frames:v", str(limit)]
    vf += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    try: raw = subprocess.run(vf,check=True,capture_output=True).stdout
    except FileNotFoundError as e: raise RuntimeError("FFmpeg not found on PATH") from e
    except subprocess.CalledProcessError as e: raise RuntimeError(e.stderr.decode(errors="replace")) from e
    frame_bytes = info.width * info.height * 3
    if len(raw) % frame_bytes: raise RuntimeError("FFmpeg returned a truncated raw frame stream")
    frames = [np.frombuffer(raw[i:i+frame_bytes], np.uint8).reshape(info.height, info.width, 3).copy()
              for i in range(0, len(raw), frame_bytes)]
    return frames, info

def write_video(frames: list[np.ndarray], output: str | Path, fps: float, audio_from: str | Path | None = None,
                codec: str = "h264", crf: int = 18, pixel_format: str = "yuv420p") -> Path:
    if not frames: raise ValueError("Cannot encode an empty frame sequence")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    codec_args = {"h264": ["-c:v", "libx264"], "h265": ["-c:v", "libx265"], "prores": ["-c:v", "prores_ks"], "vp9": ["-c:v", "libvpx-vp9"]}
    if codec not in codec_args: raise ValueError(f"Unsupported codec {codec}; choose {', '.join(codec_args)}")
    suffix=output.suffix.lower()
    if codec=="vp9" and suffix not in (".webm",".mkv"): raise ValueError("VP9 output must use .webm or .mkv")
    if codec=="prores" and suffix not in (".mov",".mkv"): raise ValueError("ProRes output must use .mov or .mkv")
    if codec in ("h264","h265") and suffix not in (".mp4",".mov",".mkv"): raise ValueError("H.264/H.265 output must use .mp4, .mov or .mkv")
    args = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if audio_from:
        args += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a?"]
    args += codec_args[codec]
    if codec in ("h264", "h265", "vp9"): args += ["-crf", str(crf)]
    if codec=="h265": args += ["-threads","1","-x265-params","pools=1:frame-threads=1"]
    if audio_from: args += ["-c:a", "libopus" if codec=="vp9" else ("pcm_s16le" if codec=="prores" else "aac")]
    if codec=="prores": pixel_format="yuv422p10le"
    # Set exact video duration without allowing a slightly short AAC packet to
    # truncate the final encoded frame (the old -shortest behavior did this).
    args += ["-t", f"{len(frames)/float(fps):.9f}"]
    if output.suffix.lower() in (".mp4",".mov"): args += ["-movflags", "+faststart"]
    args += ["-pix_fmt", pixel_format, str(output)]
    payload = b"".join(np.ascontiguousarray(f, dtype=np.uint8).tobytes() for f in frames)
    try:
        subprocess.run(args, input=payload, check=True, capture_output=True)
    except FileNotFoundError as e: raise RuntimeError("FFmpeg not found on PATH") from e
    except subprocess.CalledProcessError as e: raise RuntimeError(e.stderr.decode(errors="replace")) from e
    return output

def extract_frames(path: str | Path, directory: str | Path) -> list[Path]:
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-i", str(path), "-vsync", "0", str(directory / "%08d.png")])
    return sorted(directory.glob("*.png"))
