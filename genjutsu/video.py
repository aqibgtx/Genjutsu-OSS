"""Timestamp-aware video I/O with bounded decoding and encoding memory."""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
import json
import subprocess
import tempfile
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
    fps_rational: str = "0/1"
    time_base: str = "1/1000000"
    frame_pts: list[int] = field(default_factory=list)
    frame_durations: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    start_time: float = 0.0
    is_vfr: bool = False


def _av():
    try:
        import av
        return av
    except ImportError as exc:
        raise RuntimeError("Install genjutsu-oss[video] (PyAV) for timestamp-preserving video I/O") from exc


def _run(args):
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required executable not found: {args[0]}. Install FFmpeg on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or f"Command failed: {args[0]}") from exc


def probe_video(path: str | Path) -> VideoInfo:
    """Read actual presentation timestamps; never estimate VFR count from FPS."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(_run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]).stdout)
    video = next((s for s in raw.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"No video stream in {path}")
    rate = Fraction(video.get("avg_frame_rate", "0/1"))
    tb = Fraction(video["time_base"])
    pts, durations = [], []
    with _av().open(str(path)) as container:
        container.streams.video[0].thread_count = 1
        for frame in container.decode(video=0):
            if frame.pts is None:
                raise ValueError("Video frame has no presentation timestamp; explicitly repair source timing first")
            pts.append(round(Fraction(frame.pts) * frame.time_base / tb))
            durations.append(round(Fraction(frame.duration or 0) * frame.time_base / tb))
    if not pts:
        raise ValueError("Video contains no decoded frames")
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("Video presentation timestamps must be strictly increasing")
    gaps = [b-a for a, b in zip(pts, pts[1:])]
    nominal = round(1/rate/tb) if rate > 0 else (gaps[-1] if gaps else 1)
    durations[:-1] = gaps
    stream_end = round(Fraction(video.get("duration", "0"))/tb) if video.get("duration") else 0
    remaining = stream_end - (pts[-1]-pts[0])
    durations[-1] = durations[-1] or (remaining if remaining > 0 else (gaps[-1] if gaps else nominal))
    duration = float((pts[-1]-pts[0]+durations[-1])*tb)
    audio = next((s for s in raw.get("streams", []) if s.get("codec_type") == "audio"), None)
    metadata = {} if audio is None else {k: audio[k] for k in ("codec_name", "sample_rate", "channels", "channel_layout", "bit_rate", "start_time", "duration", "time_base") if k in audio}
    # CFR quantization can alternate by one tick (e.g. Matroska 29.97).
    vfr = bool(gaps and max(gaps)-min(gaps) > 1)
    return VideoInfo(str(path), float(rate), int(video["width"]), int(video["height"]), len(pts), duration,
                     audio is not None, video.get("codec_name"), metadata, str(rate), str(tb), pts,
                     durations, [float((p-pts[0])*tb) for p in pts], float(pts[0]*tb), vfr)


def iter_frames(path: str | Path, start: int = 0, limit: int | None = None):
    if start < 0 or (limit is not None and limit < 0):
        raise ValueError("start and limit must be non-negative")
    if limit == 0:
        return
    with _av().open(str(path)) as container:
        container.streams.video[0].thread_count = 1
        emitted = 0
        for index, frame in enumerate(container.decode(video=0)):
            if index < start:
                continue
            yield frame.to_ndarray(format="rgb24")
            emitted += 1
            if limit is not None and emitted >= limit:
                break


def read_frames(path: str | Path, start: int = 0, limit: int | None = None, *, disk_backed: bool = False, directory=None):
    """Compatibility eager reader; production orchestration opts into disk backing."""
    info = probe_video(path)
    frames = iter_frames(path, start, limit)
    if disk_backed or directory is not None:
        from .frame_store import FrameStore
        return FrameStore.from_frames(frames, directory), info
    return list(frames), info


def write_video(frames, output: str | Path, fps, audio_from=None, codec="h264", crf=18, pixel_format="yuv420p", *,
                timestamps=None, frame_durations=None, time_base=None, audio_start_time=0.0) -> Path:
    """Stream RGB frames with explicit PTS; omit timestamps only for deliberate CFR.

    FPS accepts a Fraction or rational string. Audio shifts by the original first
    video PTS, preserving its relative offset, and ends at the video endpoint.
    """
    if not len(frames):
        raise ValueError("Cannot encode an empty frame sequence")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoders = {"h264": "libx264", "h265": "libx265", "prores": "prores_ks", "vp9": "libvpx-vp9"}
    extensions = {"h264": {".mp4", ".mov", ".mkv"}, "h265": {".mp4", ".mov", ".mkv"}, "prores": {".mov", ".mkv"}, "vp9": {".webm", ".mkv"}}
    if codec not in encoders:
        raise ValueError(f"Unsupported codec {codec}")
    if output.suffix.lower() not in extensions[codec]:
        raise ValueError(f"{codec} output must use {', '.join(sorted(extensions[codec]))}")
    rate = Fraction(str(fps)).limit_denominator(1000000)
    if rate <= 0:
        raise ValueError("FPS must be positive")
    tb = Fraction(str(time_base)) if time_base else (Fraction(1, 1000000) if timestamps is not None else 1/rate)
    if timestamps is None:
        pts = [round(i/rate/tb) for i in range(len(frames))]
    else:
        if len(timestamps) != len(frames):
            raise ValueError("Timestamp count must match frame count")
        origin = Fraction(str(timestamps[0]))
        pts = [round((Fraction(str(t))-origin)/tb) for t in timestamps]
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("Timestamps must be strictly increasing at the chosen time base")
    if frame_durations is not None and len(frame_durations) != len(frames):
        raise ValueError("Duration count must match frame count")
    last = round(Fraction(str(frame_durations[-1]))/tb) if frame_durations is not None else round(1/rate/tb)
    if last <= 0:
        raise ValueError("Last-frame duration must be positive")
    durations = [b-a for a, b in zip(pts, pts[1:])] + [last]
    h, w = frames[0].shape[:2]
    av = _av()
    with tempfile.TemporaryDirectory(prefix="genjutsu-encode-", dir=output.parent) as tmp:
        # The Windows PyAV x265 wheel can corrupt the process heap. Encode a
        # lossless H.264 intermediate for other codecs and use system FFmpeg.
        encoded = Path(tmp) / "video.mp4"
        options = {"movflags": "+faststart", "video_track_timescale": str(tb.denominator)}
        with av.open(str(encoded), "w", options=options) as container:
            stream = container.add_stream("libx264", rate=rate)
            stream.width, stream.height = w, h
            stream.pix_fmt = pixel_format if codec == "h264" else "yuv444p"
            stream.time_base = stream.codec_context.time_base = tb
            stream.codec_context.thread_count = 1
            # Disable encoder lookahead/B-frame queues: otherwise even streaming
            # RGB input can retain dozens of 1080p frames inside libx264.
            stream.options = {"crf": str(crf if codec == "h264" else 0), "tune": "zerolatency"}
            by_pts = {p*tb: d*tb for p, d in zip(pts, durations)}

            def mux(packet):
                duration = by_pts.get(packet.pts*packet.time_base)
                if duration is not None:
                    packet.duration = round(duration/packet.time_base)
                container.mux(packet)

            for index, array in enumerate(frames):
                if array.shape != (h, w, 3) or array.dtype != np.uint8:
                    raise ValueError("Video frames must have constant geometry and RGB uint8 dtype")
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                frame.pts, frame.time_base, frame.duration = pts[index], tb, durations[index]
                for packet in stream.encode(frame):
                    mux(packet)
            for packet in stream.encode():
                mux(packet)
        if audio_from or codec != "h264":
            duration = float((pts[-1]+last)*tb)
            args = ["ffmpeg", "-y", "-v", "error", "-copyts", "-i", str(encoded)]
            if audio_from:
                args += ["-itsoffset", str(-audio_start_time), "-i", str(audio_from)]
            args += ["-map", "0:v:0"]
            if audio_from:
                args += ["-map", "1:a?", "-c:a", "libopus" if codec == "vp9" else ("pcm_s16le" if codec == "prores" else "aac")]
            args += ["-c:v", "copy" if codec == "h264" else encoders[codec]]
            if codec != "h264":
                args += ["-fps_mode", "passthrough", "-enc_time_base", "demux", "-pix_fmt", "yuv422p10le" if codec == "prores" else pixel_format]
                if codec in {"h265", "vp9"}:
                    args += ["-crf", str(crf)]
                if codec == "h265":
                    params="pools=none:frame-threads=1:wpp=0:log-level=error"
                    if min(w,h)<64:
                        params+=":ctu=16:asm=0"
                    args += ["-threads", "1", "-x265-params", params]
            args += ["-t", f"{duration:.12f}", "-avoid_negative_ts", "disabled"]
            if output.suffix.lower() in {".mp4", ".mov"}:
                args += ["-movflags", "+faststart", "-video_track_timescale", str(tb.denominator)]
            _run(args+[str(output)])
        else:
            encoded.replace(output)
    return output


def timing_kwargs(info):
    tb = Fraction(info.time_base)
    return {"timestamps": info.timestamps, "frame_durations": [float(d*tb) for d in info.frame_durations],
            "time_base": info.time_base, "audio_start_time": getattr(info, "start_time", getattr(info, "source_start_time", 0.0))}


def extract_frames(path: str | Path, directory: str | Path) -> list[Path]:
    from PIL import Image
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, frame in enumerate(iter_frames(path)):
        target = directory/f"{i+1:08d}.png"
        Image.fromarray(frame).save(target)
        paths.append(target)
    return paths
