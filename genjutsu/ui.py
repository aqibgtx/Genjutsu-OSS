"""High-level local Gradio application; model-specific conditioning stays automatic."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg"}


def upload_path(value):
    """Normalize Gradio filepath, FileData, and legacy upload dictionaries."""
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value)
    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    else:
        value = getattr(value, "path", None) or getattr(value, "name", None)
    if not isinstance(value, (str, os.PathLike)) or not value:
        raise ValueError("Upload is missing a file path")
    return os.fspath(value)


def upload_paths(value):
    if value is None or value == []:
        return []
    return [upload_path(item) for item in (value if isinstance(value, (list, tuple)) else [value])]


def mask_upload(value):
    """One image/video or an ordered PNG sequence; reject ambiguous mixtures."""
    paths = upload_paths(value)
    if not paths:
        return None
    suffixes = [Path(p).suffix.lower() for p in paths]
    if len(paths) == 1 and (suffixes[0] == ".png" or suffixes[0] in VIDEO_SUFFIXES):
        return paths[0]
    if any(ext != ".png" for ext in suffixes):
        raise ValueError("Upload one mask video, one PNG, or a sequence containing only PNG files")
    # Filename order, not random Gradio upload-directory order. Numeric sorting
    # also handles frame1.png, frame2.png, frame10.png.
    return sorted(paths, key=lambda p: [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", Path(p).name)])


def build_ui(config_path="config.yaml"):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install genjutsu-oss[ui] to launch the local interface") from exc
    from .pipeline import analyze_video, transform_video
    from .controls import Operation

    def analyze(source):
        if not source:return "Upload a source video first."
        try:
            path=upload_path(source)
            controls,_=analyze_video(path)
            return json.dumps({"width":controls.width,"height":controls.height,"fps":controls.fps,
                "fps_rational":controls.fps_rational,"is_vfr":controls.is_vfr,"time_base":controls.time_base,
                "frames":controls.frame_count,"duration_seconds":controls.duration,"audio":controls.audio_metadata,
                "shot_cuts_frame_indices":controls.scene_cuts},indent=2)
        except Exception as exc:return f"Analysis failed: {type(exc).__name__}: {exc}"

    def run(source,operation,backend_choice,prompt,negative,references,category,selection_prompt,mask,seed,shot_json,output,progress=gr.Progress()):
        if not source:return None,"Upload a source video first."
        settings={"reference_category":category,"selection_prompt":selection_prompt or None}
        if backend_choice!="Use config default":settings["backend_override"]=backend_choice
        if shot_json and shot_json.strip():
            try:settings["shots"]=json.loads(shot_json)
            except json.JSONDecodeError as exc:return None,f"Per-shot JSON is invalid: {exc}"
        out=Path(output or "output/transformed.mp4").expanduser().resolve()
        try:
            source_path=upload_path(source)
            ref_paths=upload_paths(references)
            mask_path=mask_upload(mask)
            progress(0.1,desc="Analyzing video and planning shots")
            result=transform_video(source_path,str(out),operation,prompt or "",ref_paths,mask_path,
                config_path=config_path,negative_prompt=negative or "",seed=int(seed or 0),**settings)
            progress(1,desc="Complete")
            return str(result),f"Saved {result}"
        except Exception as exc:return None,f"Generation failed: {type(exc).__name__}: {exc}"

    with gr.Blocks(title="Genjutsu-OSS",theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Genjutsu-OSS\nUpload a source clip, describe the edit, add reference images, and run the configured local or ComfyUI backend. Pose, face, mask, and background conditioning for Wan Animate is prepared automatically from the official Wan preprocessing tool when configured.")
        with gr.Row():
            with gr.Column(scale=1):
                source=gr.File(label="Source video",file_types=["video"],type="filepath")
                analyze_button=gr.Button("Analyze shots and audio")
                analysis=gr.Code(label="Video analysis",language="json")
                operation=gr.Dropdown([o.value for o in Operation],value=Operation.CHARACTER_SWAP.value,label="Transformation")
                backend_choice=gr.Dropdown(["Use config default","auto","wan_vace","wan22","comfyui"],value="auto",label="Inference backend",info="Auto routes source-video edits to Wan VACE; enable Wan Animate in configuration for character/motion workflows.")
                selection=gr.Textbox(label="Select subject or object",placeholder="person, red bicycle, handbag…",info="Uses configured local Grounding DINO + SAM 2; leave blank only when supplying an edit mask.")
                references=gr.File(label="Reference images",file_count="multiple",file_types=["image"],type="filepath")
                category=gr.Dropdown(["CHARACTER","OBJECT","PRODUCT","OUTFIT","LOCATION","STYLE","PROP","TEXTURE","LOGO"],value="CHARACTER",label="Reference category")
                prompt=gr.Textbox(label="Edit prompt",lines=3,placeholder="Describe the intended appearance and scene.")
                negative=gr.Textbox(label="Negative prompt",lines=2)
                with gr.Accordion("Per-shot controls",open=False):
                    gr.Markdown('Optional JSON object keyed by shot index. Each shot can override `prompt`, `negative_prompt`, strengths, `selection_prompt`, `references`, and backend `settings`.')
                    shot_json=gr.Code(label="Shot overrides",language="json",value="{}")
                with gr.Accordion("Advanced",open=False):
                    mask=gr.File(label="Optional edit-mask video or PNG sequence",file_count="multiple",file_types=["image","video"],type="filepath")
                    seed=gr.Number(value=0,precision=0,label="Seed")
                    output=gr.Textbox(value="output/transformed.mp4",label="Output path")
                generate=gr.Button("Generate video",variant="primary")
            with gr.Column(scale=1):
                result_video=gr.Video(label="Generated video",interactive=False)
                status=gr.Textbox(label="Job status",lines=4)
        analyze_button.click(analyze,source,analysis)
        generate.click(run,[source,operation,backend_choice,prompt,negative,references,category,selection,mask,seed,shot_json,output],[result_video,status])
    return demo.queue(default_concurrency_limit=1)


def launch(config_path="config.yaml", server_name="127.0.0.1", port=7860):
    build_ui(config_path).launch(server_name=server_name,server_port=port)
