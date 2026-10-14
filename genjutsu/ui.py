"""High-level local Gradio application; model-specific conditioning stays automatic."""
from __future__ import annotations
import json
from pathlib import Path


def launch(config_path="config.yaml", server_name="127.0.0.1", port=7860):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install genjutsu-oss[ui] to launch the local interface") from exc
    from .pipeline import analyze_video, transform_video
    from .controls import Operation

    def analyze(source):
        if not source:return "Upload a source video first."
        path=source if isinstance(source,str) else source.get("path")
        try:
            controls,_=analyze_video(path)
            return json.dumps({"width":controls.width,"height":controls.height,"fps":controls.fps,
                "frames":controls.frame_count,"duration_seconds":controls.duration,"audio":controls.audio_metadata,
                "shot_cuts_frame_indices":controls.scene_cuts},indent=2)
        except Exception as exc:return f"Analysis failed: {type(exc).__name__}: {exc}"

    def run(source,operation,backend_choice,prompt,negative,references,category,selection_prompt,mask,seed,shot_json,output,progress=gr.Progress()):
        if not source:return None,"Upload a source video first."
        source_path=source if isinstance(source,str) else source.get("path")
        ref_paths=[]
        if references:
            for ref in references:
                ref_paths.append(ref if isinstance(ref,str) else ref.get("path"))
        settings={"reference_category":category,"selection_prompt":selection_prompt or None}
        if backend_choice!="Use config default":settings["backend_override"]=backend_choice
        if shot_json and shot_json.strip():
            try:settings["shots"]=json.loads(shot_json)
            except json.JSONDecodeError as exc:return None,f"Per-shot JSON is invalid: {exc}"
        mask_path=mask if mask and isinstance(mask,str) else (mask.get("path") if mask else None)
        out=Path(output or "output/transformed.mp4").expanduser().resolve()
        try:
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
                source=gr.Video(label="Source video",sources=["upload"],format="mp4")
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
    demo.queue(default_concurrency_limit=1).launch(server_name=server_name,server_port=port)
