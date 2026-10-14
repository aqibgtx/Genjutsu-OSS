# Genjutsu-OSS

An open-source, model-independent video transformation pipeline with ComfyUI integration. It reuses measured information from source footage and delegates generation to real video models. It is not the source code or model implementation of any proprietary Genjutsu-style product.

## Local capabilities

- FFmpeg video probing, frame decoding/encoding, optional source-audio muxing.
- Scene-cut detection, per-shot processing/caches/retries and overrides, Grounding DINO text selection followed by SAM 2 propagation, 2D camera-motion estimates, Farneback optical flow, feathered compositing, local color matching, temporal diagnostics, project archives.
- Real model adapters for Hugging Face Diffusers Wan VACE (currently documented Wan 2.1 VACE checkpoints), Wan 2.2 Animate, and Wan image-to-video. Inference needs compatible model weights and runtime; installation does not download weights.
- Optional SAM 2, Grounding DINO, Depth Anything, MediaPipe Holistic, InsightFace, and Real-ESRGAN integrations when their dependencies and local model files are supplied.
- ComfyUI custom nodes, Python API, CLI, optional Gradio UI, and API-format workflows.

See [CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md) and [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for tested behavior and limits.

## Install on Windows

Install Python 3.10+ and FFmpeg (`ffmpeg` and `ffprobe` on `PATH`), then create an isolated environment:

```powershell
py -3.10 -m venv .venv-local
.\.venv-local\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[video,ui,api,test]"
python install.py --check
```

For Wan, install a PyTorch build matching your CUDA runtime first, then `python -m pip install -e ".[wan]"`. The tested API target is Diffusers 0.37.1 with Transformers 4.x. This installs Python code only, not model weights.

Install custom nodes with `python install.py --comfyui-path C:\ComfyUI`, then restart ComfyUI. Workflow paths must refer to files accessible to the ComfyUI process.

## Real model configuration

`config.yaml` selects the ComfyUI HTTP backend but leaves its API workflow unset. Configure an actual API-format workflow, or copy `config.wan-vace.yaml` and set `backend.model_id` to a local model directory. Model loaders use `local_files_only` by default. To let Diffusers download a model, explicitly set `backend.allow_download: true` after authorizing that download. Example:

```powershell
genjutsu check --config config.wan-vace.yaml --video input.mp4
genjutsu analyze input.mp4 --output controls.json
genjutsu transform input.mp4 output.mp4 --config config.wan-vace.yaml --operation object_swap --reference watch.jpg --mask tracked-watch-masks --prompt "A brushed steel watch with a black dial"
```

Wan VACE expects a per-frame mask (white = generate, black = preserve); references are optional in the real Diffusers API. For full environment recast, the pipeline makes a full-white mask. Wan Animate calls the official Wan 2.2 `preprocess_data.py` automatically when `backend.wan_animate_preprocess` contains a local Wan2.2 checkout and local preprocessing checkpoints. That tool requires `det/yolov10m.onnx`, `pose2d/vitpose_h_wholebody.onnx`, and `sam2/sam2_hiera_large.pt`; it produces aligned `src_pose.mp4`, `src_face.mp4`, `src_bg.mp4`, and `src_mask.mp4` videos. Pose/background/mask share the processed video dimensions; face frames are 512×512. Genjutsu validates the exact timing and geometry before inference. Configure `selection.detector_model`, `selection.sam2_config`, and `selection.sam2_checkpoint` to enable automatic text-based target selection for other selective transforms. No weights are downloaded implicitly. Wan 2.2 VACE is not claimed: the VACE adapter targets documented Wan 2.1 checkpoints.

```yaml
backend:
  type: auto
  use_wan_animate_for_character: true
  wan_animate_preprocess:
    wan_repo: C:/models/Wan2.2
    checkpoint_dir: C:/models/wan22-preprocess
    fps: -1
    resolution_area: [1280, 720]
selection:
  detector_model: C:/models/grounding-dino
  sam2_config: C:/models/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
  sam2_checkpoint: C:/models/sam2/sam2.1_hiera_large.pt
  device: cuda
```

The supplied Wan VACE profile uses the 1.3B checkpoint variant, float16, and CPU offload as a lower-memory starting point; this profile has not been GPU validated. For MediaPipe Holistic, install the `pose` extra (`python -m pip install -e ".[pose]"`); it constrains protobuf to a compatible major version.

Run the UI with `genjutsu ui --config config.yaml`. It supports uploads, references, text target selection, per-shot JSON overrides, and optional masks; it does not ask users to create Wan conditioning videos. Start the developer API with `genjutsu api`; see [docs/API.md](docs/API.md).

## Linux / GPU deployment

Use Python 3.10+, FFmpeg, and PyTorch matched to the host CUDA runtime. Install with `pip install -e ".[video,api]"`, then install the selected model runtime and place weights on local storage. The same CLI/config applies. Set device, dtype, chunk overlap, cache, and ComfyUI URL for the host. Do not include private credentials or large weights in the repository.

## CLI

```text
genjutsu analyze INPUT.mp4 --output controls.json
genjutsu transform INPUT.mp4 OUTPUT.mp4 --operation character_swap --prompt PROMPT --reference character.png --mask masks/ --config config.yaml
genjutsu project create INPUT.mp4 project.genjutsu --include-source
genjutsu project inspect project.genjutsu
genjutsu check --config config.yaml --video INPUT.mp4
genjutsu ui
genjutsu api
```

GPU validation suites are executable with `python scripts/validate_runpod.py --suite wan-vace --assets-dir /workspace/validation --models-dir /workspace/models --run-inference`; see [RUNPOD_VALIDATION.md](RUNPOD_VALIDATION.md). For a live ComfyUI server use `python scripts/validate_comfyui.py --url http://127.0.0.1:8188`; add `--queue --assets-dir ...` to submit real workflows.

`.genjutsu` is a ZIP archive containing the control manifest and checksummed references; the source video is included only with `--include-source`. Generated/intermediate frames cache under `.genjutsu-cache` by default. Caches are not archival backups.

## Limits

Depth Anything currently runs framewise and does not guarantee temporal depth consistency. Camera analysis estimates 2D affine transforms, not calibrated 3D trajectories. Optical-flow mask propagation can drift through difficult occlusions. Identity scoring is evaluative only; no correction loop is implemented. Occlusion suppression needs an explicit foreground mask and local depth model. Learned temporal repair, RIFE interpolation, and reliable face restoration are incomplete. See the capability and validation reports.

```powershell
python -m pytest
python -m compileall -q genjutsu ComfyUI-Genjutsu
```

No fabricated inference backend exists. Missing model files, preprocessed controls, masks, or supported APIs produce explicit errors.

## License

Original project code is Apache-2.0. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md); exact model checkpoints and optional components require their own license review.
