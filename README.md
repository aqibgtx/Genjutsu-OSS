# Genjutsu-OSS

An open-source, model-independent video transformation pipeline with ComfyUI integration. It reuses measured information from source footage and delegates generation to real video models. It is not the source code or model implementation of any proprietary Genjutsu-style product.

## Local capabilities

- FFmpeg video probing, frame decoding/encoding, optional source-audio muxing.
- Scene-cut detection, per-shot processing/caches/retries and overrides, Grounding DINO text selection followed by SAM 2 propagation, 2D camera-motion estimates, Farneback optical flow, feathered compositing, local color matching, temporal diagnostics, project archives.
- Real model adapters for Hugging Face Diffusers Wan VACE (currently documented Wan 2.1 VACE checkpoints), Wan 2.2 Animate, and Wan image-to-video. Inference needs compatible model weights and runtime; installation does not download weights.
- Optional SAM 2, Grounding DINO, Depth Anything, MediaPipe Holistic, InsightFace, and Real-ESRGAN integrations when their dependencies and local model files are supplied.
- ComfyUI custom nodes, Python API, CLI, optional Gradio UI, and API-format workflows.

See [CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md) and [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for tested behavior and limits.

## Simple ComfyUI workflow

For a car edit, select **Genjutsu / ObjectSwap**, set target to `car`, choose reference layout `images`, upload the car photos and start with `draft`. Multiple references are passed to VACE together; text such as `@Image1` is not a reference-role binding API. ObjectSwap edits the selected car and preserves unmasked background; use EnvironmentRecast separately to change the background. Six-reference short drafts need the adapter's reference-mask padding compatibility fix, included here.

Select **Genjutsu / CharacterSwap** in Workflows. Use its upload buttons for the source video and reference image, set target to `face` for face-only edits, and select `face_sheet_2x3` for an explicitly arranged six-photo reference sheet. Automatic detection, tracking and the four official Animate conditioning videos run internally. No manually painted masks are required by this workflow.

The default `draft` renders only five frames, at most 144×256, with one inference step. It tests the pipeline and reports identity measurements; it does **not** establish production quality. `source` uses the configured steps and retains source geometry/timing, with identity acceptance enabled by default. A failed identity gate stops publication of the accepted video. Rejected candidates are retained as disk-backed frames for inspection.

The six sidebar workflows use the registered `GenjutsuTransform` and `GenjutsuVideoOutput` nodes. Existing individual-node workflows remain in `workflows/advanced`. Configure `genjutsu-runtime.yaml` beside ComfyUI once; the installer supplies a local-model profile without downloading weights. FullGenjutsu requires separate categorized CHARACTER and LOCATION references and runs staged processing. It has not yet passed real full-recast acceptance.

Current real evidence: Wan Animate automatic preprocessing and five-frame generation execute on the existing RTX A6000 Pod; the one-step previews **failed identity acceptance**. The entire 46-section specification is not yet production-validated. See the reports for limits, including CFR-only Animate and the 256-frame shot bound.

## Install on Windows

Install Python 3.10+ and FFmpeg (`ffmpeg` and `ffprobe` on `PATH`), then create an isolated environment:

```powershell
py -3.10 -m venv .venv-local
.\.venv-local\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[video,ui,api,test]"
python install.py --check
```

For Wan, install a PyTorch build matching your CUDA runtime first, then `python -m pip install -e ".[wan]"`. Real Pod execution uses Diffusers 0.41.0, Transformers 5.17.0 and PyTorch 2.10.0+cu128. Preserve an existing ComfyUI CUDA environment with pip constraints instead of upgrading its Torch stack. This installs Python code only, not model weights.

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

### Verified RunPod ComfyUI smoke test

One real Wan2.1 VACE 1.3B Full Frame Test completed on an RTX A6000 with ComfyUI 0.30.0, Diffusers 0.41.0 and PyTorch 2.10.0+cu128. The 202-frame 720×1280 output retained 30 FPS timestamps and source audio. See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for exact settings and boundaries; other GPU features remain unverified.

Wan prompt processing requires `ftfy`, now included in the Wan dependency manifests. For the existing tested Pod it was installed with `python -m pip install --no-deps ftfy==6.3.1`, followed by `pip check` and a ComfyUI restart. Do not apply the repository's older dependency pins over an existing CUDA environment without checking compatibility.

The prepared graphical workflow appears as **Genjutsu / Wan VACE 1.3B - Full Frame Test** in that Pod's server-backed Workflows sidebar. It uses one all-white mask per decoded source frame, which allows changes anywhere in the frame. For selective editing, use target-shaped masks instead. The six repository workflows are API JSON; installation alone does not open them on the canvas.

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

Video analysis and transformation now use lossless disk-backed frame sequences. Gradio accepts one mask PNG, a numerically ordered PNG sequence, or one mask video; mixed image/video uploads are rejected. Source uploads use a file component so the UI does not transcode footage during upload. Mask videos are aligned by decoded frame index and must match source dimensions and frame count.

`VideoInfo` and `ControlPackage` retain rational FPS, time base, presentation timestamps, and frame durations. Exports preserve source VFR timestamps and the final frame's duration; audio retains its offset relative to the first video frame. Each shot receives local timestamps/indices, source indices/time range, sliced masks/pose/depth/tracks/flow, and camera transitions that exclude cuts. C-contiguous numeric `.npy` flow files are sliced into per-shot files with bounded byte copies. Wan VACE conditions on frame order; it does not receive variable time intervals as a learned control. Wan Animate requires CFR input and rejects VFR rather than silently retiming it. Explicit retiming must be done before selecting Animate.

Decoding, shot aggregation, cache reads, temporal smoothing, compositing, and encoding avoid whole-video RGB arrays. Lossless PNG stores reduce scratch space; floating-point controls use NPY. VACE materializes only each configured inference chunk. Animate/I2V retain the upstream shot-level loading behavior and enforce `backend.max_shot_frames` (default 256) to reject unbounded jobs. SAM 2 uses `selection.chunk_frames` (default 32), one-frame window overlap, CPU offload, and real mask reinitialization between windows. A fully lost target at a window boundary fails with a reinitialization request; this model path still needs GPU validation. The ComfyUI IMAGE tensor loader remains an explicit eager tensor interface.

Scratch storage defaults to the system temporary directory; provide sufficient local disk space and set `TEMP`/`TMP` on Windows or `TMPDIR` on Linux before starting Python to select a scratch disk. Final outputs and shot state/source files remain beside the output video. Generation caches remain under the configured cache directory. Compression size depends on footage, and model VRAM/host RAM requirements still require real inference measurements.

Generation cache keys include source/mask bytes, reference contents and weights, all effective controls/settings, checkpoint/config/tokenizer contents, requested/resolved revisions, adapter source hashes, and installed runtime versions. `backend.revision` is passed to actual Diffusers loaders; local snapshot resolution avoids floating revisions during a run. Hashing large local checkpoints adds startup I/O. Unresolved checkpoints disable generation-cache reuse. ComfyUI reuse requires explicit `backend.model_fingerprint` and `backend.software_fingerprint` values that identify the remote checkpoint bytes and runtime; update them whenever the server changes. Old NPZ generation caches are not reused.

The local tests include real HTTP Gradio uploads, independent FFmpeg VFR fixtures, 24000/1001, 30000/1001 and 60000/1001 FPS, delayed/offset audio, per-shot retry/resume and invalidation, and isolated-process RSS measurements on 1080p clips. Pipeline tests inject a test-only OpenCV grayscale filter to exercise orchestration; they do not validate Wan inference. See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for measured results and [RUNPOD_VALIDATION.md](RUNPOD_VALIDATION.md) for real-model acceptance commands.

Depth Anything currently runs framewise and does not guarantee temporal depth consistency. Camera analysis estimates 2D affine transforms, not calibrated 3D trajectories. Optical-flow mask propagation can drift through difficult occlusions. Identity scoring is evaluative only; no correction loop is implemented. Occlusion suppression needs an explicit foreground mask and local depth model. Learned temporal repair, RIFE interpolation, and reliable face restoration are incomplete. See the capability and validation reports.

```powershell
python -m pytest
python -m compileall -q genjutsu ComfyUI-Genjutsu
```

No fabricated inference backend exists. Missing model files, preprocessed controls, masks, or supported APIs produce explicit errors.

## License

Original project code is Apache-2.0. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md); exact model checkpoints and optional components require their own license review.
