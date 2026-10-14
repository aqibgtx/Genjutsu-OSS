# RunPod validation runbook

These commands run real implementations against already-present local weights. Diffusers adapters use `local_files_only=True`. The current real execution results are in VALIDATION_REPORT.md; unexecuted suites below remain unvalidated.

## High-level sidebar acceptance

Run one workflow at a time against the existing ComfyUI server, with assets and paths on that server:

```bash
python scripts/validate_comfyui.py --url http://127.0.0.1:8188 --workflow CharacterSwap.json --assets-dir /workspace/validation --config /workspace/runpod-slim/ComfyUI/genjutsu-runtime.yaml --quality draft --queue
```

For a measured identity test, use `--quality source` and a separate runtime YAML with `steps: 20`, `segment_frame_length: 5`, `identity.max_correction_attempts: 0`, and `processing.shot_retries: 0`. Supply a five-frame 144×256 CFR `source.mp4` with audio and a single portrait `reference.png`. A face sheet needs `reference_layout: face_sheet_2x3` in the workflow inputs. The default identity lock requires every frame to reach cosine .55; rejection is a failed quality test, not a reason to weaken the threshold. Review the real rendered result in addition to scores. Explicit `--queue` runs real GPU inference and does not imply approval for multiple expensive candidates.

| Workflow / feature | Required assets and local models | Failure criteria |
|---|---|---|
| CharacterSwap / MotionTransfer | Moving face/person CFR source with audio, same-identity portrait references; Animate, official detector/pose/SAM2 and buffalo_l | Missing conditioning frame, altered timing, empty visible-target mask, identity below .55, motion loss or visible corruption fails. Use target `face` only for a face-only swap. |
| ObjectSwap / ProductSwap / outfit operation | Moving object/product/outfit source, matching categorized reference, VACE + DINO + SAM2 | Wrong detected target, edits outside mask, missing product features, timing/audio mismatch or visible temporal discontinuity fails. Outfit uses GenjutsuTransform `operation: outfit_swap`. |
| EnvironmentRecast | Source with visible subject, location reference; VACE + DINO + SAM2 | Subject regenerated, background not transformed as requested, camera/timing/audio mismatch fails. |
| FullGenjutsu | Source, separate CHARACTER and LOCATION references, optional OBJECT/PRODUCT/OUTFIT records; both Wan models and selection/identity weights | Any failed stage, lost identity, uncontrolled subject edit, target/reference mismatch or temporal discontinuity fails. Pass categorized records through `--references-json`. No STYLE support is claimed. |
| Depth-ordered occlusion | Source with hand/object crossing subject, foreground selection prompt, Depth Anything V2 + DINO + SAM2 | Foreground overwritten, hidden face leaking through, incorrect depth ordering or missing masks fails. Enable `occlusion` in the YAML. |
| GFPGAN restoration / Real-ESRGAN | Accepted source output, local licensed GFPGAN and RealESRGAN checkpoints plus their actual runtimes | Enable the relevant profile stage and queue CharacterSwap; identity regression, missing frames, wrong upscaled geometry, timing/audio mismatch fails. No weights currently supplied/approved for these tests. |
| Long multi-shot processing | At least three shots with different references/prompts, moving subjects, occlusion and audio; both routing-specific models | Missing/cross-shot masks, stale references, duplicate/dropped frames, incorrect local PTS or unbounded host memory fails. Use existing audit validator plus shot overrides. Animate shots over 256 frames explicitly fail; unlimited Animate chunking is incomplete. |

Remaining incomplete features such as RIFE and calibrated 3D camera reconstruction do not have executable adapters and cannot be called validated. Existing executable suites below cover RAFT, MediaPipe, VFR/VACE, learned depth/selection, identity measurement and multi-shot cache/memory behavior.

## Reproduce runtime packaging repairs

With the pinned official source checkouts already present, use the existing ComfyUI interpreter:

```bash
python scripts/repair_sam2_runtime.py --source /workspace/sam2
python scripts/repair_decord_runtime.py --source /workspace/decord --work-dir /workspace/genjutsu-deploy/decord-wheels --video /workspace/validation/source.mp4
```

These commands install only their named packages with `--no-deps`. SAM2 uses existing NVIDIA headers and copies official YAML configs omitted by its old setup. Decord downloads the small official wheel, verifies PyPI SHA256, rebuilds the matching official Python wrapper, retains its native/FFmpeg libraries and performs a real batch decode. Neither updates Torch/CUDA nor downloads model weights. Run `pip check` afterwards and the actual selection/preprocessing suites; package installation alone is not model validation.

## Prepare representative assets

Create `/workspace/validation` with:

```text
validation/
  source.mp4              # H.264/H.265 video with AAC audio; for --suite all use one shot and a 4k+1 frame count
  reference.png           # target character/product/object reference
  manifest.json           # target prompt, edit prompt, operation, seed, category
  mask/                   # one 8-bit PNG per decoded source frame; white=edit, black=preserve
    00000000.png
    00000001.png
    ...
  source_multishot.mp4    # optional separate multi-shot clip when running --suite all with single-shot I2V source.mp4
  mask_multishot/         # matching mask sequence for source_multishot.mp4
```

Use a 17–81 frame clip at 480p or below for the first smoke run. Wan I2V requires a single shot and `4k+1` frame count. Retain its original audio. `mask/` is required for VACE and multi-shot VACE runs. It is not needed by the official Wan Animate preprocessor suite, Wan Animate generation, or I2V. Example manifest:

```json
{
  "target_prompt": "person",
  "prompt": "A consistent red leather travel bag with brass hardware",
  "operation": "product_swap",
  "reference_category": "PRODUCT",
  "seed": 4312,
  "resolution_area": [832, 480]
}
```

Place weights in `/workspace/models`; create `/workspace/models/manifest.json` only when a non-default local path is needed. Supported keys are `wan_vace_model`, `wan_animate_model`, `wan_i2v_model`, `wan_repo`, `wan_preprocess_checkpoint`, `grounding_dino`, `sam2_config`, `sam2_checkpoint`, `depth_model`, `raft_checkpoint`, and `realesrgan_weights`. They may be absolute paths. Default paths are listed by `python scripts/validate_runpod.py --help` and in the table below.

Install the repository and optional model adapters into the RunPod environment. Install the CUDA-matched PyTorch/Torchvision and SAM 2 according to their upstream install instructions; do not install CPU-only PyTorch over a working CUDA build.

```bash
python -m pip install -e '.[video,api,wan,pose]'
```

## Suites and pass criteria

| Suite | Executable command | Model/checkpoint dependencies | Pass criteria |
|---|---|---|---|
| Text selection + mask tracking | `python scripts/validate_runpod.py --suite selection --assets-dir /workspace/validation --models-dir /workspace/models` | Local Transformers Grounding DINO model; official SAM 2 package, YAML config, and checkpoint. | Grounding DINO returns a detection in each shot; SAM 2 returns one non-empty mask per decoded frame. |
| Wan Animate preprocessing | `python scripts/validate_runpod.py --suite wan-preprocess --assets-dir /workspace/validation --models-dir /workspace/models` | Wan2.2 repository checkout; local `yolov10m.onnx`, `vitpose_h_wholebody.onnx`, and `sam2_hiera_large.pt` under the official checkpoint directory; upstream preprocessing Python dependencies. | Four actual upstream files are emitted with matching frame count/FPS. Pose, background and mask share processed video dimensions; face frames are 512×512; mask is grayscale RGB with white edit and black preserve values. |
| Wan 2.2 Animate replacement | `python scripts/validate_runpod.py --suite wan-animate --assets-dir /workspace/validation --models-dir /workspace/models --run-inference` | Local `Wan-AI/Wan2.2-Animate-14B-Diffusers` or compatible converted Diffusers model; Wan2.2 checkout and preprocessing checkpoints above; CUDA PyTorch; Diffusers 0.41.0. | Official controls validate, real generation completes, output frame count/FPS match source and AAC audio remains present when source has audio. Any OOM, missing control, invalid output, or timing/audio change fails. |
| Wan VACE masked editing | `python scripts/validate_runpod.py --suite wan-vace --assets-dir /workspace/validation --models-dir /workspace/models --run-inference` | Local `Wan-AI/Wan2.1-VACE-1.3B-diffusers` or compatible local model; CUDA PyTorch; Diffusers 0.41.0; one mask PNG per frame. | Real VACE generation completes; output frame count/FPS match, audio is retained, and the pipeline completes masked compositing. OOM, API mismatch, or output validation failure fails. |
| Wan image-to-video | `python scripts/validate_runpod.py --suite wan-i2v --assets-dir /workspace/validation --models-dir /workspace/models --run-inference` | Local `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` or compatible local model; CUDA PyTorch; Diffusers 0.41.0; reference image; source clip with `4k+1` frames. | Real `WanImageToVideoPipeline` inference completes and emits exact requested frame count/FPS with source audio muxed. This suite checks I2V execution; it does not assert source-video preservation. |
| Multi-shot end-to-end | `python scripts/validate_runpod.py --suite multishot-e2e --assets-dir /workspace/validation --models-dir /workspace/models --run-inference` | Same VACE dependencies and mask sequence as above; source with at least two detected cuts and audio. | Independent shot jobs finish; output has exact source frame count/FPS, source audio and per-shot cache/state records. Any missing/duplicate frame or failed shot fails. |
| Depth Anything | `python scripts/validate_runpod.py --suite depth --assets-dir /workspace/validation --models-dir /workspace/models` | Local Transformers Depth Anything model directory; PyTorch. | One finite, frame-sized depth map is returned for each source frame. |
| MediaPipe body pose | `python scripts/validate_runpod.py --suite pose --assets-dir /workspace/validation --models-dir /workspace/models` | MediaPipe and a protobuf-compatible runtime (`protobuf<5`). | Finite pose records align to frames and at least one body is detected in the subject-containing clip. |
| Torchvision RAFT | `python scripts/validate_runpod.py --suite raft --assets-dir /workspace/validation --models-dir /workspace/models` | Local `raft_large` Torchvision state dict and a compatible PyTorch/Torchvision build. | Exactly `frames-1` finite forward and backward flow fields are produced. |
| InsightFace identity measurement | `python scripts/validate_runpod.py --suite identity --assets-dir /workspace/validation --models-dir /workspace/models` | Local InsightFace `buffalo_l` ONNX files in `models/insightface/models/buffalo_l`; ONNX Runtime CUDA or CPU. | Reference and source faces are detected, and finite per-frame cosine scores are emitted. The score measures identity; it does not apply correction. |
| Real-ESRGAN upscaling | `python scripts/validate_runpod.py --suite realesrgan --assets-dir /workspace/validation --models-dir /workspace/models` | `realesrgan`, compatible BasicSR/PyTorch, local `RealESRGAN_x4plus.pth`. | Three or fewer frames are enhanced by the real model to exactly 2x output dimensions. |
| Shipped ComfyUI workflows | `python scripts/validate_comfyui.py --url http://127.0.0.1:8188` | Live ComfyUI with Genjutsu custom nodes installed and restarted. For `--queue`, provide `source.mp4`, `reference.png`, `mask/`, and for Animate workflows `src_pose.mp4`, `src_face.mp4`, `src_bg.mp4`, `src_mask.mp4` in the assets directory; model/node checkpoints must be installed in ComfyUI. | Every workflow node type and required/unknown input matches live `/object_info`. For real execution, add `--queue --assets-dir /workspace/comfy-inputs`; all six queued jobs must reach successful terminal state and save outputs. |

Each suite fails with a nonzero exit code on missing assets/dependencies, unsupported APIs, model-load errors, non-finite outputs, timing/audio failures, or generation errors. `--suite all` runs the whole list and performs actual generation; it requires `--run-inference` and every listed model. It needs `source.mp4` to be a single-shot `4k+1` I2V clip. Supply a separate multi-shot clip in `source_multishot.mp4` (plus `mask_multishot/`) so the multi-shot test remains representative. Run suites individually first so failures identify the subsystem. Save the JSON stdout, console log, generated outputs, GPU model/VRAM, and exact checkpoint revision with the deployment validation report.

These commands establish runtime and structural acceptance. Human review is still needed for task correctness, identity similarity thresholds, occlusion quality, temporal flicker, camera preservation, and visual artifacts; the suite does not convert a successful model call into a visual-quality claim.

## Audit regression acceptance on real models

The audit changes are CPU/media tested locally; model runtime, memory, revision loading, and mask reinitialization remain untested. Install `.[video,wan,test]` plus CUDA PyTorch and existing local model files. Use a fresh `audit-results/` directory for the following acceptance run. Prepare:

```text
/workspace/validation/audit/
  source_vfr.mp4       # three detectable shots, genuinely varying PTS, AAC audio with a known delayed onset
  source_ntsc.mp4      # three detectable shots, exactly 30000/1001 FPS, AAC audio
  reference.png        # representative target image
  reference_b.png      # different valid reference contents for same-path cache invalidation
  mask_vfr/           # one frame-sized PNG per decoded VFR source frame
  mask_ntsc/          # one frame-sized PNG per decoded NTSC source frame
```

Start with 17-81 frames per shot at 480p. Include target movement and partial occlusion. Supply a YAML config with `backend.type: wan_vace`, `backend.model_id` pointing to an existing Diffusers Wan 2.1 VACE directory, CUDA device/dtype, and an appropriate chunk size/overlap. No model downloads occur. Then run:

```bash
python scripts/validate_runpod_audit.py --assets-dir /workspace/validation/audit --config /workspace/config.local-vace.yaml --run-inference
```

This performs actual inference on both clips, repeats each run to require cache hits, changes one shot seed to require exactly one cache miss, and changes reference bytes at the same path to require all relevant shots to regenerate. It fails on model/API errors, OOM, missing frames, geometry changes, PTS drift beyond one container tick, endpoint drift, lost audio, audio-offset drift over 25 ms, incorrect cache reuse, or host RSS growth exceeding `--max-host-mib` (default 16384). It records measured host/GPU peaks and shot state in `audit-results/validation.json`. These limits are acceptance thresholds, not claims that a particular GPU fits the model.

To verify actual revision loading, provide a second config pointing to a second already-installed, compatible VACE checkpoint with different real checkpoint contents:

```bash
python scripts/validate_runpod_audit.py --assets-dir /workspace/validation/audit --config /workspace/config.revision-a.yaml --revision-b-config /workspace/config.revision-b.yaml --run-inference
```

Use fresh audit results before rerunning. This additionally requires revision B to perform real generation and reject every stale shot cache. Merely changing a directory name or revision label fails the checkpoint-content prerequisite. Without revision B, the report explicitly records `revision_b_validated: false`.

For bounded SAM 2 propagation, run the existing `selection` suite on a moving subject clip longer than 96 frames with `selection_frame: 40` and `selection_chunk_frames: 16` in its asset manifest. It requires local DINO/SAM2 checkpoints and official SAM 2. Inspect masks on both sides of the seed and all overlapping window boundaries; missing frames, lost/nonmatching identities, empty visible-target masks or discontinuities at boundaries fail acceptance. Fully occluded targets at a boundary may require reinitialization and are an explicit limitation, not a passing output. Repeat with a 1080p clip while monitoring RSS and GPU memory.

For disk-backed Depth Anything normalization and framewise InsightFace evaluation, rerun the `depth` and `identity` suites with their local weights on a clip longer than 96 frames. Depth must return finite maps with shared clip normalization; identity must retain frame-aligned scores and detections. Confirm host memory does not grow by retaining whole decoded RGB/depth arrays. The CPU histogram normalization tests do not validate the learned depth model.

For Wan Animate, rerun `wan-preprocess` and `wan-animate` on CFR, multi-shot subject footage within `backend.max_shot_frames`. Verify that every shot's four actual conditioning videos and generated output match its own frames and rate. VFR must fail before model inference with the explicit CFR requirement. Long Animate/I2V shots exceeding the configured frame limit must fail explicitly. Inspect quality at cuts and check the model's actual host/VRAM usage before increasing the limit.

For ComfyUI, run the live `/object_info` and queue acceptance commands above after installing the updated PyAV requirement and restarting the custom nodes. Include the VFR source in VACE and CPU compositor/temporal workflows; saved video PTS and audio offset must match the source. ComfyUI IMAGE tensors and upstream server model memory remain outside the client's disk-backed memory guarantee.
