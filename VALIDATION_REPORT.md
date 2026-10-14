# Validation report

**Run date:** 2026-10-14 (workspace date)  
**Platform:** Windows, Python 3.10, FFmpeg 8.0.1  
**GPU:** NVIDIA GeForce GTX 1650 Ti, 4 GiB visible VRAM  
**Model downloads / GPU inference:** None; intentionally not performed.

## Audit fixes and final verification (2026-10-14)

- **52 tests passed in 54.89 seconds**, including all 23 original tests. Five warnings are upstream Gradio/Starlette deprecations; no test failed or was skipped.
- Installed PyAV 16.1.0 and psutil 7.2.2 only in `.venv-local`; editable `[video,ui,api,test]` dependencies install successfully and `pip check` reports no broken requirements. System packages were not modified.
- `compileall` passed for `genjutsu`, scripts, tests, ComfyUI custom nodes and installer. `git diff --check` passed for the changes. Both RunPod validation script help commands passed.
- Live ComfyUI `/object_info` check was executed and failed with connection refused at `127.0.0.1:8188`. No workflow was queued and no live ComfyUI success is claimed.
- No model weights were downloaded, no learned model was loaded, and no GPU inference was executed.

| Audit area | Actual local evidence | Boundary |
|---|---|---|
| UI uploads | Real Gradio HTTP source/mask uploads and callbacks for single PNG, 18 PNGs, and one mask video; output is decoded and checked. Dict/FileData/pathlike/legacy upload normalization also tested. | CPU grayscale processing is a test-only backend injection; it does not validate Wan or AI quality. |
| Shot controls | VFR local duration/PTS/time ranges; local/source frame indices; masks, pose, depth, sparse tracks, forward/backward flow and camera transitions sliced; C-contiguous numeric NPY flow files are copied into shot artifacts in bounded blocks, excluding transitions across cuts; original controls remain unchanged. | Sparse record schema recognizes frame/frame_index records and numeric frame maps. Custom third-party temporal schemas require an explicit adapter. |
| VFR/rational video | Independently generated FFmpeg VFR fixture; all four export codecs preserve decoded frame count, PTS and endpoint within container tick tolerance. Exact 24000/1001, 30000/1001 and 60000/1001 round trips. | Codecs/container time bases quantize timing. Missing or nonmonotonic timestamps are rejected rather than guessed. |
| Audio synchronization | Delayed AAC track, nonzero video origin, audio start offset and decoded waveform correlation tested; lag stays within one AAC packet (1024 samples at 48 kHz). | Audio is reencoded. Sample-exact bitstream/audio preservation is not claimed. |
| Multi-shot pipeline/cache | Three-shot moving-target/partial-occlusion clip; real OpenCV grayscale processing, temporal smoothing, masked compositing and audio export; retries, cache hits, missing-file recovery, and changing one shot's prompt tested. | Backend substitution exists only in tests. Real GPU backend retries/cache output remain unverified. |
| Cache invalidation | Actual file byte hashing detects same-size/same-mtime model/reference changes; control strengths/software version changes alter keys; unresolved models and unverified ComfyUI disable reuse. | Small byte fixtures validate identity calculation, not model loading or two real checkpoint revisions. |
| Memory | Isolated processes measure Python and native RSS while decoding, lazily slicing, reading and encoding real 1920x1080 footage. Peak growth: **111.38 MiB for 16 frames**, **113.73 MiB for 96 frames** (2.35 MiB difference). Source RGB alone would be about 570 MiB at 96 frames. | Synthetic solid-color footage, CPU media stages only. This is not a long-duration GPU/learned-model memory benchmark. |
| ComfyUI CPU nodes | Actual temporal and compositor node methods export VFR with source PTS and audio. | No live server or GPU generation nodes executed. |
| SAM window/depth helpers | Contract probe checks maximum window length, bidirectional coverage and real API argument management. Shared depth histogram normalization executes on finite numeric maps. | SAM contract probe is a test double; masks and depth maps are fixtures, not model inference. |

The audit exposed x264 lookahead retaining dozens of native frames, and tiny-frame x265 heap failures on Windows. x264 uses zero-latency encoding to bound queues; H.265 uses system FFmpeg with worker pools/wavefront disabled and a smaller CTU with assembly disabled for tiny frames. Twenty consecutive standalone H.265 exports/decodes and eight repeated exports inside the regression suite passed after the final workaround. H.264/H.265/ProRes/VP9 tests pass after these fixes. Linux codec execution remains untested here.

Scratch frames are compressed lossless PNGs (NPY for floating-point controls), with manifests written last. Generation no longer stores complete videos in compressed NPZ archives or aggregates all shots in RAM. Checkpoint hashing is streaming but adds startup disk I/O. Upstream inference still requires bounded shot/chunk inputs; Animate/I2V reject shots over their configured limit, and SAM 2 reseeds overlapping bounded windows.

Real acceptance is executable with `scripts/validate_runpod_audit.py --run-inference`, including VFR/NTSC multi-shot generation, cache hits, one-shot seed invalidation, same-path reference changes, host/GPU memory measurements, and optional two-checkpoint revision validation. Assets, models, commands and failure criteria are in [RUNPOD_VALIDATION.md](RUNPOD_VALIDATION.md). Existing suites cover Animate preprocessing/inference, SAM 2 windows, depth, identity and live ComfyUI. None has been marked passed through file existence or import success.

## Executed checks

| Check | Result | Details |
|---|---|---|
| Isolated local environment | **Passed** | Created `.venv-local` without system site packages. Installed editable project and `[video,ui,api,test]`; `pip check` reported no broken requirements. This environment contains no Torch or model weights. |
| Wan API compatibility | **Passed (API surface only)** | In a separate non-deployment overlay that reused the existing CUDA Torch, imported Wan Animate and VACE from Diffusers 0.37.1 and inspected their actual call signatures. No model was loaded. |
| Diffusers version selection | **Resolved** | Diffusers 0.41.0 failed with the shared Hub 0.36.2; 0.35.2 did not expose Wan Animate; 0.37.1 exposes both Wan Animate and VACE and is now pinned with Transformers 4.x and Hub `<1`. |
| Python compilation | **Passed** | `.venv-local\Scripts\python -m compileall -q genjutsu scripts ComfyUI-Genjutsu install.py`. |
| Complete local test suite | **52 passed, 5 dependency warnings** | `.venv-local\Scripts\python -m pytest -o addopts='' -q -s`; final run: 54.89 seconds. |
| Synthetic multi-shot media integration | **Passed** | Source clips at 12 and 24 FPS with three colored shots, moving target, partial moving occlusion, mask propagation, AAC audio, and exact cut/frame/audio checks after mux. |
| Wan Animate input contract | **Passed (media format validator only)** | Aligned pose/face/background/mask videos validated for FPS, count, dimensions, grayscale RGB mask and white-generation/black-preserve values. Fixture videos are not model outputs and do not validate Wan preprocessing itself. |
| Scene/motion/mask/occlusion tests | **Passed** | Synthetic shot segmentation, Farneback flow, mask centroid propagation, both depth conventions, and scene-cut temporal reset. |
| Compositing/audio/codec tests | **Passed** | Local ring color matching, preserved pixels, H.264/H.265/ProRes/VP9, AAC mux. A 24-FPS test exposed last-frame truncation under `-shortest`; output duration handling was fixed and retested. |
| Gradio UI | **Passed real HTTP upload/callback tests** | A local server and gradio_client upload actual source/mask files, analyze shots, and process one PNG, a PNG sequence, and a mask video through the full pipeline using a test-only CPU grayscale filter. Real model generation remains untested. |
| CLI and validation scripts | **Passed smoke checks** | Local CLI config check, compileall, and help output for RunPod and ComfyUI validation scripts. |
| Live ComfyUI workflows | **Not validated** | No server answered at `127.0.0.1:8188`; `/object_info` comparison and workflow queueing were not performed. |
| Generative/learned-model inference | **Not run** | Wan, Grounding DINO, SAM2, Depth Anything, MediaPipe, RAFT, InsightFace and Real-ESRGAN weights were not downloaded or loaded. |

## Environment notes

- `.venv-local` is the supported isolated Windows development/test environment and does not inherit the machine’s conflicting global packages. The Wan API signature investigation used a separate ignored venv overlay to reuse the already-installed CUDA PyTorch without changing system packages; that overlay is for API inspection only.
- MediaPipe 0.10.14 in the preinstalled global environment conflicts with protobuf 7.36.2. The project `pose` extra constrains protobuf `<5`; model execution was not attempted.
- The visible GPU has 4 GiB. No model configuration is asserted to fit or run on it.
- No live ComfyUI server answered during this run. Workflow queue execution and outputs remain unverified.

## Remaining validation

Use the executable suites in [RUNPOD_VALIDATION.md](RUNPOD_VALIDATION.md) with representative source/reference/mask assets and local licensed checkpoint revisions. They specify commands, model dependencies, and failure criteria for Wan Animate preprocessing and generation, Wan VACE, Wan I2V, DINO/SAM2 selection, depth, pose, RAFT, InsightFace, Real-ESRGAN, multi-shot generation, and live ComfyUI workflow execution. Structural validation or a successful model call does not establish visual correctness; human review is still required for identity, task accuracy, temporal flicker, occlusion, camera preservation and artifacts.

## Changed files in this audit

```text
CAPABILITY_MATRIX.md
ComfyUI-Genjutsu/genjutsu_comfy_nodes/core.py
ComfyUI-Genjutsu/requirements.txt
README.md
RUNPOD_VALIDATION.md
THIRD_PARTY_LICENSES.md
VALIDATION_REPORT.md
genjutsu/analysis.py
genjutsu/backends/comfyui.py
genjutsu/backends/vace.py
genjutsu/backends/wan.py
genjutsu/cache.py
genjutsu/controls.py
genjutsu/frame_store.py
genjutsu/identity.py
genjutsu/pipeline.py
genjutsu/segmentation.py
genjutsu/selection.py
genjutsu/shots.py
genjutsu/temporal.py
genjutsu/ui.py
genjutsu/video.py
genjutsu/wan_preprocess.py
install.py
pyproject.toml
requirements.txt
scripts/validate_runpod.py
scripts/validate_runpod_audit.py
tests/conftest.py
tests/test_audit_regressions.py
tests/test_comfy_nodes.py
tests/test_sam_window_contract.py
tests/test_streaming_memory.py
tests/test_ui_integration.py
tests/test_video_io.py
```
