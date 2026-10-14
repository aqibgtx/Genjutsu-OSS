# Validation report

**Run date:** 2026-10-14 (workspace date)  
**Platform:** Windows, Python 3.10, FFmpeg 8.0.1  
**GPU:** NVIDIA GeForce GTX 1650 Ti, 4 GiB visible VRAM  
**Model downloads / GPU inference:** None; intentionally not performed.

## Executed checks

| Check | Result | Details |
|---|---|---|
| Isolated local environment | **Passed** | Created `.venv-local` without system site packages. Installed editable project and `[video,ui,api,test]`; `pip check` reported no broken requirements. This environment contains no Torch or model weights. |
| Wan API compatibility | **Passed (API surface only)** | In a separate non-deployment overlay that reused the existing CUDA Torch, imported Wan Animate and VACE from Diffusers 0.37.1 and inspected their actual call signatures. No model was loaded. |
| Diffusers version selection | **Resolved** | Diffusers 0.41.0 failed with the shared Hub 0.36.2; 0.35.2 did not expose Wan Animate; 0.37.1 exposes both Wan Animate and VACE and is now pinned with Transformers 4.x and Hub `<1`. |
| Python compilation | **Passed** | `.venv-local\Scripts\python -m compileall -q genjutsu scripts ComfyUI-Genjutsu install.py`. |
| Complete local test suite | **23 passed** | `.venv-local\Scripts\python -m pytest -q`. |
| Synthetic multi-shot media integration | **Passed** | Source clips at 12 and 24 FPS with three colored shots, moving target, partial moving occlusion, mask propagation, AAC audio, and exact cut/frame/audio checks after mux. |
| Wan Animate input contract | **Passed (media format validator only)** | Aligned pose/face/background/mask videos validated for FPS, count, dimensions, grayscale RGB mask and white-generation/black-preserve values. Fixture videos are not model outputs and do not validate Wan preprocessing itself. |
| Scene/motion/mask/occlusion tests | **Passed** | Synthetic shot segmentation, Farneback flow, mask centroid propagation, both depth conventions, and scene-cut temporal reset. |
| Compositing/audio/codec tests | **Passed** | Local ring color matching, preserved pixels, H.264/H.265/ProRes/VP9, AAC mux. A 24-FPS test exposed last-frame truncation under `-shortest`; output duration handling was fixed and retested. |
| Gradio UI | **Passed launch probe** | Started from `.venv-local` at `127.0.0.1:7863`; HTTP 200 response contained `Genjutsu-OSS`. Server stopped after probe. Upload and generation interactions were not exercised. |
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
