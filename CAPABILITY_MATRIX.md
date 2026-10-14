# Capability matrix

Status reflects this Windows validation run only. “Locally tested” describes actual CPU/media execution; it does not imply generative quality. No model weights were downloaded and no GPU inference was run.

| Subsystem | Status | Evidence and boundary |
|---|---|---|
| Video ingest, metadata, frame timing | **Locally tested** | Actual decoded PTS/durations, rational FPS/time base, VFR fixtures and 24000/1001, 30000/1001, 60000/1001 round trips. Missing/nonmonotonic PTS fail explicitly. |
| Encoding and audio | **Locally tested** | H.264/H.265/ProRes/VP9 retain VFR frame count/PTS/endpoint. AAC delayed onset and nonzero video origin tested, including decoded waveform lag within one AAC packet. Source audio is reencoded. |
| CPU/media memory | **Locally measured** | Disk-backed lossless PNG/NPY sequences, lazy slices, streaming decode/encode, disk shot aggregation, smoothing/compositing and cache reads. Isolated RSS growth: 111.38 MiB at 16 frames and 113.73 MiB at 96 frames, 1920x1080. No GPU memory claim. Model pipelines have separate limits. |
| Scene/shot detection | **Locally tested** | HSV histogram cuts found the two transitions in synthetic three-shot clips. This is a visual histogram detector, not semantic shot understanding. |
| Camera motion | **Locally tested** | OpenCV RANSAC partial-affine translation, rotation and zoom. It does not recover calibrated 3D camera paths. |
| Farneback optical flow | **Locally tested** | Forward/backward fields on synthetic moving-object clips. |
| RAFT flow | **Implemented, GPU/runtime untested** | Torchvision RAFT adapter requires a local compatible state dict and Torch/Torchvision pair. RunPod suite: `raft`. |
| Prompt-based object detection | **Implemented, GPU/runtime untested** | Transformers Grounding DINO uses a local model directory and configured box/text thresholds. RunPod suite: `selection`. |
| SAM 2 segmentation and propagation | **Implemented, GPU/runtime untested** | Official predictor now uses bounded CPU-offloaded windows, box seed, mask reseeding on one-frame overlaps, and bidirectional propagation. Local test verifies window/API management only using a test double. Real masks, identity continuity and memory require the `selection` RunPod suite. |
| Automatic target selection | **Implemented, GPU/runtime untested** | UI target text -> local Grounding DINO detection -> SAM 2 initialization and shot-local propagation. Requires local DINO/SAM2 models and config. |
| Manual mask inputs | **Locally tested, including HTTP UI uploads** | Single PNG, numerically ordered PNG sequence, and mask video; count/dimension and mixed-input errors checked. Mask video alignment is by decoded frame index. |
| Classical mask tracking | **Locally tested** | Bidirectional Farneback mask warp; can drift through long occlusions or scene changes. Automatic SAM2 is run independently per detected shot. |
| Wan 2.2 Animate preprocessing | **Implemented, CPU wrapper/format tests passed; upstream runtime untested** | Calls Wan2.2 `preprocess_data.py` with local detector, whole-body pose and SAM2 checkpoints; validates `src_pose.mp4`, `src_face.mp4`, `src_bg.mp4`, `src_mask.mp4` alignment, grayscale mask channels, and white-edit/black-preserve convention. No Wan preprocess model was available to execute it. |
| Human pose extraction | **Implemented, runtime untested here** | MediaPipe Holistic landmark adapter. Installed global MediaPipe/protobuf pair remains incompatible; optional `pose` extra pins protobuf `<5`. RunPod suite: `pose`. |
| Depth Anything | **Implemented, GPU/model untested** | Framewise local Transformers inference can write disk-backed maps and use shared bounded-histogram percentile normalization. Histogram normalization is CPU tested; learned inference/temporal depth quality remain unverified. RunPod: `depth`. |
| Reference package | **Locally tested** | Categorized and weighted reference assets. Classical color histogram/ORB metadata is descriptive, not a semantic identity embedding. |
| Cross-shot reference consistency | **Implemented, structural test through shot/cache paths** | Global references and weights are shared across shots; reference SHA-256/category/weight are recorded and included in per-shot cache keys. Per-shot overrides can intentionally replace them. This does not guarantee generated identity consistency. |
| ControlPackage and prompt compiler | **Locally tested** | JSON round trip and deterministic operation-aware prompt construction. |
| Wan 2.2 Animate Diffusers adapter | **Implemented, inference untested** | Preserves real WanAnimatePipeline API and revision argument. Explicit VFR rejection and shot frame limit tested without loading a model. Official preprocessing and real generation still require `wan-preprocess` / `wan-animate`; default maximum is 256 frames per shot. |
| Wan image-to-video adapter | **Implemented, inference untested** | Real Diffusers `WanImageToVideoPipeline`; I2V alone does not preserve or edit source-video regions. RunPod suite: `wan-i2v`. |
| Wan VACE video editing | **Implemented, API signature checked, inference untested** | Real `WanVACEPipeline`; source video, white generation mask, optional reference images, 4k+1 frame padding, scene-local overlapping chunks, and seed. Adapter targets Wan 2.1 VACE, not Wan 2.2 VACE. RunPod suite: `wan-vace`. |
| Automatic operation routing | **Locally tested** | `auto` routes source-video editing to Wan VACE by default. Character/motion use Wan Animate only when explicitly enabled and its local preprocessing paths are configured. Object/product/outfit/environment/full recast use VACE. |
| Per-shot orchestration | **Locally tested with real CPU media processing; Wan inference untested** | Independent prompts/settings/references, local PTS/duration/indices, original source range, sliced masks/pose/depth/tracks/flow and camera transitions. Retry, resume, missing-cache recovery and one-shot override invalidation tested through a test-only OpenCV grayscale backend. RunPod: `validate_runpod_audit.py` and `multishot-e2e`. |
| Shot retries and cache | **Locally tested for orchestration/content identity; GPU reuse untested** | Lossless per-shot stores; content hashes for references, masks, actual model/config/tokenizer files, revisions, effective controls and runtime/source versions. Same-size/same-mtime changes invalidate. Unresolved models and unverified ComfyUI server fingerprints disable reuse. Actual VACE chunk/revision reuse requires RunPod audit inference. |
| Reference/identity measurement | **Implemented, GPU/model untested** | InsightFace local ONNX scoring against a reference; reports per-frame cosine scores and detection rate. It does not perform correction or certify identity thresholds. RunPod suite: `identity`. |
| Temporal consistency | **Locally tested on synthetic flow** | Flow-warp smoothing rejects forward/backward inconsistent pixels and resets at cuts. This is a classical stabilizer, not learned temporal repair. |
| Occlusion handling | **Partly implemented, learned depth untested** | Depth-order suppression can use Depth Anything plus an explicit foreground mask. Automatic layered scene graph, hand/object interaction, and recovered hidden surfaces are incomplete. |
| Compositing and color matching | **Locally tested** | Soft masks, preserve masks, local ring-based color moments, and original/generated blend. Perceptual blend quality was not assessed. |
| Image upscaling | **Implemented, GPU/model untested** | Lanczos utility plus optional local Real-ESRGAN adapter. Video restoration, face restoration and interpolation are absent. RunPod suite: `realesrgan`. |
| Identity lock/correction | **Incomplete** | Identity is measured only. There is no iterative correction loop, threshold calibration, or guaranteed identity preservation. |
| ComfyUI custom nodes | **Locally CPU/media tested; GPU nodes untested** | Nodes register; actual compositor and temporal nodes preserve VFR timestamps/audio in local execution. The IMAGE tensor loader intentionally materializes a tensor. Live server/GPU node execution remains unverified. |
| Shipped ComfyUI workflows | **Locally schema-tested; live server unavailable** | Six API-format workflows match repository node schemas. `scripts/validate_comfyui.py` checks live `/object_info` inputs and can queue actual jobs. No ComfyUI server was available during this run. |
| ComfyUI HTTP backend | **Implemented, live server untested** | Real queue/history/output HTTP client. No server response or generated workflow output was observed. |
| High-level Gradio UI | **Locally HTTP interaction-tested** | Real file-upload endpoint, source analysis, one PNG, multiple PNGs and mask video exercise callbacks and produce valid CPU-processed output. No model-specific files are needed by the UI. End-to-end Wan execution remains untested. |
| CLI / API | **Locally smoke-tested** | CLI config/runtime check and FastAPI app import succeeded. No API transform request was sent. |
| `.genjutsu` project archives | **Locally tested** | Archive create/inspect/restore, checksums and asset relocation. Model weights are not bundled. |

## Incomplete capabilities and blockers

- No Wan, DINO, SAM2, depth, pose, RAFT, InsightFace or Real-ESRGAN model weights are in this repository; none were downloaded.
- No real GPU inference ran. GPU memory, speed, model compatibility at checkpoint revision, and output quality remain unverified.
- No live ComfyUI server was available at `127.0.0.1:8188`; the shipped workflows are not yet proven to finish on ComfyUI.
- Text selection depends on local Grounding DINO/SAM2 checkpoints and may select the wrong instance when a prompt has several similar objects. User-provided per-shot selection prompts and indexes can steer the choice.
- Depth-based occlusion requires a foreground mask. Full scene graphs, automatic hand/person overlap reasoning, identity correction, learned temporal repair, video interpolation and face restoration remain incomplete.
- GPU validation commands and pass/fail criteria are documented in [RUNPOD_VALIDATION.md](RUNPOD_VALIDATION.md). Model/API or file existence alone does not count as a pass.

## Audit boundaries

- Full local suite: **52 passed**, five upstream Gradio deprecation warnings, in 54.89 seconds. No weights were downloaded and no GPU inference ran.
- VFR timestamps are preserved by the media pipeline; VACE receives frame order rather than a learned variable-interval control. Animate requires explicit CFR source preparation.
- Scratch/cache disk use depends on footage and compression. The eager `read_frames` compatibility option and ComfyUI IMAGE tensors are outside the streaming memory guarantee. Animate/I2V and third-party inference retain shot/chunk arrays within documented limits.
- SAM mask reseeding at windows changes temporal memory context; fully lost targets at a boundary fail explicitly. Actual tracking quality, checkpoint revision loading, VACE cache reuse and GPU memory must pass the new audit acceptance commands in [RUNPOD_VALIDATION.md](RUNPOD_VALIDATION.md).
