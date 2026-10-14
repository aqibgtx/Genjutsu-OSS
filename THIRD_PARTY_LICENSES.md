# Third-party licenses and model terms

This is an initial inventory for optional dependencies and integrations. It is not a substitute for checking the exact package version, transitive dependencies, bundled FFmpeg build, checkpoint files, and intended use before redistribution. Source code and model weights can carry different terms.

| Component | Upstream | Code license / status | Model license / restrictions |
|---|---|---|---|
| Genjutsu-OSS original code | This repository | Apache-2.0 | Not applicable |
| NumPy | https://github.com/numpy/numpy | BSD-3-Clause | Not applicable |
| Pillow | https://github.com/python-pillow/Pillow | HPND | Not applicable |
| PyYAML | https://github.com/yaml/pyyaml | MIT | Not applicable |
| Requests | https://github.com/psf/requests | Apache-2.0 | Not applicable |
| OpenCV | https://github.com/opencv/opencv | Apache-2.0 (current upstream) | Not applicable |
| PyAV | [PyAV 16.1.0 license](https://github.com/PyAV-Org/PyAV/blob/v16.1.0/LICENSE.txt) | BSD-3-Clause for bindings; bundled FFmpeg/codec libraries have separate notices and build-dependent LGPL/GPL terms | Not applicable |
| psutil (test dependency) | https://github.com/giampaolo/psutil | BSD-3-Clause | Not applicable |
| Gradio | https://github.com/gradio-app/gradio | Apache-2.0 | Not applicable |
| FastAPI | https://github.com/fastapi/fastapi | MIT | Not applicable |
| Uvicorn | https://github.com/encode/uvicorn | BSD-3-Clause | Not applicable |
| Pytest / pytest-cov | https://github.com/pytest-dev/pytest | MIT (both projects) | Not applicable |
| PyTorch | https://github.com/pytorch/pytorch | BSD-style; inspect bundled third-party notices | Not applicable |
| Torchvision | https://github.com/pytorch/vision | BSD-3-Clause | Not applicable |
| Diffusers | https://github.com/huggingface/diffusers | Apache-2.0 | Check each checkpoint's own terms |
| Transformers | https://github.com/huggingface/transformers | Apache-2.0 | Check each checkpoint's own terms |
| Accelerate | https://github.com/huggingface/accelerate | Apache-2.0 | Not applicable |
| Safetensors | https://github.com/huggingface/safetensors | Apache-2.0 | Not applicable |
| Protobuf | https://github.com/protocolbuffers/protobuf | BSD-3-Clause | Not applicable |
| ComfyUI | https://github.com/Comfy-Org/ComfyUI | GPL-3.0; used as an external host | Check model terms separately |
| SAM 2 | https://github.com/facebookresearch/sam2 | Apache-2.0 for code/checkpoints; retain notices | Upstream states released checkpoints are Apache-2.0 |
| Wan 2.2 | https://github.com/Wan-Video/Wan2.2 | Apache-2.0 per upstream repository | Upstream states model files Apache-2.0; verify exact model card |
| Wan 2.1 VACE | https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers | This adapter is Apache-2.0 | Check exact checkpoint card; do not infer from Wan 2.2 terms |
| MediaPipe | https://github.com/google-ai-edge/mediapipe | Apache-2.0 | Model assets can carry separate notices |
| InsightFace | https://github.com/deepinsight/insightface | MIT code | Downloaded buffalo_l weights are for non-commercial research. User approved non-commercial testing only. This deployment is not qualified for commercial use of those weights. |
| Real-ESRGAN | https://github.com/xinntao/Real-ESRGAN | BSD-3-Clause | Check selected pretrained weights separately |
| BasicSR | https://github.com/XPixelGroup/BasicSR | Apache-2.0 | Check architectures/weights separately |
| FFmpeg | https://ffmpeg.org/ | LGPL or GPL depending on build flags | Not applicable |

The workspace FFmpeg reports a GPL-enabled build. Redistributing that executable invokes the corresponding obligations; this repository does not bundle it. Check its `ffmpeg -L` license output and `-buildconf` flags.

Before publication, regenerate the inventory from exact locked dependency versions and verify every checkpoint, codec, and optional extension. No weights are bundled or automatically downloaded during installation.
