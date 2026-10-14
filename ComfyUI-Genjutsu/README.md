# ComfyUI-Genjutsu

Install this folder under `ComfyUI/custom_nodes/ComfyUI-Genjutsu` (or run `python install.py --comfyui-path PATH` from the repository root), then restart ComfyUI. The installer bundles the shared Python package and the six API-format workflows into the extension directory.

Core prerequisites are listed in `requirements.txt`; inference extras are in `optional-requirements.txt`. Install them with the same Python interpreter ComfyUI uses. Install CUDA-matched PyTorch separately. Neither script nor requirements download model weights.

Nodes use local video/image/mask paths. The shipped workflows require you to set those paths and supply masks. Wan Animate workflows require Wan-preprocessed pose, face, and mask videos. Wan VACE workflows use local Diffusers checkpoints and a per-frame edit-mask directory (white = generate, black = preserve). The Wan VACE model is the documented Wan 2.1 checkpoint family; it is not Wan 2.2 VACE.

`GenjutsuVideoOutput` copies completed video files into ComfyUI's output folder and emits a video preview reference. See the repository root README and capability matrix for limitations and validation status.
