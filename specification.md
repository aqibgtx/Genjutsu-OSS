# Genjutsu-OSS
 
## Open-Source ComfyUI Recreation of a Genjutsu-Style Video Transformation Pipeline
 
### Project Overview
 
Genjutsu-OSS is an open-source, modular video transformation system built around ComfyUI.
 
The goal is to reproduce the *observable capabilities* of modern Genjutsu-style video editing systems using publicly available open-source models and software.
 
The system should support:
 
1. Motion Transfer
2. Character Replacement
3. Object Replacement
4. Product Replacement
5. Outfit Replacement
6. Environment / Background Recasting
7. Style Recasting
8. Reference-driven video generation
9. Selective region editing
10. Temporal consistency
11. Identity preservation
12. Camera-motion preservation
13. Original-background preservation
14. Video compositing
15. Upscaling / restoration
16. Audio preservation
17. Modular model replacement
 
This project is NOT intended to reproduce proprietary Higgsfield source code or proprietary models.
 
Instead, it recreates the functionality using an open architecture.
 
The design principle is:
 
SOURCE VIDEO
→ UNDERSTAND VIDEO
→ EXTRACT CONTROLS
→ BUILD CONTROL PACKAGE
→ GENERATE / RECONSTRUCT
→ TEMPORAL REPAIR
→ COMPOSITE
→ RESTORE / UPSCALE
→ FINAL VIDEO
 
---
 
# 1. Core Architecture
 
The system should consist of five major layers.
 
## Layer 1 — Video Understanding
 
Analyze the original video and extract:
 
* frames
* FPS
* resolution
* scene cuts
* subjects
* objects
* segmentation masks
* tracking information
* human pose
* optical flow
* depth
* camera movement
* foreground/background relationships
* occlusion relationships
 
The original video should be converted into a structured representation rather than being treated as a simple image sequence.
 
---
 
# 2. Layer 2 — Reference Understanding
 
Reference images should be analyzed and converted into conditioning information.
 
Supported references should include:
 
* characters
* faces
* products
* objects
* clothing
* environments
* locations
* styles
* textures
* logos
* props
 
The reference system should support multiple reference images.
 
The system should determine whether each reference represents:
 
* identity
* appearance
* object geometry
* product design
* style
* environment
* texture
* clothing
 
References should be stored in a reusable Reference Package.
 
Example:
 
ReferencePackage:
 
```
identity_embedding
appearance_features
object_features
style_features
image_features
masks
metadata
```
 
---
 
# 3. Layer 3 — Control Package
 
All extracted information should be combined into a unified control structure.
 
Example:
 
ControlPackage:
 
```
source_video
frames
fps
resolution
 
subject_masks
object_masks
preservation_masks
 
tracking_data
pose_sequence
optical_flow
depth_sequence
camera_motion
 
references
 
operation
prompt
negative_prompt
 
temporal_settings
identity_strength
motion_strength
reference_strength
preservation_strength
```
 
This ControlPackage should be model-independent.
 
That means the rest of the system should not be permanently tied to one video-generation model.
 
---
 
# 4. Layer 4 — Generative Engine
 
The first-generation backend should prioritize open-source Wan 2.2 models.
 
Primary candidates:
 
* Wan2.2
* Wan2.2 I2V
* Wan2.2 TI2V
* Wan2.2 Animate
* compatible future open video models
 
The generative engine should be implemented behind an abstraction layer.
 
Example:
 
VideoGenerator:
 
```
generate()
generate_i2v()
generate_t2v()
generate_character()
generate_region()
generate_recast()
```
 
The goal is to make it possible to replace Wan2.2 later without rewriting the rest of the project.
 
---
 
# 5. Layer 5 — Post Processing
 
Generated video should not automatically become the final video.
 
The system should perform:
 
* temporal consistency correction
* identity stabilization
* mask cleanup
* boundary repair
* occlusion repair
* color matching
* lighting matching
* background preservation
* optical-flow-based stabilization
* frame interpolation when necessary
* detail restoration
* upscaling
* audio restoration/preservation
* final encoding
 
The post-processing layer should be modular.
 
---
 
# 6. ComfyUI Integration
 
The entire system should be exposed through ComfyUI custom nodes.
 
Create a custom node package:
 
```
ComfyUI-Genjutsu
```
 
Suggested structure:
 
```
ComfyUI-Genjutsu/
 
    __init__.py
 
    nodes/
        video_loader.py
        video_analysis.py
        segmentation.py
        tracking.py
        pose.py
        depth.py
        optical_flow.py
        camera.py
        references.py
        control_package.py
        prompt_compiler.py
        wan_generator.py
        temporal.py
        compositor.py
        identity.py
        upscale.py
        video_output.py
 
    genjutsu/
        video.py
        segmentation.py
        tracking.py
        motion.py
        depth.py
        camera.py
        references.py
        controls.py
        temporal.py
        compositing.py
        restoration.py
 
    workflows/
        motion_transfer.json
        object_swap.json
        character_swap.json
        product_swap.json
        recast.json
        full_pipeline.json
 
    examples/
 
    tests/
 
    requirements.txt
 
    install.py
 
    README.md
```
 
---
 
# 7. Required Custom Nodes
 
## Genjutsu Video Loader
 
Input:
 
```
video
```
 
Output:
 
```
frames
fps
width
height
frame_count
duration
```
 
Must preserve original audio metadata.
 
---
 
# 8. Genjutsu Scene Analyzer
 
Input:
 
```
video
```
 
Output:
 
```
subjects
objects
scene information
scene cuts
frame metadata
```
 
The node should provide automatic scene decomposition.
 
---
 
# 9. Genjutsu Segmenter
 
Use SAM 2 or another open segmentation model.
 
Input:
 
```
video
optional prompt
```
 
Output:
 
```
mask sequence
```
 
Must support:
 
* person
* face
* hands
* clothing
* object
* product
* foreground
* background
 
The mask must remain temporally stable.
 
---
 
# 10. Genjutsu Tracker
 
Input:
 
```
video
initial mask
```
 
Output:
 
```
tracked mask sequence
```
 
The system should track the selected object/person through the entire video.
 
It should handle:
 
* translation
* rotation
* scaling
* deformation
* partial occlusion
* temporary disappearance
* reappearance
 
---
 
# 11. Genjutsu Pose Extractor
 
Use DWPose/OpenPose or equivalent.
 
Output:
 
```
pose sequence
```
 
For humans, capture:
 
* body
* arms
* hands
* legs
* head
* face landmarks where available
 
The pose sequence should be usable as a temporal control signal.
 
---
 
# 12. Genjutsu Depth Extractor
 
Use Video Depth Anything or equivalent.
 
Output:
 
```
depth sequence
```
 
Depth should remain temporally consistent.
 
Depth should be used for:
 
* foreground/background separation
* occlusion
* compositing
* object placement
* camera understanding
* scene reconstruction
 
---
 
# 13. Genjutsu Optical Flow
 
Use RAFT or an equivalent optical-flow model.
 
Output:
 
```
forward_flow
backward_flow
```
 
Optical flow should be used for:
 
* mask propagation
* frame alignment
* temporal stabilization
* generated-frame comparison
* compositing
* temporal repair
 
---
 
# 14. Genjutsu Camera Analyzer
 
Estimate:
 
* camera translation
* camera rotation
* zoom
* perspective changes
* camera shake
* camera trajectory
 
The system should attempt to preserve the original camera language when performing a scene recast.
 
---
 
# 15. Genjutsu Reference Manager
 
Allow the user to upload multiple references.
 
Each reference should be categorized.
 
Example:
 
```
CHARACTER
PRODUCT
OBJECT
OUTFIT
LOCATION
STYLE
PROP
```
 
References should have adjustable weights.
 
Example:
 
```
identity_strength = 0.85
appearance_strength = 0.75
style_strength = 0.50
object_strength = 1.00
```
 
---
 
# 16. Genjutsu Prompt Compiler
 
Users should not need to construct extremely complicated prompts manually.
 
The node should accept high-level parameters.
 
Example:
 
```
operation:
    OBJECT_SWAP
 
target:
    WATCH
 
reference:
    PRODUCT_REFERENCE
 
preserve:
    PERSON
    HAND
    BACKGROUND
    CAMERA
    LIGHTING
 
motion:
    ORIGINAL
```
 
The compiler converts this into appropriate model conditioning.
 
Example conceptual prompt:
 
```
Preserve the original subject identity, body movement,
facial expression, camera trajectory, composition,
lighting and background.
 
Replace only the selected object with the supplied reference.
 
Maintain correct geometry, perspective, lighting,
reflections, shadows and temporal consistency.
```
 
---
 
# 17. Preservation System
 
One of the most important components.
 
Generate separate masks:
 
## Edit Mask
 
Defines what is allowed to change.
 
## Preserve Mask
 
Defines what should remain unchanged.
 
## Soft Boundary Mask
 
Defines a transition area between generated and original content.
 
Example:
 
```
EDIT_MASK
    ↓
dilation
    ↓
feather
    ↓
SOFT_MASK
```
 
The system should never unnecessarily regenerate regions that the user wants preserved.
 
---
 
# 18. Object Swap Pipeline
 
Workflow:
 
```
SOURCE VIDEO
      ↓
OBJECT DETECTION
      ↓
SAM2 SEGMENTATION
      ↓
OBJECT TRACKING
      ↓
DEPTH
      ↓
REFERENCE OBJECT
      ↓
CONTROL PACKAGE
      ↓
WAN GENERATION
      ↓
GENERATED OBJECT
      ↓
OCCLUSION REPAIR
      ↓
MASK REFINEMENT
      ↓
COMPOSITING
      ↓
TEMPORAL REPAIR
      ↓
UPSCALE
      ↓
FINAL VIDEO
```
 
The surrounding image should be preserved wherever possible.
 
---
 
# 19. Character Replacement Pipeline
 
Workflow:
 
```
SOURCE VIDEO
      ↓
PERSON SEGMENTATION
      ↓
POSE EXTRACTION
      ↓
FACE / IDENTITY ANALYSIS
      ↓
DEPTH
      ↓
TEMPORAL TRACKING
      ↓
CHARACTER REFERENCE
      ↓
WAN2.2 ANIMATE
      ↓
TEMPORAL STABILIZATION
      ↓
IDENTITY CORRECTION
      ↓
BACKGROUND COMPOSITING
      ↓
FINAL VIDEO
```
 
The goal is to preserve:
 
* movement
* timing
* pose
* expression
* camera
* lighting
* environment
 
while changing the character.
 
---
 
# 20. Motion Transfer Pipeline
 
Input:
 
```
motion source video
reference character
```
 
Process:
 
```
SOURCE VIDEO
      ↓
POSE
      ↓
OPTICAL FLOW
      ↓
SUBJECT TRACKING
      ↓
CAMERA ANALYSIS
      ↓
CHARACTER REFERENCE
      ↓
WAN2.2 ANIMATE
      ↓
TEMPORAL REPAIR
      ↓
OUTPUT
```
 
The original motion should be treated as a primary constraint.
 
---
 
# 21. Environment Recast
 
Example:
 
```
ORIGINAL:
    person walking
    city street
 
TARGET:
    person walking
    futuristic city
```
 
The system should preserve:
 
* subject motion
* timing
* pose
* camera movement
 
while reconstructing:
 
* environment
* architecture
* lighting
* atmosphere
* background
 
This should use the full generative video pipeline rather than simple compositing.
 
---
 
# 22. Full Recast Mode
 
Allow:
 
```
CHARACTER = REFERENCE
ENVIRONMENT = REFERENCE
OBJECTS = REFERENCE
STYLE = REFERENCE
MOTION = ORIGINAL
CAMERA = ORIGINAL
```
 
This produces an entirely new visual interpretation while retaining the motion/camera structure of the source.
 
---
 
# 23. Temporal Consistency Engine
 
Generated frames must be analyzed across time.
 
Required tools:
 
* optical flow
* frame warping
* temporal masks
* feature comparison
* identity similarity
* flicker detection
* motion consistency
 
Possible algorithm:
 
```
frame[t]
   ↓
generated[t]
 
frame[t+1]
   ↓
generated[t+1]
 
optical flow
   ↓
warp generated[t]
   ↓
compare generated[t+1]
   ↓
detect inconsistency
   ↓
repair
```
 
The system should expose:
 
```
temporal_strength
identity_strength
smoothing_strength
repair_strength
```
 
---
 
# 24. Identity Lock
 
For character workflows, identity should be measured throughout the video.
 
Use face/identity embeddings when applicable.
 
For every frame:
 
```
reference_identity
       ↓
generated_identity
       ↓
similarity
       ↓
identity_score
```
 
If identity falls below a threshold, trigger correction.
 
This should be optional.
 
---
 
# 25. Occlusion Engine
 
Objects and characters must correctly interact with the environment.
 
Example:
 
```
HAND
  ↓
WATCH
```
 
The watch should sometimes appear:
 
```
behind hand
```
 
and sometimes:
 
```
in front of hand
```
 
depending on depth.
 
Use:
 
* segmentation
* depth
* temporal tracking
* layer masks
 
to construct an approximate scene graph.
 
---
 
# 26. Compositor
 
The compositor should support:
 
```
original frame
generated frame
edit mask
preserve mask
soft mask
depth
foreground
background
```
 
Conceptually:
 
```
output =
    original * preserve
    +
    generated * edit
    +
    blended_boundary
```
 
The compositor must operate temporally, not just frame-by-frame.
 
---
 
# 27. Color / Lighting Matching
 
Generated content must match the original video.
 
Perform:
 
* exposure matching
* white-balance matching
* contrast matching
* color-temperature matching
* local luminance matching
* shadow matching
* highlight matching
 
This should be performed before final compositing.
 
---
 
# 28. Restoration
 
Optional restoration pipeline:
 
```
temporal repair
      ↓
face/detail restoration
      ↓
Real-ESRGAN
      ↓
final sharpening
      ↓
output
```
 
The system should avoid over-sharpening AI-generated frames.
 
---
 
# 29. Video Encoding
 
Use FFmpeg.
 
Requirements:
 
* preserve FPS
* preserve resolution
* preserve audio
* support H.264
* support H.265
* support ProRes where available
* support image sequences
* support MP4
* support MOV
* support WebM
 
---
 
# 30. Required Workflows
 
The repository should ship with these ComfyUI workflows.
 
## Workflow 1
 
```
MotionTransfer.json
```
 
Input:
 
```
source video
character reference
```
 
Output:
 
```
character performing original motion
```
 
---
 
## Workflow 2
 
```
ObjectSwap.json
```
 
Input:
 
```
source video
object reference
```
 
Output:
 
```
only selected object replaced
```
 
---
 
## Workflow 3
 
```
CharacterSwap.json
```
 
Input:
 
```
source video
character reference
```
 
Output:
 
```
original performance with replacement character
```
 
---
 
## Workflow 4
 
```
ProductSwap.json
```
 
Input:
 
```
source advertisement
product reference
```
 
Output:
 
```
same commercial with replacement product
```
 
---
 
## Workflow 5
 
```
EnvironmentRecast.json
```
 
Input:
 
```
source video
environment reference/prompt
```
 
Output:
 
```
original motion/camera with new environment
```
 
---
 
## Workflow 6
 
```
FullGenjutsu.json
```
 
Input:
 
```
source video
multiple references
prompt
```
 
Output:
 
```
complete transformed video
```
 
---
 
# 31. Model Abstraction
 
Do not hard-code Wan2.2 into the entire application.
 
Create:
 
```
VideoBackend
```
 
with implementations:
 
```
Wan22Backend
WanAnimateBackend
FutureBackend
```
 
The control pipeline should remain independent.
 
This allows future integration of:
 
* newer Wan models
* HunyuanVideo
* other open video diffusion models
* specialized video editors
* local proprietary models
* distilled models
 
---
 
# 32. Configuration
 
Provide a configuration file:
 
```
config.yaml
```
 
Example:
 
```
backend: wan22
 
segmentation:
    model: sam2
 
depth:
    model: video_depth_anything
 
pose:
    model: dwpose
 
optical_flow:
    model: raft
 
identity:
    enabled: true
 
temporal:
    enabled: true
    strength: 0.75
 
compositing:
    enabled: true
 
upscale:
    enabled: true
    model: realesrgan
```
 
All major components should be configurable.
 
---
 
# 33. Performance Requirements
 
The system should support:
 
* low VRAM mode
* CPU offloading
* model caching
* tiled processing
* frame batching
* temporal chunking
* latent caching
* intermediate-result caching
* resume after failure
 
Long videos should be processed in chunks.
 
Example:
 
```
30-second video
 
↓
 
chunk 1: frames 0–80
chunk 2: frames 72–152
chunk 3: frames 144–224
```
 
with overlapping frames used for temporal continuity.
 
---
 
# 34. Intermediate Cache
 
Every expensive stage should be cacheable.
 
Example:
 
```
/cache/project/
 
    frames/
    masks/
    tracking/
    pose/
    depth/
    flow/
    references/
    generated/
    temporal/
    composite/
```
 
If generation fails, the entire pipeline should not need to restart.
 
---
 
# 35. Project File
 
Create a project format:
 
```
.genjutsu
```
 
containing:
 
```
source video
references
masks
controls
prompt
model settings
generation parameters
processing state
```
 
The project should be reproducible.
 
---
 
# 36. User Interface
 
ComfyUI remains the primary interface.
 
However, provide a high-level workflow that exposes only:
 
```
SOURCE VIDEO
 
OPERATION
 
    Motion Transfer
    Character Swap
    Object Swap
    Product Swap
    Environment Recast
    Full Recast
 
REFERENCES
 
PROMPT
 
PRESERVATION
 
    Motion
    Camera
    Background
    Identity
    Lighting
 
GENERATION
 
    Quality
    Strength
    Seed
 
OUTPUT
 
    Resolution
    FPS
    Codec
```
 
Advanced users can expand the underlying graph.
 
---
 
# 37. Safety / Attribution
 
The project must remain open-source and must not claim to be the proprietary implementation of Higgsfield Genjutsu.
 
Use language such as:
 
```
"Genjutsu-style open-source video transformation pipeline"
```
 
rather than:
 
```
"Higgsfield Genjutsu source-code clone"
```
 
The project should clearly distinguish:
 
```
OBSERVED BEHAVIOR
```
 
from:
 
```
IMPLEMENTATION INFERENCE
```
 
and:
 
```
OPEN-SOURCE IMPLEMENTATION
```
 
---
 
# 38. Licensing
 
Before redistribution, verify the license of every dependency and model.
 
Do not assume that "open source" means unrestricted commercial redistribution.
 
Create:
 
```
THIRD_PARTY_LICENSES.md
```
 
containing:
 
* dependency
* repository
* license
* model license
* commercial-use restrictions
* attribution requirements
 
Model weights and source code must be treated separately.
 
---
 
# 39. Testing
 
Create automated tests for:
 
## Segmentation
 
```
mask accuracy
temporal stability
```
 
## Tracking
 
```
object persistence
re-identification
```
 
## Depth
 
```
temporal consistency
```
 
## Motion
 
```
optical-flow accuracy
```
 
## Identity
 
```
face similarity
```
 
## Temporal
 
```
flicker score
motion consistency
```
 
## Compositing
 
```
mask leakage
boundary artifacts
```
 
## End-to-end
 
```
source → transformation → output
```
 
---
 
# 40. Quality Metrics
 
Build an evaluation system.
 
Measure:
 
```
identity_similarity
temporal_consistency
motion_similarity
camera_similarity
background_similarity
object_similarity
prompt_alignment
perceptual_quality
```
 
Example:
 
```
GenjutsuScore
 
    Identity       0.92
    Motion         0.95
    Camera         0.91
    Background     0.97
    Object         0.89
    Temporal       0.94
```
 
This allows different model backends to be objectively compared.
 
---
 
# 41. Development Phases
 
## Phase 1 — Foundation
 
Implement:
 
* ComfyUI integration
* video loading
* frame extraction
* FFmpeg
* Wan2.2
* basic output
 
Goal:
 
```
working video generation
```
 
---
 
## Phase 2 — Video Understanding
 
Add:
 
* SAM2
* DWPose
* Video Depth Anything
* RAFT
* tracking
 
Goal:
 
```
structured ControlPackage
```
 
---
 
## Phase 3 — Motion Transfer
 
Implement:
 
* reference character
* pose conditioning
* Wan2.2 Animate
* temporal processing
 
Goal:
 
```
convincing character motion transfer
```
 
---
 
## Phase 4 — Object Swap
 
Implement:
 
* object segmentation
* tracking
* reference object
* region generation
* preservation masks
* compositing
 
Goal:
 
```
change one object without unnecessarily changing the scene
```
 
---
 
## Phase 5 — Recast
 
Implement:
 
* environment reconstruction
* camera preservation
* subject preservation
* reference environment
* temporal generation
 
Goal:
 
```
same performance, different world
```
 
---
 
## Phase 6 — Temporal Engine
 
Implement:
 
* optical-flow stabilization
* identity correction
* flicker detection
* temporal repair
* occlusion correction
 
Goal:
 
```
production-quality temporal consistency
```
 
---
 
## Phase 7 — Optimization
 
Implement:
 
* caching
* chunking
* VRAM optimization
* model offloading
* batch processing
* resume/recovery
 
Goal:
 
```
usable local application
```
 
---
 
# 42. Final Target Architecture
 
The completed project should look like:
 
```
             USER
               │
               ▼
          COMFYUI
               │
               ▼
      GENJUTSU WORKFLOW
               │
    ┌──────────┴──────────┐
    │                     │
```
 
SOURCE VIDEO           REFERENCES
│                     │
▼                     ▼
┌─────────────────────────────────┐
│       VIDEO UNDERSTANDING       │
│                                 │
│ SAM2                            │
│ DWPose                          │
│ Video Depth Anything            │
│ RAFT                            │
│ Tracking                        │
│ Camera Analysis                 │
└───────────────┬─────────────────┘
│
▼
CONTROL PACKAGE
│
▼
PROMPT COMPILER
│
▼
┌──────────────────┐
│  VIDEO BACKEND   │
│                  │
│     WAN 2.2      │
│     WAN ANIMATE  │
└────────┬─────────┘
│
▼
GENERATED VIDEO
│
▼
TEMPORAL ENGINE
│
┌────────┼─────────┐
▼        ▼         ▼
Identity  Motion   Flicker
Repair    Repair   Repair
│        │         │
└────────┼─────────┘
▼
OCCLUSION ENGINE
│
▼
COMPOSITOR
│
▼
DETAIL / UPSCALE
│
▼
FFMPEG
│
▼
FINAL VIDEO
 
---
 
# 43. Definition of Done
 
The project is considered successful when a user can:
 
1. Load a video.
2. Select a person or object.
3. Provide one or more reference images.
4. Select an operation.
5. Press Generate.
6. Receive a transformed video.
7. Preserve original motion.
8. Preserve original camera movement.
9. Preserve unedited regions.
10. Maintain temporal consistency.
11. Maintain identity where requested.
12. Correctly handle occlusion.
13. Preserve or replace audio.
14. Export the final video.
15. Reopen the project and reproduce the result.
 
The system should work entirely locally wherever model licenses permit.
 
---
 
# 44. Primary Design Philosophy
 
The most important design principle is:
 
DO NOT BUILD AN "AI VIDEO GENERATOR."
 
Build a:
 
```
VIDEO TRANSFORMATION ENGINE
```
 
The distinction is critical.
 
A normal video generator asks:
 
```
"What video should I create?"
```
 
Genjutsu-OSS asks:
 
```
"What parts of this existing video should remain invariant,
 what parts should change,
 and what information must be transferred
 from the original video into the new generation?"
```
 
Therefore the fundamental abstraction is:
 
```
SOURCE VIDEO
      +
REFERENCES
      +
TRANSFORMATION
      +
INVARIANTS
      ↓
NEW VIDEO
```
 
The invariants are:
 
```
motion
timing
camera
composition
identity
lighting
background
objects
```
 
depending on the selected operation.
 
This invariant-based architecture is the central idea of Genjutsu-OSS.
 
---
 
# 45. Initial Open-Source Technology Stack
 
Recommended first implementation:
 
```
ComfyUI
Wan2.2
Wan2.2 Animate
ComfyUI-WanVideoWrapper
SAM 2
Video Depth Anything
DWPose
RAFT
OpenCV
PyTorch
FFmpeg
IP-Adapter
InsightFace
RIFE
Real-ESRGAN
```
 
Additional models should remain optional.
 
---
 
# 46. Repository Goal
 
The final GitHub repository should provide:
 
```
One-click installation
 
Automatic dependency checking
 
Model installation instructions
 
Prebuilt ComfyUI workflows
 
Example videos
 
Example references
 
Documentation
 
Troubleshooting
 
VRAM requirements
 
Performance benchmarks
 
Model licensing information
 
Developer API
 
Custom node documentation
```
 
The end result should be a reusable open-source framework for building Genjutsu-style video transformation workflows on top of ComfyUI and interchangeable open video models.
