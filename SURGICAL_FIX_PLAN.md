# Surgical fix plan

Authority: all 46 sections of `specification.md`. Keep the existing repository, media I/O, disk-backed stores, model adapters and working tests. The current successful VACE smoke run proves execution, not character identity replacement or production readiness.

## Fix order

1. **Correct operation routing (sections 4, 18–22, 31).** Character replacement and motion transfer select Wan 2.2 Animate replacement mode. Object/product/outfit edits select supported masked VACE workflows. Never silently substitute generic VACE generation for character replacement when Animate assets are missing. Preserve explicit backend selection for advanced users, with visible capability limits.
2. **Connect automatic understanding and references (sections 1–3, 7–16).** Expose existing detection, SAM2 selection/tracking, pose, depth, flow and camera stages through the ComfyUI path. Produce model-specific conditioning automatically. Split this supplied six-panel face sheet into individual reference views without text labels; retain them in one character Reference Package. Do not present a contact sheet as an identity embedding. Connect Animate background conditioning, currently absent from its ComfyUI node, and pass the complete ControlPackage to generation.
3. **Enforce preservation (sections 5, 17–22, 25–27).** Select and track the requested subject automatically. Generate edit, preserve and soft-boundary masks internally. Preserve source pixels outside the selected region; account for foreground occluders and depth when compositing. Full-white masks belong only to explicitly selected full-frame recast/testing, never the default character workflow.
4. **Finish temporal and identity controls (sections 23–24, 39–40).** Connect flow-based repair and shot resets. Measure reference identity across generated frames with real supported embeddings. Add bounded correction/retry behavior through the selected backend's supported controls; report failure when identity remains unacceptable. No prompt-only identity guarantee or invented quality scores. Calibrate thresholds with actual outputs.
5. **Deliver the simple ComfyUI workflow (sections 6, 28–36, 42–43, 46).** User supplies source, operation, target, references and preservation choices, then presses Generate. Hide automatic masks/model-specific files behind the expandable graph. Ship all six graphical workflows, model/dependency preflight, shot state, cache/resume, project reopening, audio/export and optional restoration. Finish the missing restoration/interpolation paths using real supported implementations. Reuse existing modules rather than replacing the architecture.
6. **Prove production readiness (sections 33–35, 37–41, 43–46).** Run local regression tests plus real GPU acceptance for every operation, identity, preservation, occlusion, temporal continuity, multi-shot references, timing/audio, cache invalidation and resume. Verify Windows/Linux deployment and dependency/model licensing. Mark each specification requirement passed only with its acceptance evidence; failures stay visible in the capability matrix.

## Next-render rule

- Use a **Draft preview**, not another full-resolution, full-video run.
- Start with **5 frames, 1 denoising step, seed 1234**, CPU offload and restoration/upscaling disabled.
- Initial preview target: **144×256**, preserving the portrait source aspect ratio. Check the chosen real backend's constraints before inference; if it requires a larger minimum, select that minimum and disclose it. This size is a requested preview target, not a verified Animate minimum.
- Resize/crop all temporal controls together and retain the source frame timestamps/audio interval. Do not merely lower MP4 bitrate after expensive full-resolution inference.
- This lowest-cost render checks the repaired pipeline wiring. Identity and production-quality acceptance require a subsequent quality run; one-step output is not evidence of identity fidelity.
- Do not run the character preview through VACE while claiming Animate behavior. Animate and its preprocessing models are not currently installed/validated; disclose exact missing assets, storage and download sizes before requesting model-download approval.

## Completion boundary

The requested target is the complete specification. Automatic masking, character routing/conditioning, identity correction, occlusion reasoning and several restoration paths are not production-validated today. The interface can be simple while these stages run internally; they cannot be replaced with fake outputs or assumed model capabilities.
