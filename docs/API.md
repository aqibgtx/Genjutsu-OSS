# Developer API

Install with `pip install -e ".[api]"` and start with `genjutsu api`. The server binds to `127.0.0.1` by default.

- `GET /health` returns process status.
- `POST /v1/analyze` accepts `{"source":"/path/to/video.mp4"}` and returns measured ControlPackage data.
- `POST /v1/transform` accepts source/output paths, operation, prompt, references, optional mask video/PNG sequence/directory, config path, and seed. It blocks until inference completes, then returns the output path and media metadata.

Paths are resolved by the server process. Wan models and preprocessing dependencies must be separately installed/configured. Wan Animate can generate its model-native pose/face/background/mask inputs by invoking Wan's official preprocessing checkout when `backend.wan_animate_preprocess` is configured; no model weights are downloaded automatically. Missing backend, model, selection model, mask or preprocessor assets returns HTTP 400 with the underlying error. Do not expose arbitrary local file access to untrusted clients.

Python entry points: `genjutsu.pipeline.analyze_video`, `genjutsu.pipeline.transform_video`, `genjutsu.controls.ControlPackage`, `genjutsu.references.ReferenceManager`, `genjutsu.project.create_project`, and `genjutsu.project.inspect_project`.
