"""Face identity embeddings and scoring with an optional local InsightFace model."""
from __future__ import annotations
import numpy as np

def cosine_similarity(a, b) -> float:
    x=np.asarray(a,dtype=np.float32).ravel(); y=np.asarray(b,dtype=np.float32).ravel()
    if x.shape!=y.shape: raise ValueError("Embedding dimensions differ")
    den=float(np.linalg.norm(x)*np.linalg.norm(y))
    return float(np.dot(x,y)/den) if den else 0.0

class LocalInsightFace:
    def __init__(self,model_name="buffalo_l",root="models/insightface",providers=None):
        from pathlib import Path
        self.root=Path(root)
        model_dir=self.root/model_name
        if not model_dir.is_dir() or not list(model_dir.glob("*.onnx")): raise FileNotFoundError(f"InsightFace model files missing: {model_dir}. Place licensed .onnx weights locally; automatic downloads are disabled.")
        try: from insightface.app import FaceAnalysis
        except ImportError as e: raise RuntimeError("Install insightface and onnxruntime-gpu (or onnxruntime) to use identity embeddings.") from e
        selected=providers or ["CUDAExecutionProvider","CPUExecutionProvider"]
        self.app=FaceAnalysis(name=model_name,root=str(self.root),providers=selected); self.app.prepare(ctx_id=0 if "CUDAExecutionProvider" in selected else -1)
    def embeddings(self,frames):
        import cv2
        return [self.app.get(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)) for frame in frames]
    def score(self,reference_embedding,frames):
        result=[]
        for frame in frames:
            faces=self.embeddings([frame])[0]
            result.append(max((cosine_similarity(reference_embedding,f.normed_embedding) for f in faces),default=0.0))
        return result

    def evaluate_reference(self,reference_image,frames,threshold:float=.55):
        """Measure detected face similarity against a local reference image embedding."""
        from PIL import Image
        image=np.asarray(Image.open(reference_image).convert("RGB"))
        ref_faces=self.embeddings([image])[0]
        if not ref_faces:raise RuntimeError(f"No face detected in identity reference: {reference_image}")
        ref=max(ref_faces,key=lambda face:float(face.det_score))
        scores=self.score(ref.normed_embedding,frames)
        valid=[x for x in scores if x>0]
        return {"metric":"InsightFace normalized embedding cosine similarity","scores":scores,
            "mean":float(np.mean(valid)) if valid else 0.0,"minimum":float(np.min(valid)) if valid else 0.0,
            "detected_frames":len(valid),"total_frames":len(frames),"threshold":float(threshold),
            "passes_threshold":bool(valid and np.mean(valid)>=threshold),"correction_performed":False}
