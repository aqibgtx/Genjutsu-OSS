"""Face identity embeddings and scoring with an optional local InsightFace model."""
from __future__ import annotations
import numpy as np

class IdentityAcceptanceError(RuntimeError):
    """A completed generation failed its measured quality gate, not a transient error."""

def cosine_similarity(a, b) -> float:
    x=np.asarray(a,dtype=np.float32).ravel(); y=np.asarray(b,dtype=np.float32).ravel()
    if x.shape!=y.shape: raise ValueError("Embedding dimensions differ")
    den=float(np.linalg.norm(x)*np.linalg.norm(y))
    return float(np.dot(x,y)/den) if den else 0.0

class LocalInsightFace:
    def __init__(self,model_name="buffalo_l",root="models/insightface",providers=None):
        from pathlib import Path
        self.root=Path(root)
        # FaceAnalysis uses root/models/name. Checking another directory would
        # let its helper silently download weights despite our offline policy.
        model_dir=self.root/"models"/model_name
        if not model_dir.is_dir() or not list(model_dir.glob("*.onnx")): raise FileNotFoundError(f"InsightFace model files missing: {model_dir}. Place licensed .onnx weights locally; automatic downloads are disabled.")
        try: from insightface.app import FaceAnalysis
        except ImportError as e: raise RuntimeError("Install insightface and onnxruntime-gpu (or onnxruntime) to use identity embeddings.") from e
        selected=providers or ["CUDAExecutionProvider","CPUExecutionProvider"]
        self.app=FaceAnalysis(name=model_name,root=str(self.root),providers=selected,allowed_modules=["detection","recognition"])
        if 'detection' not in self.app.models or 'recognition' not in self.app.models:
            raise RuntimeError("Local identity model bundle needs both detection and recognition ONNX models")
        self.app.prepare(ctx_id=0 if "CUDAExecutionProvider" in selected else -1)
    def embeddings(self,frames):
        import cv2
        return [self.app.get(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)) for frame in frames]
    def score(self,reference_embedding,frames,masks=None):
        result=[]
        for index,frame in enumerate(frames):
            faces=self.embeddings([frame])[0]
            if masks is not None:
                mask=masks[index]
                faces=[face for face in faces if _face_inside_mask(face,mask)]
            result.append(max((cosine_similarity(reference_embedding,f.normed_embedding) for f in faces),default=0.0))
        return result

    def evaluate_reference(self,reference_image,frames,threshold:float=.55,masks=None):
        """Measure detected face similarity against a local reference image embedding."""
        from PIL import Image
        references=reference_image if isinstance(reference_image,list) else [reference_image]
        vectors=[];weights=[]
        for item in references:
            path=item.path if hasattr(item,'path') else str(item)
            weight=float(item.weight) if hasattr(item,'weight') else 1.
            if weight<=0:continue
            with Image.open(path) as image:array=np.asarray(image.convert('RGB'))
            ref_faces=self.embeddings([array])[0]
            if not ref_faces:raise RuntimeError(f"No face detected in identity reference: {path}")
            ref=max(ref_faces,key=lambda face:float(face.det_score))
            vectors.append(ref.normed_embedding);weights.append(weight)
        if not vectors:raise ValueError('Identity evaluation needs a nonzero-weight reference')
        embedding=np.average(np.asarray(vectors),axis=0,weights=weights)
        scores=self.score(embedding,frames,masks)
        # Missing faces count as zero rather than disappearing from the mean.
        # An average cannot conceal frames where the selected identity is lost.
        return {"metric":"InsightFace normalized embedding cosine similarity","scores":scores,
            "mean":float(np.mean(scores)) if scores else 0.0,"minimum":float(np.min(scores)) if scores else 0.0,
            "detected_frames":sum(score!=0 for score in scores),"total_frames":len(frames),"threshold":float(threshold),
            "passes_threshold":bool(scores and all(score>=threshold for score in scores)),"correction_performed":False}

def _face_inside_mask(face,mask):
    x1,y1,x2,y2=map(float,face.bbox)
    x,y=int((x1+x2)/2),int((y1+y2)/2)
    return 0<=y<mask.shape[0] and 0<=x<mask.shape[1] and mask[y,x]>0

def correct_identity(backend,controls,source_frames,generated,directory,evaluator,masks=None,threshold=.55,max_attempts=2):
    """Bounded real regeneration with measured acceptance, never fake face repair."""
    from copy import deepcopy
    from pathlib import Path
    import json
    reference=next((r for r in controls.references.references if r.category.upper()=="CHARACTER"),None)
    if reference is None:raise ValueError("Identity correction needs a CHARACTER reference")
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    best=generated
    references=[r for r in controls.references.references if r.category.upper()=='CHARACTER']
    report=evaluator.evaluate_reference(references,best,threshold,masks)
    attempts=[report]
    for attempt in range(max_attempts):
        if report['passes_threshold']:break
        local=deepcopy(controls)
        local.settings['seed']=int(controls.settings.get('seed',0))+attempt+1
        candidate=backend.generate(local,source_frames,directory/f"attempt_{attempt+1}")
        candidate_report=evaluator.evaluate_reference(references,candidate,threshold,masks)
        attempts.append(candidate_report)
        if candidate_report['passes_threshold'] or candidate_report['minimum']>report['minimum']:
            best,report=candidate,candidate_report
    final={**report,'correction_performed':len(attempts)>1,'attempts':attempts,'method':'bounded measured regeneration with supported backend seed control'}
    (directory/'identity_correction.json').write_text(json.dumps(final,indent=2))
    if not final['passes_threshold']:
        # Keep the real rejected candidate reviewable, without publishing it as
        # accepted output or putting it in the generation cache.
        from .frame_store import FrameStore
        FrameStore.from_frames(best,directory/'rejected_candidate')
        raise IdentityAcceptanceError(f"Identity acceptance failed after {len(attempts)} real candidates; minimum similarity {report['minimum']:.4f} < {threshold}. See {directory/'identity_correction.json'}")
    return best,final
