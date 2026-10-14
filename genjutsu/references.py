"""Reusable reference package management; feature extraction is intentionally explicit."""
from pathlib import Path
from .controls import Reference, ReferencePackage

class ReferenceManager:
    CATEGORIES={"CHARACTER","PRODUCT","OBJECT","OUTFIT","LOCATION","STYLE","PROP","TEXTURE","LOGO"}
    def __init__(self): self.package=ReferencePackage()
    def add(self,path: str,category: str,weight: float=1.0,metadata: dict|None=None) -> Reference:
        p=Path(path).expanduser().resolve()
        if not p.is_file(): raise FileNotFoundError(p)
        cat=category.upper()
        if cat not in self.CATEGORIES: raise ValueError(f"Unsupported category {category}; choose {sorted(self.CATEGORIES)}")
        if not 0<=weight<=1: raise ValueError("Reference weight must be between 0 and 1")
        ref=Reference(str(p),cat,weight,metadata or {}); self.package.references.append(ref); return ref
    def save(self,path: str): self.package.metadata["package_file"]=str(Path(path).resolve()); self.package.metadata["reference_files"]=[r.path for r in self.package.references]

    @staticmethod
    def extract_classical_features(path: str) -> dict:
        """Measured color and ORB descriptors; these are not identity or semantic embeddings."""
        import cv2, numpy as np
        image=cv2.imread(path)
        if image is None: raise ValueError(f"Unsupported or unreadable reference image: {path}")
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        hist=cv2.calcHist([hsv],[0,1,2],None,[16,8,8],[0,180,0,256,0,256]); cv2.normalize(hist,hist)
        orb=cv2.ORB_create(nfeatures=500); _,desc=orb.detectAndCompute(image,None)
        return {"color_histogram":hist.flatten().tolist(),"orb_descriptor_count":0 if desc is None else int(len(desc)),"feature_kind":"classical_appearance_not_identity","width":int(image.shape[1]),"height":int(image.shape[0])}
