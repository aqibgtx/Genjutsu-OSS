from __future__ import annotations
from pathlib import Path
import hashlib, json, os

class StageCache:
    def __init__(self,root: str|Path,enabled: bool=True): self.root=Path(root); self.enabled=enabled
    @staticmethod
    def key(stage: str, inputs: dict) -> str:
        raw=json.dumps({"stage":stage,"inputs":inputs},sort_keys=True,default=str).encode(); return hashlib.sha256(raw).hexdigest()
    def path(self,stage: str,key: str) -> Path: return self.root/stage/key
    def get_or_create(self,stage: str,inputs: dict,writer):
        key=self.key(stage,inputs); p=self.path(stage,key)
        if self.enabled and (p/"complete.json").exists(): return p,True
        p.mkdir(parents=True,exist_ok=True); writer(p)
        if self.enabled: (p/"complete.json").write_text(json.dumps({"key":key,"stage":stage}),encoding="utf-8")
        return p,False
