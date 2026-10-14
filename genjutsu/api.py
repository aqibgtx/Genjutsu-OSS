"""Optional local developer REST API; import requires FastAPI."""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .pipeline import analyze_video, transform_video
from .controls import Operation
from .video import probe_video

app=FastAPI(title="Genjutsu-OSS API",version="0.1.0",description="Local, model-backed video transformation developer API")
class AnalyzeRequest(BaseModel): source: str
class TransformRequest(BaseModel):
    source: str; output: str; operation: Operation; prompt: str; negative_prompt: str=""; references: list[str]=Field(default_factory=list); edit_mask: str|None=None; config: str="config.yaml"; seed: int=0
@app.get("/health")
def health(): return {"status":"ok","application":"genjutsu-oss"}
@app.post("/v1/analyze")
def analyze(req: AnalyzeRequest):
    try:
        c,_=analyze_video(req.source); return c.to_dict()
    except Exception as e: raise HTTPException(400,str(e)) from e
@app.post("/v1/transform")
def transform(req: TransformRequest):
    try:
        out=transform_video(req.source,req.output,req.operation,req.prompt,req.references,req.edit_mask,req.config,req.negative_prompt,req.seed); return {"output":str(out),"metadata":probe_video(out).__dict__}
    except Exception as e: raise HTTPException(400,str(e)) from e
