"""Portable .genjutsu project archives with checksummed assets and reproducible controls."""
from pathlib import Path
import hashlib, json, zipfile
from .controls import ControlPackage

def create_project(path: str|Path, controls: ControlPackage, include_source: bool=False) -> Path:
    dest=Path(path)
    if dest.suffix!=".genjutsu": dest=dest.with_suffix(".genjutsu")
    dest.parent.mkdir(parents=True,exist_ok=True)
    project={"format":"genjutsu-project-v1","controls":controls.to_dict(),"assets":[]}
    with zipfile.ZipFile(dest,"w",zipfile.ZIP_DEFLATED) as z:
        candidates=[Path(r.path) for r in controls.references.references]
        if include_source:candidates.append(Path(controls.source_video))
        for asset in candidates:
            if not asset.is_file(): continue
            digest=hashlib.sha256(asset.read_bytes()).hexdigest(); arc=f"assets/{digest[:12]}_{asset.name}"
            z.write(asset,arc); project["assets"].append({"original":str(asset.resolve()),"archive":arc,"sha256":digest})
        z.writestr("project.json",json.dumps(project,indent=2))
    return dest

def inspect_project(path: str|Path) -> dict:
    with zipfile.ZipFile(path) as z:
        data=json.loads(z.read("project.json")); names=set(z.namelist())
        data["missing_assets"]=[a["archive"] for a in data.get("assets",[]) if a["archive"] not in names]
        return data

def restore_project(path: str|Path,directory: str|Path) -> Path:
    """Safely extract verified assets and write a controls manifest with relocated references."""
    root=Path(directory).resolve(); root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        project=json.loads(z.read("project.json")); mapping={}
        for asset in project.get("assets",[]):
            arc=Path(asset["archive"])
            if arc.is_absolute() or ".." in arc.parts or not arc.parts or arc.parts[0]!="assets": raise ValueError(f"Unsafe project asset path: {arc}")
            destination=(root/arc).resolve()
            if root not in destination.parents: raise ValueError("Project asset escaped restore directory")
            content=z.read(arc.as_posix()); digest=hashlib.sha256(content).hexdigest()
            if digest!=asset["sha256"]: raise ValueError(f"Checksum mismatch for {arc}")
            destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(content); mapping[asset["original"]]=str(destination)
    controls=project["controls"]
    if controls.get("source_video") in mapping: controls["source_video"]=mapping[controls["source_video"]]
    for ref in controls.get("references",{}).get("references",[]):
        if ref.get("path") in mapping:ref["path"]=mapping[ref["path"]]
    output=root/"controls.json"; output.write_text(json.dumps(controls,indent=2),encoding="utf-8"); return output
