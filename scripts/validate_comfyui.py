#!/usr/bin/env python
"""Validate shipped API workflows against a live ComfyUI server; optionally queue real jobs."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import requests


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--url",default="http://127.0.0.1:8188")
    p.add_argument("--workflow-dir",default="workflows")
    p.add_argument("--assets-dir",help="Directory containing the source/reference/control files referenced by workflow inputs")
    p.add_argument("--queue",action="store_true",help="Submit each workflow to the real server and wait for successful completion")
    p.add_argument("--timeout",type=int,default=7200)
    args=p.parse_args()
    base=args.url.rstrip("/");
    try: response=requests.get(base+"/object_info",timeout=10); response.raise_for_status(); nodes=response.json()
    except Exception as exc: raise SystemExit(f"Cannot query live ComfyUI /object_info at {base}: {exc}")
    root=Path(args.workflow_dir); results=[]
    for path in sorted(root.glob("*.json")):
        workflow=json.loads(path.read_text(encoding="utf-8")); missing=sorted({node["class_type"] for node in workflow.values()}-nodes.keys())
        if missing:raise SystemExit(f"{path.name}: server is missing required node classes: {missing}")
        for node_id,node in workflow.items():
            schema=nodes[node["class_type"]].get("input",{}); accepted=set(schema.get("required",{}))|set(schema.get("optional",{}))|set(schema.get("hidden",{}))
            absent=set(schema.get("required",{}))-set(node.get("inputs",{})); unknown=set(node.get("inputs",{}))-accepted
            if absent or unknown:raise SystemExit(f"{path.name} node {node_id} ({node['class_type']}) input mismatch: missing={sorted(absent)}, unknown={sorted(unknown)}")
        result={"workflow":path.name,"node_types":sorted({n["class_type"] for n in workflow.values()}),"server_schema":"present"}
        if args.queue:
            if not args.assets_dir:raise SystemExit("--queue requires --assets-dir with real source/reference/mask/conditioning inputs")
            assets=Path(args.assets_dir).resolve()
            aliases={"source_video":assets/"source.mp4","video_path":assets/"source.mp4","reference_image":assets/"reference.png","reference":assets/"reference.png",
                "edit_mask_directory":assets/"mask","mask_directory":assets/"mask","mask_path":assets/"mask"/"00000000.png",
                "pose_video":assets/"src_pose.mp4","face_video":assets/"src_face.mp4","background_video":assets/"src_bg.mp4","mask_video":assets/"src_mask.mp4"}
            for node in workflow.values():
                for key,value in list(node.get("inputs",{}).items()):
                    if key in aliases and aliases[key].exists():node["inputs"][key]=str(aliases[key]);continue
                    if isinstance(value,str) and (key.endswith(("path","video","image","mask")) or "/" in value):
                        candidate=assets/Path(value).name
                        if candidate.exists():node["inputs"][key]=str(candidate)
                        elif "output" in key.lower():
                            destination=assets/"output";destination.mkdir(parents=True,exist_ok=True);node["inputs"][key]=str(destination/Path(value).name)
            queued=requests.post(base+"/prompt",json={"prompt":workflow},timeout=30); queued.raise_for_status(); prompt_id=queued.json()["prompt_id"]
            deadline=time.monotonic()+args.timeout; history=None
            while time.monotonic()<deadline:
                r=requests.get(base+f"/history/{prompt_id}",timeout=20); r.raise_for_status(); history=r.json().get(prompt_id)
                if history and history.get("status",{}).get("completed"):break
                if history and history.get("status",{}).get("status_str")=="error":raise SystemExit(f"{path.name} failed on ComfyUI: {history['status']}")
                time.sleep(2)
            if not history or not history.get("status",{}).get("completed"):raise SystemExit(f"{path.name} did not complete within {args.timeout}s (prompt {prompt_id})")
            result.update({"status":"completed_on_server","prompt_id":prompt_id})
        results.append(result)
    print(json.dumps({"server":base,"workflows":results,"queued":args.queue},indent=2))


if __name__=="__main__":main()
