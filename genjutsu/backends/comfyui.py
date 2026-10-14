"""ComfyUI HTTP API client for a user's actual API-format workflow."""
from __future__ import annotations
from pathlib import Path
import json, time, uuid, requests
from .base import VideoBackend, BackendUnavailable
from ..controls import ControlPackage

class ComfyUIBackend(VideoBackend):
    name="comfyui-http"
    def __init__(self,settings: dict):
        self.settings=dict(settings)
        self.url=settings.get("url","http://127.0.0.1:8188").rstrip("/")
        self.workflow=settings.get("workflow"); self.timeout=int(settings.get("timeout",3600))
        self.node_map=settings.get("node_map",{})
    def generate(self,controls: ControlPackage,source_frames: list,output_dir: str|Path)->list:
        Path(output_dir).mkdir(parents=True,exist_ok=True)
        if not self.workflow: raise BackendUnavailable("ComfyUI backend is selected but backend.workflow is unset. Configure an API-format workflow that runs a real video generation model and produces a video output.")
        path=Path(self.workflow)
        if not path.is_file(): raise FileNotFoundError(f"ComfyUI API workflow not found: {path}")
        workflow=json.loads(path.read_text(encoding="utf-8"))
        if "prompt" in workflow and isinstance(workflow["prompt"],dict): workflow=workflow["prompt"]
        if not isinstance(workflow,dict) or not workflow: raise ValueError("ComfyUI workflow must be a nonempty API-format node graph")
        ref=controls.references.references[0].path if controls.references.references else ""
        for key,value in {"prompt":controls.prompt,"negative_prompt":controls.negative_prompt,"seed":controls.settings.get("seed",0),"source_video":controls.source_video,"reference":ref,"operation":controls.operation.value,"edit_mask":controls.object_masks[0] if controls.object_masks else ""}.items():
            target=self.node_map.get(key)
            if target:
                node_id,input_name=target
                if node_id not in workflow or input_name not in workflow[node_id].get("inputs",{}): raise ValueError(f"Invalid workflow node_map for {key}: {target}")
                workflow[node_id]["inputs"][input_name]=value
        client_id=str(uuid.uuid4())
        try:
            resp=requests.post(self.url+"/prompt",json={"prompt":workflow,"client_id":client_id},timeout=30); resp.raise_for_status()
        except requests.RequestException as e: raise BackendUnavailable(f"Could not queue ComfyUI workflow at {self.url}: {e}") from e
        prompt_id=resp.json().get("prompt_id")
        if not prompt_id: raise BackendUnavailable(f"ComfyUI rejected workflow: {resp.text}")
        deadline=time.monotonic()+self.timeout
        while time.monotonic()<deadline:
            try:
                hist=requests.get(f"{self.url}/history/{prompt_id}",timeout=30); hist.raise_for_status(); data=hist.json().get(prompt_id,{})
            except requests.RequestException as e: raise BackendUnavailable(f"Lost connection to ComfyUI while waiting: {e}") from e
            status=data.get("status",{}).get("status_str")
            if status=="error": raise BackendUnavailable(f"ComfyUI generation failed: {data.get('status',{}).get('messages',[])}")
            if data.get("outputs"):
                (Path(output_dir)/"comfyui-history.json").write_text(json.dumps(data,indent=2),encoding="utf-8")
                node_id=self.node_map.get("output_node")
                outputs=[data["outputs"].get(str(node_id),{})] if node_id else list(data["outputs"].values())
                media=[]
                for output in outputs:
                    for key in ("videos","gifs","files"):
                        for entry in output.get(key,[]):
                            if entry.get("filename","").lower().endswith((".mp4",".mov",".webm",".mkv")): media.append(entry)
                if not media: raise BackendUnavailable("ComfyUI finished but no video output metadata was found. Use a video output node and set backend.node_map.output_node to its node id.")
                entry=media[0]
                try:
                    res=requests.get(self.url+"/view",params={"filename":entry["filename"],"subfolder":entry.get("subfolder",""),"type":entry.get("type","output")},timeout=120); res.raise_for_status()
                except requests.RequestException as e: raise BackendUnavailable(f"Could not download ComfyUI output video: {e}") from e
                target=Path(output_dir)/entry.get("subfolder","")/entry["filename"]; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(res.content)
                from ..video import read_frames
                result,_=read_frames(target,disk_backed=True)
                return result
            time.sleep(2)
        raise TimeoutError(f"ComfyUI workflow {prompt_id} did not finish within {self.timeout} seconds")
