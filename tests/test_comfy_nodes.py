import importlib.util
from pathlib import Path
import json

def test_custom_nodes_register_without_loading_gpu_models():
    package=Path(__file__).resolve().parents[1]/"ComfyUI-Genjutsu"
    spec=importlib.util.spec_from_file_location("genjutsu_comfy_test",package/"__init__.py",submodule_search_locations=[str(package)])
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert "GenjutsuWanVACEGenerator" in mod.NODE_CLASS_MAPPINGS
    assert "GenjutsuSAM2Segmenter" in mod.NODE_CLASS_MAPPINGS

def test_shipped_api_workflows_use_registered_nodes_and_valid_inputs():
    root=Path(__file__).resolve().parents[1]; package=root/"ComfyUI-Genjutsu"
    spec=importlib.util.spec_from_file_location("genjutsu_comfy_workflow_test",package/"__init__.py",submodule_search_locations=[str(package)])
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for path in (root/"workflows").glob("*.json"):
        graph=json.loads(path.read_text())
        for node in graph.values():
            cls=mod.NODE_CLASS_MAPPINGS[node["class_type"]]
            required=cls.INPUT_TYPES().get("required",{})
            assert set(node["inputs"])==set(required), f"{path.name} has stale inputs for {node['class_type']}"
