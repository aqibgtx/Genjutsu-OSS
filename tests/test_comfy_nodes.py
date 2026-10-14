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


def test_cpu_temporal_and_compositor_nodes_preserve_vfr_timing(multishot_source,tmp_path):
    import numpy as np
    from PIL import Image
    from genjutsu.video import probe_video
    package=Path(__file__).resolve().parents[1]/"ComfyUI-Genjutsu"
    spec=importlib.util.spec_from_file_location("genjutsu_comfy_media",package/"__init__.py",submodule_search_locations=[str(package)])
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    temporal=mod.NODE_CLASS_MAPPINGS["GenjutsuTemporalRepair"]()
    repaired=tmp_path/"repaired.mp4";temporal.repair(str(multishot_source),.1,str(repaired))
    info=probe_video(multishot_source);actual=probe_video(repaired)
    assert actual.frame_count==info.frame_count and np.allclose(actual.timestamps,info.timestamps,atol=.001)
    masks=tmp_path/"masks";masks.mkdir()
    for i in range(info.frame_count):Image.fromarray(np.full((info.height,info.width),255,np.uint8)).save(masks/f"{i:08d}.png")
    compositor=mod.NODE_CLASS_MAPPINGS["GenjutsuCompositor"]()
    output=tmp_path/"composite.mp4";compositor.composite_video(str(multishot_source),str(repaired),str(masks),str(output),0)
    final=probe_video(output)
    assert final.has_audio and np.allclose(final.timestamps,info.timestamps,atol=.001)
