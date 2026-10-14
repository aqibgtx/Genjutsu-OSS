import json
from genjutsu.controls import ControlPackage, Operation, Reference, ReferencePackage
from genjutsu.prompt import compile_prompt

def test_control_package_round_trip(tmp_path):
    c=ControlPackage(source_video="sample.mp4",operation=Operation.PRODUCT_SWAP,prompt="brushed metal")
    c.references=ReferencePackage(references=[Reference("watch.png","PRODUCT",.8)])
    path=tmp_path/"controls.json"; c.save(path); got=ControlPackage.load(path)
    assert got.operation is Operation.PRODUCT_SWAP
    assert got.references.references[0].weight==.8
    assert got.to_dict()["prompt"]=="brushed metal"

def test_compiled_prompt_records_invariants():
    c=ControlPackage(source_video="x",operation=Operation.OBJECT_SWAP,prompt="red watch")
    c.settings["preserve"]=["hand","camera"]
    result=compile_prompt(c)
    assert "Replace only the selected object" in result
    assert "hand, camera" in result

