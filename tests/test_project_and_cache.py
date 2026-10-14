import json, zipfile
from pathlib import Path
from genjutsu.cache import StageCache
from genjutsu.controls import ControlPackage, Reference, ReferencePackage
from genjutsu.project import create_project, inspect_project, restore_project

def test_project_archive_checksums_reference(tmp_path):
    ref=tmp_path/"ref.txt"; ref.write_text("reference")
    controls=ControlPackage(source_video="source.mp4"); controls.references=ReferencePackage(references=[Reference(str(ref),"OBJECT",1)])
    archive=create_project(tmp_path/"take",controls)
    result=inspect_project(archive)
    assert result["format"]=="genjutsu-project-v1" and result["missing_assets"]==[]
    with zipfile.ZipFile(archive) as z: assert "project.json" in z.namelist() and len(z.namelist())==2

def test_cache_key_depends_on_stage_and_inputs(tmp_path):
    cache=StageCache(tmp_path)
    assert cache.key("pose",{"source":"a"})!=cache.key("depth",{"source":"a"})
    def write(p): (p/"value").write_text("measured")
    path,hit=cache.get_or_create("stage",{"x":1},write)
    assert not hit and (path/"value").exists()
    path2,hit=cache.get_or_create("stage",{"x":1},lambda p: (_ for _ in ()).throw(AssertionError()))
    assert hit and path==path2

def test_project_restore_verifies_and_relocates_assets(tmp_path):
    source=tmp_path/"source.mp4"; source.write_bytes(b"video-bytes")
    ref=tmp_path/"ref.png"; ref.write_bytes(b"reference-bytes")
    controls=ControlPackage(source_video=str(source)); controls.references=ReferencePackage(references=[Reference(str(ref),"CHARACTER",.9)])
    archive=create_project(tmp_path/"portable.genjutsu",controls,include_source=True)
    restored=restore_project(archive,tmp_path/"restored")
    loaded=ControlPackage.load(restored)
    assert Path(loaded.source_video).read_bytes()==b"video-bytes"
    assert Path(loaded.references.references[0].path).read_bytes()==b"reference-bytes"
