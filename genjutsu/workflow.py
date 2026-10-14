"""High-level ComfyUI transformation entry point using real existing stages."""
from copy import deepcopy
import json
from pathlib import Path
from .controls import Operation
from .routing import route_config
from .config import load_config

def resolve_input(path):
    candidate=Path(path).expanduser()
    if candidate.is_file():return str(candidate.resolve())
    if not candidate.is_absolute():
        try:
            import folder_paths
            root=Path(folder_paths.get_input_directory()).resolve()
            candidate=(root/candidate).resolve()
            if not candidate.is_relative_to(root):raise ValueError("Uploaded file path must stay inside ComfyUI/input")
            if candidate.is_file():return str(candidate)
        except ImportError:pass
    raise FileNotFoundError(f"Uploaded asset is missing: {path}")

def preflight(config, operation, target=None):
    """Read local files only; never download weights or initialize CUDA."""
    routed=route_config(config,Operation(operation))
    backend=routed['backend'];missing=[]
    model=Path(backend.get('model_id',''))
    if not (model/'model_index.json').is_file():missing.append(f"local model directory: {model}")
    else:
        index=json.loads((model/'model_index.json').read_text())
        expected='WanAnimatePipeline' if backend.get('mode')=='animate' else 'WanVACEPipeline'
        if index.get('_class_name')!=expected:missing.append(f"model pipeline must be {expected}, got {index.get('_class_name')}")
        for name,value in index.items():
            if isinstance(value,list) and len(value)==2 and value[0] and value[1]:
                component=model/name
                if not component.is_dir():missing.append(f"model component: {component}")
                if name in {'transformer','vae','text_encoder','image_encoder'} and not list(component.glob('*.safetensors')):
                    missing.append(f"model weights: {component}")
                for manifest in component.glob('*.index.json'):
                    mapping=json.loads(manifest.read_text()).get('weight_map',{})
                    for filename in sorted(set(mapping.values())):
                        if not (component/filename).is_file():missing.append(f"model shard: {component/filename}")
    if backend.get('mode')=='animate':
        prep=backend.get('wan_animate_preprocess',{})
        root=Path(prep.get('checkpoint_dir',''))
        required=[Path(prep.get('wan_repo',''))/'wan/modules/animate/preprocess/preprocess_data.py',
                  root/'det/yolov10m.onnx',root/'pose2d/vitpose_h_wholebody.onnx',root/'sam2/sam2_hiera_large.pt']
        for path in required:
            if not path.exists():missing.append(f"Animate preprocessing asset: {path}")
        pose=root/'pose2d/vitpose_h_wholebody.onnx'
        if pose.is_dir() and not (pose/'end2end.onnx').is_file():missing.append(f"ONNX graph: {pose/'end2end.onnx'}")
    elif Operation(operation) not in {Operation.FULL_RECAST,Operation.STYLE_RECAST}:
        selection=routed.get('selection',{})
        detector=Path(selection.get('detector_model',''))
        if not (detector/'config.json').is_file() or not list(detector.glob('*.safetensors')):
            missing.append(f"automatic target detector weights/config: {detector}")
        if not Path(selection.get('sam2_checkpoint','')).is_file():missing.append('automatic target SAM2 checkpoint')
        if not selection.get('sam2_config'):missing.append('automatic target SAM2 configuration')
    if routed.get('identity',{}).get('enabled') or routed.get('identity',{}).get('evaluate_only'):
        identity=routed['identity'];identity_dir=Path(identity.get('root','models/insightface'))/'models'/identity.get('model_name','buffalo_l')
        if not list(identity_dir.glob('*.onnx')):missing.append(f"local licensed identity models: {identity_dir}")
    if target and target.strip().lower()=='face':
        identity_dir=Path(routed.get('identity',{}).get('root','models/insightface'))/'models'/'buffalo_l'
        if not list(identity_dir.glob('*.onnx')):missing.append(f"local face detector/recognition models: {identity_dir}")
        if not Path(routed.get('selection',{}).get('sam2_checkpoint','')).is_file():missing.append('face tracking SAM2 checkpoint')
    if missing:raise RuntimeError("Genjutsu preflight failed before inference:\n"+'\n'.join(missing))
    return routed

def run_transform(source_video, reference_images_json, operation, target, prompt, output_path,
                  config_path, quality='draft', seed=1234, reference_layout='images',identity_lock=True):
    source=resolve_input(source_video)
    try:rows=json.loads(reference_images_json or '[]')
    except json.JSONDecodeError as exc:raise ValueError('References must be a JSON list of image paths; use the Upload references button') from exc
    if not isinstance(rows,list) or not rows:raise ValueError('Upload at least one reference image')
    refs=[]
    for row in rows:
        if isinstance(row,str):refs.append(resolve_input(row))
        elif isinstance(row,dict) and isinstance(row.get('path'),str):refs.append({**row,'path':resolve_input(row['path'])})
        else:raise ValueError('Each reference must be an image path or a reference record')
    if quality not in {'draft','source'}:raise ValueError('Quality must be draft or source')
    config=load_config(config_path)
    if Operation(operation) in {Operation.CHARACTER_SWAP,Operation.MOTION_TRANSFER,Operation.FULL_RECAST}:
        config.setdefault('identity',{})['enabled']=bool(identity_lock and quality=='source')
        config['identity']['evaluate_only']=bool(identity_lock and quality=='draft')
    if config.get('backend',{}).get('type')=='comfyui':
        raise ValueError('High-level Genjutsu requires its local runtime configuration with backend.type=auto; it cannot recursively submit itself to ComfyUI')
    if Operation(operation)!=Operation.FULL_RECAST:
        config=preflight(config,operation,target)
    output=Path(output_path).resolve();work=output.parent/(output.stem+'.inputs');work.mkdir(parents=True,exist_ok=True)
    if reference_layout=='face_sheet_2x3':
        if len(refs)!=1 or not isinstance(refs[0],str):raise ValueError('Face-sheet layout requires exactly one uploaded sheet')
        from .references import split_face_sheet
        refs=split_face_sheet(refs[0],work/'reference_views')
    elif reference_layout!='images':raise ValueError('Unknown reference layout')
    # Animate supports one primary image. Additional views remain in the
    # Reference Package for evaluation; no unsupported multi-image argument.
    if quality=='draft':
        from .preview import prepare_preview,DRAFT
        source=str(prepare_preview(source,work,width=DRAFT['width'],height=DRAFT['height'],frames=DRAFT['frames']))
        config['backend']['steps']=DRAFT['steps']
        config['backend']['chunk_frames']=5;config['backend']['overlap_frames']=0
        config['backend']['segment_frame_length']=5
        prep=config['backend'].get('wan_animate_preprocess')
        if prep is not None:
            from .video import probe_video
            info=probe_video(source);prep['resolution_area']=[info.width,info.height]
        config['upscale']={'enabled':False};config['restoration']={'enabled':False}
    from .pipeline import transform_video
    if Operation(operation)==Operation.FULL_RECAST:
        return run_full_recast(source,refs,prompt,output,config,seed,target)
    result=transform_video(source,output,operation,prompt,references=refs,config_path=None,seed=seed,
        runtime_config=config,selection_prompt=target or 'person',quality=quality,segment_frame_length=config['backend'].get('segment_frame_length',77))
    return str(result),str(result.parent/(result.stem+'.work')/'run.json')

def run_full_recast(source,references,prompt,output,config,seed,target):
    """Compose supported masked workflows; retain source performance between stages.

    Full-white VACE cannot enforce original motion, so full recast explicitly
    edits environment, character and selected props through separate stages.
    """
    from .pipeline import transform_video
    import shutil
    roles={}
    for row in references:
        if not isinstance(row,dict) or not row.get('category'):
            raise ValueError('Full recast needs categorized references: upload Character and Environment references using their buttons')
        roles.setdefault(row['category'].upper(),[]).append(row)
    if not roles.get('CHARACTER') or not roles.get('LOCATION'):
        raise ValueError('Full recast requires Character and Environment references')
    unsupported=set(roles)-{'CHARACTER','LOCATION','OBJECT','PRODUCT','OUTFIT','PROP'}
    if unsupported:raise ValueError(f'No verified full-recast stage for reference categories {sorted(unsupported)}')
    stages=[(Operation.ENVIRONMENT_RECAST,roles['LOCATION'],'person'),(Operation.CHARACTER_SWAP,roles['CHARACTER'],'person')]
    for category in ('OBJECT','PRODUCT','OUTFIT','PROP'):
        for reference in roles.get(category,[]):
            description=reference.get('metadata',{}).get('target') or target
            if not description or description in {'object','person'}:
                raise ValueError('Full-recast object/product/outfit references need a specific target description, e.g. watch or shirt')
            op={'PRODUCT':Operation.PRODUCT_SWAP,'OUTFIT':Operation.OUTFIT_SWAP}.get(category,Operation.OBJECT_SWAP)
            stages.append((op,[reference],description))
    jobs=[]
    for operation,refs,description in stages:
        stage_config=deepcopy(config);stage_config['backend']['type']='auto'
        # Identity evaluation/correction belongs to character generation.
        if operation!=Operation.CHARACTER_SWAP:
            stage_config.setdefault('identity',{})['enabled']=False;stage_config['identity']['evaluate_only']=False
        stage_config=preflight(stage_config,operation)
        jobs.append((operation,refs,description,stage_config))
    folder=output.parent/(output.stem+'.work');folder.mkdir(parents=True,exist_ok=True)
    current=source;results=[]
    for index,(operation,refs,description,stage_config) in enumerate(jobs):
        destination=folder/f'{index:02d}_{operation.value}.mp4'
        current=transform_video(current,destination,operation,prompt,references=refs,config_path=None,seed=seed+index,
            runtime_config=stage_config,selection_prompt=description,segment_frame_length=config['backend'].get('segment_frame_length',77))
        results.append({'operation':operation.value,'video':str(current),'report':str(destination.parent/(destination.stem+'.work')/'run.json')})
        (folder/'stages.json').write_text(json.dumps(results,indent=2))
    shutil.copy2(current,output)
    report=folder/'run.json';report.write_text(json.dumps({'operation':'full_recast','source':str(source),'output':str(output),
        'stages':results,'method':'masked environment -> Animate character -> selected masked props; no all-white motion fallback'},indent=2))
    return str(output),str(report)
