"""Build actual API and graphical workflows for the high-level registered node."""
from pathlib import Path
import json

OPERATIONS={'MotionTransfer':'motion_transfer','CharacterSwap':'character_swap','ObjectSwap':'object_swap',
            'ProductSwap':'product_swap','EnvironmentRecast':'environment_recast','FullGenjutsu':'full_recast'}

def build(root):
    root=Path(root);advanced=root/'advanced';graphical=root/'gui'
    advanced.mkdir(parents=True,exist_ok=True);graphical.mkdir(parents=True,exist_ok=True)
    for name,operation in OPERATIONS.items():
        path=root/(name+'.json')
        if path.is_file() and not (advanced/(name+'.api.json')).exists():
            existing=json.loads(path.read_text())
            if existing.get('1',{}).get('class_type')!='GenjutsuTransform':
                (advanced/(name+'.api.json')).write_text(json.dumps(existing,indent=2))
        values={'source_video':'genjutsu/source.mp4','operation':operation,'target':'person' if operation in {'motion_transfer','character_swap','environment_recast'} else 'object',
                'reference_images_json':'["genjutsu/reference.png"]','reference_layout':'images','prompt':'',
                'quality':'draft','seed':1234,'config_path':'genjutsu-runtime.yaml','output_path':f'output/{name}_draft.mp4'}
        api={'1':{'class_type':'GenjutsuTransform','inputs':values},'2':{'class_type':'GenjutsuVideoOutput','inputs':{'video_path':['1',0],'validation_report':['1',1]}}}
        path.write_text(json.dumps(api,indent=2)+'\n')
        widgets=list(values.values());widgets.insert(8,'fixed')
        graph={'last_node_id':2,'last_link_id':2,'nodes':[
            {'id':1,'type':'GenjutsuTransform','title':name+' — Automatic Genjutsu','pos':[80,100],'size':[650,540],
             'flags':{},'order':0,'mode':0,'inputs':[],
             'outputs':[{'name':'video_path','type':'STRING','links':[1],'slot_index':0},{'name':'validation_report','type':'STRING','links':[2],'slot_index':1}],
             'properties':{'Node name for S&R':'GenjutsuTransform'},'widgets_values':widgets},
            {'id':2,'type':'GenjutsuVideoOutput','pos':[850,100],'size':[420,200],'flags':{},'order':1,'mode':0,
             'inputs':[{'name':'video_path','type':'STRING','link':1,'widget':{'name':'video_path'}},{'name':'validation_report','type':'STRING','link':2,'widget':{'name':'validation_report'}}],'outputs':[],
             'properties':{'Node name for S&R':'GenjutsuVideoOutput'},'widgets_values':[None,None]}],
            'links':[[1,1,0,2,0,'STRING'],[2,1,1,2,1,'STRING']],'groups':[],'config':{},'extra':{'ds':{'scale':0.8,'offset':[30,60]}},'version':0.4}
        (graphical/(name+'.json')).write_text(json.dumps(graph,indent=2)+'\n')

if __name__=='__main__':build(Path(__file__).resolve().parents[1]/'workflows')
