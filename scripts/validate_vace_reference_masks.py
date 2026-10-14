"""Run actual installed Diffusers mask packing; no weights or AI generation."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda','both'],default='cpu')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    import torch
    from diffusers import WanVACEPipeline
    from genjutsu.backends.vace import ReferenceMaskPaddingMixin
    class Compatible(ReferenceMaskPaddingMixin,WanVACEPipeline):pass
    # Fixture config supplies only the packing method's geometry. There is no
    # substitute VAE, transformer forward pass or generated video in this test.
    pipe=Compatible.__new__(Compatible)
    attributes={'vae_scale_factor_temporal':4,'vae_scale_factor_spatial':8,
        'transformer':SimpleNamespace(config=SimpleNamespace(patch_size=(1,2,2))),'transformer_2':None}
    for key,value in attributes.items():object.__setattr__(pipe,key,value)
    results=[]
    for device in ['cpu','cuda'] if args.device=='both' else [args.device]:
        for frames in [5,17]:
            for count in [0,1,2,6,13]:
                mask=torch.rand(1,1,frames,16,32,device=device)
                refs=[[torch.zeros(3,16,32,device=device) for _ in range(count)]]
                upstream=WanVACEPipeline.prepare_masks(pipe,mask,refs)
                corrected=pipe.prepare_masks(mask,refs)
                base=(frames+3)//4
                assert corrected.shape==(1,64,base+count,2,4)
                assert torch.equal(corrected[:,:,count:],upstream[:,:,-base:])
                assert torch.count_nonzero(corrected[:,:,:count])==0
                assert corrected.dtype==mask.dtype and corrected.device==mask.device
                results.append({'device':device,'source_frames':frames,'references':count,
                    'upstream_temporal':upstream.shape[2],'corrected_temporal':corrected.shape[2]})
    report={'status':'passed','scope':'actual installed SDK tensor packing only; no VAE/denoising quality claim','cases':results}
    if args.report:args.report.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
