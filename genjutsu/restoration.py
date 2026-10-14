"""Image scaling and export helpers; learned restoration remains opt-in."""
import numpy as np

def resize_frames(frames: list[np.ndarray],scale: float=2) -> list[np.ndarray]:
    if scale<=0: raise ValueError("scale must be > 0")
    import cv2
    return [cv2.resize(f,None,fx=scale,fy=scale,interpolation=cv2.INTER_LANCZOS4 if scale>1 else cv2.INTER_AREA) for f in frames]

def realesrgan(frames: list[np.ndarray],model_path: str|None=None,scale: int=2,*,disk_backed=False) -> list[np.ndarray]:
    if not model_path: raise RuntimeError("Real-ESRGAN needs an explicitly installed runtime and model weight path. No model weights are downloaded automatically.")
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError as e: raise RuntimeError("Install the Real-ESRGAN inference package and its compatible basicsr/PyTorch dependencies.") from e
    import cv2
    architecture=RRDBNet(num_in_ch=3,num_out_ch=3,num_feat=64,num_block=23,num_grow_ch=32,scale=4)
    engine=RealESRGANer(scale=4,model_path=model_path,model=architecture,half=True)
    sequence=(cv2.cvtColor(engine.enhance(cv2.cvtColor(f,cv2.COLOR_RGB2BGR),outscale=scale)[0],cv2.COLOR_BGR2RGB) for f in frames)
    if disk_backed:
        from .frame_store import FrameStore
        return FrameStore.from_frames(sequence)
    return list(sequence)

def restore_faces(frames,model_path,*,disk_backed=True):
    """Actual GFPGANer inference. Model files/runtime must already be installed."""
    from pathlib import Path
    if not Path(model_path).is_file():raise FileNotFoundError('GFPGAN local checkpoint is missing')
    try:from gfpgan import GFPGANer
    except ImportError as exc:raise RuntimeError('GFPGAN and a compatible BasicSR runtime are required; no weights are downloaded') from exc
    import cv2
    engine=GFPGANer(model_path=str(model_path),upscale=1,arch='clean',channel_multiplier=2,bg_upsampler=None)
    def sequence():
        for frame in frames:
            _,_,result=engine.enhance(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR),has_aligned=False,only_center_face=False,paste_back=True)
            if result is None:raise RuntimeError('GFPGAN returned no restored image')
            yield cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
    if disk_backed:
        from .frame_store import FrameStore
        return FrameStore.from_frames(sequence())
    return list(sequence())
