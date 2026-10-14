"""Actual HTTP upload/callback tests; generation uses a test-only CPU filter."""
import json
import socket
from pathlib import Path
import numpy as np
from PIL import Image

from genjutsu.ui import build_ui
from genjutsu.video import probe_video, write_video


def test_gradio_real_uploads_single_png_sequence_video_and_analysis(multishot_source,tmp_path,cpu_backend,local_config,monkeypatch):
    from gradio_client import Client, handle_file
    monkeypatch.setattr("genjutsu.pipeline.create_backend",lambda config:cpu_backend)
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED","False")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
    demo=build_ui(str(local_config))
    try:
        demo.launch(server_name="127.0.0.1",server_port=port,prevent_thread_lock=True,quiet=True)
        client=Client(f"http://127.0.0.1:{port}",verbose=False)
        analysis=json.loads(client.predict(handle_file(str(multishot_source)),fn_index=0))
        assert analysis["frames"]==18 and analysis["shot_cuts_frame_indices"]==[6,12]
        masks=[]
        for index in range(18):
            path=tmp_path/f"frame{index:02d}.png"
            mask=np.zeros((48,64),np.uint8);mask[12:34,3:30]=255
            Image.fromarray(mask).save(path);masks.append(path)
        video=tmp_path/"mask.mp4"
        write_video([np.repeat(np.asarray(Image.open(p))[:,:,None],3,axis=2) for p in masks],video,12)
        for index,uploads in enumerate([[masks[0]],list(reversed(masks)),[video]]):
            output=tmp_path/f"ui{index}.mp4"
            preview,status=client.predict(handle_file(str(multishot_source)),"object_swap","Use config default","grayscale","",None,"OBJECT","",
                [handle_file(str(p)) for p in uploads],0,"{}",str(output),fn_index=1)
            assert status.startswith("Saved "),status
            assert preview is not None and output.is_file()
            assert probe_video(output).frame_count==18
    finally:
        demo.close()
