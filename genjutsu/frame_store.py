"""Disk-backed sequences. Slices are views and never materialize a video in RAM."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import json
import tempfile
import numpy as np


class FrameView(Sequence):
    def __init__(self, parent, indices):
        self.parent, self.indices = parent, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return FrameView(self.parent, self.indices[index])
        return self.parent[self.indices[index]]


class FrameStore(Sequence):
    """Lossless PNG for uint8 images, NPY for other arrays; explicit directories
    belong to the caller.

    Temporary stores live as long as their store or any slice of it. No global
    array, mapping, or decoded-frame cache is retained. Finish writes a manifest
    last, so an interrupted store cannot be mistaken for a completed cache.
    """
    def __init__(self, directory=None):
        self._temporary = tempfile.TemporaryDirectory(prefix="genjutsu-frames-") if directory is None else None
        self.directory = Path(self._temporary.name if self._temporary else directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.shape = None
        self.dtype = None
        self.storage = None
        self.metadata = {}

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return FrameView(self, range(self.count)[index])
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        if self.storage=="png":
            from PIL import Image
            with Image.open(self.directory/f"{index:08d}.png") as image:
                return np.array(image)
        return np.load(self.directory / f"{index:08d}.npy", allow_pickle=False)

    def _save(self,index,frame):
        if self.storage=="png":
            from PIL import Image
            Image.fromarray(frame).save(self.directory/f"{index:08d}.png",compress_level=1)
        else:
            np.save(self.directory/f"{index:08d}.npy",frame,allow_pickle=False)

    def append(self, frame):
        frame = np.asarray(frame)
        if frame.dtype.hasobject:
            raise ValueError("Frame stores cannot contain object arrays")
        if self.shape is None:
            self.shape, self.dtype = frame.shape, str(frame.dtype)
            self.storage="png" if frame.dtype==np.uint8 and (frame.ndim==2 or (frame.ndim==3 and frame.shape[2] in {3,4})) else "npy"
        if frame.shape != self.shape or str(frame.dtype) != self.dtype:
            raise ValueError("Frame geometry or dtype changes within sequence")
        self._save(self.count,frame)
        self.count += 1

    def extend(self, frames):
        for frame in frames:
            self.append(frame)

    def set(self, index, frame):
        if not 0 <= index < self.count:
            raise IndexError(index)
        frame=np.asarray(frame)
        if frame.shape!=self.shape or str(frame.dtype)!=self.dtype:
            raise ValueError("Frame geometry or dtype changes within sequence")
        self._save(index,frame)

    def finish(self, **metadata):
        self.metadata = metadata
        manifest = {"count": self.count, "shape": self.shape, "dtype": self.dtype, "storage":self.storage,"metadata": metadata}
        temporary = self.directory / "manifest.tmp"
        temporary.write_text(json.dumps(manifest), encoding="utf-8")
        temporary.replace(self.directory / "manifest.json")
        return self

    @classmethod
    def open(cls, directory):
        store = cls(directory)
        data = json.loads((store.directory / "manifest.json").read_text(encoding="utf-8"))
        store.count = int(data["count"])
        store.shape = tuple(data["shape"]) if data["shape"] else None
        store.dtype = data["dtype"]
        store.storage=data.get("storage","npy")
        store.metadata = data.get("metadata", {})
        for frame in store:
            if frame.shape != store.shape or str(frame.dtype) != store.dtype:
                raise ValueError("Corrupted frame store")
        return store

    @classmethod
    def from_frames(cls, frames, directory=None):
        store = cls(directory)
        store.extend(frames)
        return store.finish()


class ImageSequence(Sequence):
    """Lazy image paths; a single static mask can repeat without disk/RAM copies."""
    def __init__(self, paths, count=None, shape=None, mode="L"):
        from PIL import Image
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("Image sequence is empty")
        self.count = count if count is not None else len(self.paths)
        self.mode = mode
        if len(self.paths) not in {1, self.count}:
            raise ValueError(f"Mask sequence has {len(self.paths)} frames but source has {self.count}")
        for path in self.paths:
            with Image.open(path) as image:
                if shape is not None and image.size != shape[::-1]:
                    raise ValueError(f"Mask {path} dimensions do not match source {shape[::-1]}")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        from PIL import Image
        if isinstance(index, slice):
            return FrameView(self, range(self.count)[index])
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        with Image.open(self.paths[0] if len(self.paths) == 1 else self.paths[index]) as image:
            return np.asarray(image.convert(self.mode))
