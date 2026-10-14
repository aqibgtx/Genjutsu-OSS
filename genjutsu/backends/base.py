from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from ..controls import ControlPackage

class BackendUnavailable(RuntimeError): pass

class VideoBackend(ABC):
    name="base"
    @abstractmethod
    def generate(self, controls: ControlPackage, source_frames: list, output_dir: str|Path) -> list:
        """Return generated RGB frames. Implementations must perform actual inference."""
