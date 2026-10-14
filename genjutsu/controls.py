"""Serializable, model-independent transformation controls."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import json

class Operation(str, Enum):
    MOTION_TRANSFER = "motion_transfer"
    CHARACTER_SWAP = "character_swap"
    OBJECT_SWAP = "object_swap"
    PRODUCT_SWAP = "product_swap"
    OUTFIT_SWAP = "outfit_swap"
    ENVIRONMENT_RECAST = "environment_recast"
    FULL_RECAST = "full_recast"
    STYLE_RECAST = "style_recast"

@dataclass
class Reference:
    path: str
    category: str = "OBJECT"
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ReferencePackage:
    references: list[Reference] = field(default_factory=list)
    identity_embedding: list[float] | None = None
    appearance_features: dict[str, Any] = field(default_factory=dict)
    object_features: dict[str, Any] = field(default_factory=dict)
    style_features: dict[str, Any] = field(default_factory=dict)
    masks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ControlPackage:
    source_video: str
    fps: float = 0.0
    width: int = 0
    height: int = 0
    frame_count: int = 0
    duration: float = 0.0
    audio_metadata: dict[str, Any] = field(default_factory=dict)
    scene_cuts: list[int] = field(default_factory=list)
    subject_masks: list[str] = field(default_factory=list)
    object_masks: list[str] = field(default_factory=list)
    preservation_masks: list[str] = field(default_factory=list)
    tracking_data: dict[str, Any] = field(default_factory=dict)
    pose_sequence: list[Any] = field(default_factory=list)
    optical_flow: dict[str, Any] = field(default_factory=dict)
    depth_sequence: list[str] = field(default_factory=list)
    camera_motion: dict[str, Any] = field(default_factory=dict)
    references: ReferencePackage = field(default_factory=ReferencePackage)
    operation: Operation = Operation.OBJECT_SWAP
    prompt: str = ""
    negative_prompt: str = ""
    temporal_settings: dict[str, float] = field(default_factory=lambda: {"strength": .4, "smoothing": .25, "repair": .25})
    identity_strength: float = .75
    motion_strength: float = 1.0
    reference_strength: float = .8
    preservation_strength: float = 1.0
    settings: dict[str, Any] = field(default_factory=dict)
    fps_rational: str = "0/1"
    time_base: str = "1/1000000"
    frame_pts: list[int] = field(default_factory=list)
    frame_durations: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    source_frame_indices: list[int] = field(default_factory=list)
    source_start_time: float = 0.0
    is_vfr: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["operation"] = self.operation.value
        return data

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ControlPackage":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        refs = data.pop("references", {})
        refs["references"] = [Reference(**r) for r in refs.get("references", [])]
        data["references"] = ReferencePackage(**refs)
        data["operation"] = Operation(data.get("operation", Operation.OBJECT_SWAP.value))
        return cls(**data)
