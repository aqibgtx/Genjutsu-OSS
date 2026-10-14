"""Depth-ordered mask logic for approximate foreground/edited-region occlusion."""
from __future__ import annotations
import numpy as np

def visibility_mask(edit_mask: np.ndarray, edit_depth: np.ndarray, foreground_depth: np.ndarray|None=None, foreground_mask: np.ndarray|None=None, near_is_high: bool=False) -> np.ndarray:
    """Return visible edit pixels; configure inverse-depth models where larger means nearer."""
    visible=(edit_mask>0).copy()
    if foreground_depth is not None:
        if foreground_depth.shape!=edit_depth.shape: raise ValueError("Depth maps must have the same shape")
        occ=(edit_depth<foreground_depth) if near_is_high else (edit_depth>foreground_depth)
        if foreground_mask is not None: occ &= foreground_mask>0
        visible &= ~occ
    return visible.astype(np.uint8)*255

def composite_layers(original,generated,edit_mask,edit_depth,foreground_depth=None,foreground_mask=None,feather_px=4):
    from .compositing import composite
    visible=visibility_mask(edit_mask,edit_depth,foreground_depth,foreground_mask)
    return composite(original,generated,visible,feather_px=feather_px)
