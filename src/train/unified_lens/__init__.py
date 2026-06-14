"""Unified lens: a single affine per ViT block mapping patch tokens to classifier space.

Two trainers share the dataset / model plumbing in :mod:`unified_lens.lens` and
:mod:`unified_lens.data`:

- ``trainv1`` -- proxy-only (FG distill to y_hat, BG to uniform).
- ``trainv2`` -- proxy + differentiable deletion/insertion faithfulness surrogate.

Supervision masks come from either ImageNet-S segmentation or ImageNet box unions,
selected by ``mask_source`` in the YAML config.
"""

from .config import UnifiedLensConfig
from .data import FGBGMaskDataset, resolve_data_dir
from .lens import AffineLens, MultiLayerHook, build_head_fn, load_vit

__all__ = [
    "UnifiedLensConfig",
    "FGBGMaskDataset",
    "resolve_data_dir",
    "AffineLens",
    "MultiLayerHook",
    "build_head_fn",
    "load_vit",
]
