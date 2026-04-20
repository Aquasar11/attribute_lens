"""Configuration dataclasses for the attribution evaluation pipeline."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# YAML-loading helpers (same idiom as tuned_lens/config.py)
# ---------------------------------------------------------------------------

def _dataclass_from_dict(klass: type, d: dict[str, Any]) -> Any:
    """Recursively instantiate a dataclass from a (possibly nested) dict."""
    if not hasattr(klass, "__dataclass_fields__"):
        return d
    kwargs = {}
    for k, v in d.items():
        if k not in klass.__dataclass_fields__:
            continue
        ft = klass.__dataclass_fields__[k].type
        resolved = _resolve_type(ft, klass)
        if resolved is not None and hasattr(resolved, "__dataclass_fields__") and isinstance(v, dict):
            kwargs[k] = _dataclass_from_dict(resolved, v)
        else:
            kwargs[k] = v
    return klass(**kwargs)


def _resolve_type(type_str: Any, parent_class: type) -> type | None:
    import sys
    module = sys.modules[parent_class.__module__]
    type_map = {
        name: obj for name, obj in vars(module).items()
        if hasattr(obj, "__dataclass_fields__")
    }
    if isinstance(type_str, str):
        return type_map.get(type_str)
    if isinstance(type_str, type):
        return type_str
    return None


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class ModelSection:
    """Vision model backbone settings."""
    model_name: str = "vit_large_patch14_clip_224.openai_ft_in1k"
    weights_path: str | None = None
    # None = discover from available lens checkpoints; list to restrict
    target_layers: list[int] | None = None


@dataclass
class LensSection:
    """Paths and settings for trained lens checkpoints."""
    cls_lens_dir: str = ""          # directory containing layer_*.pt CLS lens files
    patch_lens_dir: str = ""        # directory containing layer_*.pt patch lens files
    patch_map_dir: str = ""         # directory containing layer_*.pt patch map checkpoints (for PatchMapCLSLensScorer)
    patch_neighbor_size: int = 3    # must match the value used during training
    patch_border: int = 2           # must match the value used during training
    means_path: str = ""            # .pt file with precomputed mean embeddings (for CLS scorer)


@dataclass
class LayerAvgSection:
    """Layer-wise averaged scoring settings."""
    enabled: bool = False
    min_layer: int | None = None   # inclusive; None = first available target layer
    max_layer: int | None = None   # inclusive; None = last available target layer
    weights_path: str = ""         # empty = uniform; path to .pt from fg_bg_layer_weights notebook


@dataclass
class NeighborAvgSection:
    """Neighbor-averaging settings for patches."""
    enabled: bool = False
    size: int = 3                  # N in N×N square of neighbors; must be odd
    mode: str = "score"            # "score" | "embedding" | "both"


@dataclass
class EvalSection:
    """Evaluation settings."""
    # Image sources — image_paths takes precedence over image_dir
    image_paths: list[str] = field(default_factory=list)
    image_dir: str = ""             # recursively scanned for .jpg/.jpeg/.png/.JPEG

    num_images: int | None = None   # None = all; positive int limits total images
    num_save_images: int | None = None  # None = save all; positive int saves per-image PNGs/JSON for a random subset only

    output_dir: str = "outputs/attribution"
    device: str = "cuda"            # "cuda" or "cpu"
    scorer_type: str = "both"       # "cls" | "patch" | "both"

    # Gaussian blur baseline for insertion/deletion
    blur_kernel_size: int = 55
    blur_sigma: float = 10.0

    # Batch sizes
    # How many perturbed images to stack into one forward pass during
    # insertion/deletion curves.  Larger = faster (fewer GPU launches).
    # Rule of thumb: as large as GPU VRAM allows (128–512 for ViT-L on 24 GB).
    perturbation_batch_size: int = 128
    # How many images to stack for the feature-extraction forward pass.
    extraction_batch_size: int = 8

    # Precision
    use_fp16: bool = True   # FP16 autocast for perturbation forward passes; no accuracy loss at inference

    # Visualization
    heatmap_colormap: str = "hot"
    heatmap_alpha: float = 0.6
    plot_dpi: int = 150

    # Optional post-processing modes
    layer_avg: LayerAvgSection = field(default_factory=LayerAvgSection)
    neighbor_avg: NeighborAvgSection = field(default_factory=NeighborAvgSection)


@dataclass
class AttributionConfig:
    """Top-level attribution evaluation configuration."""
    model: ModelSection = field(default_factory=ModelSection)
    lens: LensSection = field(default_factory=LensSection)
    eval: EvalSection = field(default_factory=EvalSection)
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> AttributionConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        if raw is None:
            return cls()
        return _dataclass_from_dict(cls, raw)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
