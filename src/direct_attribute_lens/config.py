"""Configuration dataclasses for the direct_attribute_lens training pipeline."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ── YAML / dataclass helpers (same pattern as tuned_lens/config.py) ──────────

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


def _resolve_type(type_str: str, parent_class: type) -> type | None:
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


# ── Config sections ───────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    model_name: str = "vit_large_patch14_clip_224.openai_ft_in1k"
    pretrained: bool = True
    weights_path: str | None = None
    head_weights_path: str | None = None
    target_layer: int = 6   # single layer per run
    freeze_model: bool = True


@dataclass
class PrecomputeConfig:
    imagenet_root: str = ""
    max_images_per_class: int = 500          # images per class for train split
    val_max_images_per_class: int = 10       # images per class for val split (max 50)
    chunk_size: int = 50_000                 # images per on-disk chunk
    precompute_batch_size: int = 256         # backbone forward pass batch size
    num_workers: int = 16
    cache_dir: str = "outputs/direct_lens/cache"


@dataclass
class LensConfig:
    lens_type: str = "embedding"   # "embedding" (d_model→d_model) | "direct" (d_model→num_classes)
    bias: bool = True


@dataclass
class TrainingConfig:
    lens_batch_size: int = 16384   # tokens per gradient step (GPU linear op, can be very large)
    num_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"        # "adam" | "adamw" | "sgd"
    loss_type: str = "kld"         # "kld" | "ce" | "combined" | "best_ce"
    ce_weight: float = 0.1         # weight of CE term in combined loss
    best_fraction: float = 0.2     # fraction of lowest-loss samples used by best_ce loss
    temperature: float = 1.0       # softmax temperature for KLD
    grad_clip_norm: float = 1.0    # 0 to disable
    val_interval: int = 5          # validate every N epochs
    # ReduceLROnPlateau scheduler — monitors training loss, reduces when it stops improving
    scheduler: str = "plateau"     # "plateau" | "none"
    scheduler_patience: int = 3    # epochs with no improvement before reducing LR
    scheduler_factor: float = 0.5  # multiplicative factor for LR reduction
    scheduler_min_lr: float = 1e-7 # floor for LR


@dataclass
class DirectLensConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    precompute: PrecomputeConfig = field(default_factory=PrecomputeConfig)
    lens: LensConfig = field(default_factory=LensConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "outputs/direct_lens"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> DirectLensConfig:
        with open(path) as f:
            d = yaml.safe_load(f)
        return _dataclass_from_dict(cls, d or {})

    def to_yaml(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
