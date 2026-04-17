"""Configuration dataclasses for the tuned lens training pipeline."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def _dataclass_from_dict(klass: type, d: dict[str, Any]) -> Any:
    """Recursively instantiate a dataclass from a (possibly nested) dict."""
    if not hasattr(klass, "__dataclass_fields__"):
        return d
    field_types = {f.name: f.type for f in klass.__dataclass_fields__.values()}
    kwargs = {}
    for k, v in d.items():
        if k not in field_types:
            continue
        ft = klass.__dataclass_fields__[k].type
        # Resolve the actual type for nested dataclasses
        resolved = _resolve_type(ft, klass)
        if resolved is not None and hasattr(resolved, "__dataclass_fields__") and isinstance(v, dict):
            kwargs[k] = _dataclass_from_dict(resolved, v)
        else:
            kwargs[k] = v
    return klass(**kwargs)


def _resolve_type(type_str: str, parent_class: type) -> type | None:
    """Resolve a type annotation string to an actual type."""
    # Get the module where the parent class is defined
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


@dataclass
class ModelConfig:
    """Configuration for the vision model backbone."""
    model_name: str = "vit_large_patch14_clip_224.openai_ft_in1k"
    pretrained: bool = True
    weights_path: str | None = None
    target_layers: list[int] | None = None  # None = all layers
    freeze_model: bool = True
    patch_mode: bool = False  # capture all patch tokens instead of CLS token


@dataclass
class LensConfig:
    """Configuration for the lens architecture."""
    lens_type: str = "affine"  # "affine" | "mlp"
    bias: bool = True
    mlp_hidden_dim: int | None = None
    mlp_num_layers: int = 2
    init_from_head: bool = False
    init_from_pretrained: str | None = None  # Path to a saved lens .pt file
    dropout: float = 0.0
    use_patch_tokens: bool = False   # train on patch tokens instead of CLS token
    patch_neighbor_size: int = 3     # k — each patch sees a k×k neighborhood (must be odd)
    patch_border: int = 2            # exclude this many patch-rows/cols from each edge


@dataclass
class DataConfig:
    """Configuration for data loading."""
    imagenet_root: str = ""
    batch_size: int = 64
    num_workers: int = 4
    image_size: int = 224
    max_images_per_class: int | None = None  # None = use all images


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"  # "adam" | "adamw" | "sgd"
    scheduler: str = "cosine"  # "cosine" | "step" | "none"
    warmup_steps: int = 100
    max_epochs: int = 10
    gradient_accumulation_steps: int = 1
    val_check_interval: int | float = 1.0  # int = every N steps, float = fraction of epoch
    loss_type: str = "kld"      # "kld" | "ce" | "combined"
    ce_target: str = "model"    # "model" = model's argmax; "gt" = ground truth label
    ce_weight: float = 0.1
    temperature: float = 1.0
    grad_clip_norm: float = 1.0


@dataclass
class SweepConfig:
    """Configuration for Optuna hyperparameter sweep."""
    n_trials: int = 50
    max_epochs_per_trial: int = 3
    lr_range: list[float] = field(default_factory=lambda: [1e-5, 1e-2])
    batch_size_choices: list[int] = field(default_factory=lambda: [32, 64, 128])
    optimizer_choices: list[str] = field(default_factory=lambda: ["adam", "adamw", "sgd", "rmsprop", "nadam"])
    scheduler_choices: list[str] = field(default_factory=lambda: ["cosine", "cosine_warmup", "step", "exponential", "linear", "plateau", "none"])
    weight_decay_range: list[float] = field(default_factory=lambda: [1e-6, 1e-2])
    temperature_range: list[float] = field(default_factory=lambda: [0.5, 2.0])


@dataclass
class TunedLensConfig:
    """Top-level configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lens: LensConfig = field(default_factory=LensConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    output_dir: str = "outputs/tuned_lens"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> TunedLensConfig:
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


@dataclass
class PatchMapConfig:
    """Configuration for the patch map architecture."""
    map_type: str = "low_rank"  # "full" | "low_rank"
    rank: int = 128             # only used when map_type == "low_rank"
    fg_threshold: float = 0.80
    bg_threshold: float = 0.20
    neg_weight: float = 0.5
    neg_clip: float = 0.3


@dataclass
class PatchMapDataConfig:
    """Configuration for bbox-aware ImageNet data loading."""
    imagenet_root: str = ""
    bbox_dir_train: str = ""  # e.g. /opt/models/datasets/imagenet/boxes/train
    bbox_dir_val: str = ""    # e.g. /opt/models/datasets/imagenet/boxes/val
    batch_size: int = 32
    num_workers: int = 4
    max_images_per_class: int | None = None


@dataclass
class PatchMapTrainingConfig:
    """Configuration for patch map training loop."""
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"   # "adam" | "adamw" | "sgd" | "rmsprop"
    scheduler: str = "cosine"  # "cosine" | "step" | "plateau" | "none"
    warmup_steps: int = 100
    max_epochs: int = 20
    grad_clip_norm: float = 1.0
    val_check_interval: int | float = 1.0


@dataclass
class PatchMapFullConfig:
    """Top-level configuration for contrastive patch map training."""
    model: ModelConfig = field(default_factory=ModelConfig)
    patch_map: PatchMapConfig = field(default_factory=PatchMapConfig)
    data: PatchMapDataConfig = field(default_factory=PatchMapDataConfig)
    training: PatchMapTrainingConfig = field(default_factory=PatchMapTrainingConfig)
    output_dir: str = "outputs/patch_map"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> PatchMapFullConfig:
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
