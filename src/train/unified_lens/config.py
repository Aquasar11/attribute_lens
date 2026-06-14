"""
unified_lens.config
===================
Config dataclass for the unified-lens trainers, loaded from YAML.

One dataclass serves both v1 (proxy-only) and v2 (faithfulness-aware); the v2-only
fields are ignored by ``trainv1``. ``mask_source`` (imagenet_s | box) folds the former
``*_box`` script variants into a single switch.
"""

from dataclasses import dataclass, field, fields
from typing import List, Optional

import yaml


@dataclass
class UnifiedLensConfig:
    # --- data / model / io ---
    mask_source: str = "imagenet_s"           # "imagenet_s" | "box"
    data_dir: Optional[str] = None            # None -> default for mask_source
    save_dir: str = "outputs/unified_lens"
    model_name: str = "vit_large_patch14_clip_224.openai_ft_in1k"
    model_weights_path: Optional[str] = \
        "./model/pretrained_models/vit_large_patch14_clip_224.openai_ft_in1k.pt"

    # --- which ViT blocks get an affine ---
    layers: List[int] = field(default_factory=lambda: [22])

    # --- training ---
    batch_size: int = 64
    workers: int = 4
    epochs: int = 30
    patience: int = 5
    val_frac: float = 0.1
    seed: int = 42

    # --- optimizer ---
    lr: float = 1.0e-4
    wd: float = 0.01
    grad_clip: float = 1.0

    # --- proxy loss (v1 + v2) ---
    temperature: float = 3.0
    bg_weight: float = 1.0

    # --- v2 only: concentration + faithfulness surrogate ---
    conc_weight: float = 0.1
    grad_checkpoint: bool = False
    warmup_epochs: int = 3
    faith_weight: float = 1.0
    faith_fractions: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.4])
    faith_samples: int = 2
    faith_beta: float = 10.0
    del_weight: float = 1.0
    ins_weight: float = 1.0

    @classmethod
    def from_yaml(cls, path, **overrides):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}
