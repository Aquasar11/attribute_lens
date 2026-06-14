"""Config dataclasses for the faithfulness and localization evaluations.

Each replaces the block of module-level globals the original scripts used. Both load
from a flat YAML via ``from_yaml`` (unknown keys are rejected so typos surface early).
"""

from dataclasses import dataclass, field, fields
from typing import List, Optional

import yaml


def _from_yaml(cls, path, **overrides):
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    data.update({k: v for k, v in overrides.items() if v is not None})
    return cls(**data)


@dataclass
class _EvalCommon:
    """Fields shared by both evaluations (model + lens checkpoint locations)."""
    # backbone
    model_name: str = "vit_large_patch14_clip_224.openai_ft_in1k"
    timm_weights_path: str = "./model/pretrained_models/vit_large_patch14_clip_224.openai_ft_in1k.pt"

    # lens checkpoints (which set is used depends on mapper_type)
    mapper_type: str = "unified"      # "unified" | "full" | "low_rank" | "direct_lens"
    mapper_weights_dir: str = "./outputs/clip/patch_map_full/best_maps"   # full / low_rank
    lens_weights_dir: str = "./outputs/clip/mlp_kld/best_lenses"          # full / low_rank
    unified_weights_dir: str = "./outputs/unified_lens/v1"                # unified
    direct_lens_path: str = "./direct_lens/layer22_ce/best_lens.pt"       # direct_lens
    direct_lens_layer: str = "layer_22"
    lens_out_dim: Optional[int] = None  # full / low_rank only; None = auto-detect

    # which ViT blocks to score
    target_layers: List[str] = field(default_factory=lambda: ["layer_22"])


@dataclass
class FaithfulnessConfig(_EvalCommon):
    # --- data ---
    image_dir: str = "data/first_100"
    y_hat_csv: str = "data/first_100/y_hat_baseline.csv"

    # --- outputs ---
    all_scores_out_file: str = "all_layers_attribution_scores.npz"
    all_metrics_out_csv: str = "all_layers_evaluation_metrics.csv"
    plots_out_dir: str = "faithfulness_plots"

    eval_batch: int = 128

    # --- neighbour-averaging kernel ---
    use_kernel: bool = True
    kernel_approach: str = "ring_sigma"           # "ring_sigma" | "multi_kernel"
    kernel_mode: str = "score"                    # "score" | "embedding"
    kernel_size: int = 7                          # ring_sigma: odd kernel side
    kernel_sigmas: List[float] = field(default_factory=lambda: [2.0, 1.5, 1.0])
    multi_kernel_sizes: List[int] = field(default_factory=lambda: [1, 3, 5, 7])
    multi_kernel_sigmas: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.5, 2.0])
    multi_kernel_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])

    # --- edge-aware (bilateral) weighting ---
    bilateral: bool = True
    bilateral_sigma_range: float = 1.0

    # --- score space used to rank/smooth (curves always use true P(y_hat)) ---
    score_space: str = "logprob"                  # "logprob" | "logit" | "prob"

    # --- multi-layer aggregation ---
    layer_agg: str = "off"                        # "off" | "uniform" | "confidence" | "ramp"
    layer_norm: str = "rank"                      # "rank" | "minmax"
    layer_conf_topk: int = 5
    layer_conf_gamma: float = 2.0
    eval_per_layer: bool = False
    agg_layer_name: str = "agg"

    @classmethod
    def from_yaml(cls, path, **overrides):
        return _from_yaml(cls, path, **overrides)


@dataclass
class LocalizationConfig(_EvalCommon):
    # --- data (annotated images + FG/BG masks) ---
    image_dir: str = "data/validation/annotated_dataset/sample/images"
    mask_dir: str = "data/validation/annotated_dataset/sample/masks_fg_bg"

    # --- output ---
    all_metrics_out_csv: str = "localization_metrics_results.csv"

    @classmethod
    def from_yaml(cls, path, **overrides):
        return _from_yaml(cls, path, **overrides)
