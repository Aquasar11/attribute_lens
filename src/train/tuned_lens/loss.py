"""Loss functions for tuned lens training.

The KLD loss always targets the model's final-layer soft distribution.
The CE loss can target either the model's argmax prediction or the ground truth label,
controlled by TrainingConfig.ce_target ("model" | "gt").
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .config import TrainingConfig


def kl_divergence_loss(
    lens_logits: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL(target || lens): measures how well the lens approximates the target distribution.

    Both inputs are raw logits. Temperature-scaled softmax is applied internally.
    Uses F.kl_div with log_target=True for numerical stability.
    """
    target_log_probs = F.log_softmax(target_logits / temperature, dim=-1)
    lens_log_probs = F.log_softmax(lens_logits / temperature, dim=-1)
    return F.kl_div(lens_log_probs, target_log_probs, log_target=True, reduction="batchmean")


def cross_entropy_loss(
    lens_logits: torch.Tensor,
    target_logits: torch.Tensor,
    gt_labels: torch.Tensor | None = None,
    use_gt: bool = False,
) -> torch.Tensor:
    """CE between lens predictions and a class target.

    Args:
        lens_logits: Raw logits from the lens [B, num_classes].
        target_logits: Final model logits [B, num_classes].
        gt_labels: Ground truth integer labels [B]. Required when use_gt=True.
        use_gt: If True, use gt_labels as target; if False, use argmax(target_logits).
    """
    if use_gt:
        if gt_labels is None:
            raise ValueError("gt_labels must be provided when use_gt=True")
        return F.cross_entropy(lens_logits, gt_labels)
    return F.cross_entropy(lens_logits, target_logits.argmax(dim=-1))


def combined_loss(
    lens_logits: torch.Tensor,
    target_logits: torch.Tensor,
    gt_labels: torch.Tensor | None = None,
    use_gt: bool = False,
    temperature: float = 1.0,
    ce_weight: float = 0.1,
) -> torch.Tensor:
    """KLD + ce_weight * CE."""
    kld = kl_divergence_loss(lens_logits, target_logits, temperature)
    ce = cross_entropy_loss(lens_logits, target_logits, gt_labels, use_gt)
    return kld + ce_weight * ce


LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


def get_loss_fn(config: TrainingConfig) -> LossFn:
    """Factory that returns the appropriate loss function based on config.

    All returned functions have the signature:
        loss_fn(lens_logits, target_logits, gt_labels) -> Tensor

    gt_labels is only used when config.loss_type in ("ce", "combined")
    and config.ce_target == "gt".
    """
    use_gt = config.ce_target == "gt"

    if config.loss_type == "ce":
        def _ce_loss(
            lens_logits: torch.Tensor,
            target_logits: torch.Tensor,
            gt_labels: torch.Tensor | None,
        ) -> torch.Tensor:
            return cross_entropy_loss(lens_logits, target_logits, gt_labels, use_gt)
        return _ce_loss

    elif config.loss_type == "combined":
        def _combined_loss(
            lens_logits: torch.Tensor,
            target_logits: torch.Tensor,
            gt_labels: torch.Tensor | None,
        ) -> torch.Tensor:
            return combined_loss(
                lens_logits, target_logits, gt_labels, use_gt,
                config.temperature, config.ce_weight,
            )
        return _combined_loss

    else:  # kld (default) — ce_target has no effect
        def _kld_loss(
            lens_logits: torch.Tensor,
            target_logits: torch.Tensor,
            gt_labels: torch.Tensor | None,
        ) -> torch.Tensor:
            return kl_divergence_loss(lens_logits, target_logits, config.temperature)
        return _kld_loss
