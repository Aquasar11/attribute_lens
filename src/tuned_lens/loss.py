"""Loss functions for tuned lens training."""

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
    labels: torch.Tensor,
) -> torch.Tensor:
    """Standard cross-entropy between lens predictions and ground truth labels."""
    return F.cross_entropy(lens_logits, labels)


def combined_loss(
    lens_logits: torch.Tensor,
    target_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    ce_weight: float = 0.1,
) -> torch.Tensor:
    """KLD + ce_weight * CE."""
    kld = kl_divergence_loss(lens_logits, target_logits, temperature)
    ce = cross_entropy_loss(lens_logits, labels)
    return kld + ce_weight * ce


LossFn = Callable[..., torch.Tensor]


def get_loss_fn(config: TrainingConfig) -> LossFn:
    """Factory that returns the appropriate loss function based on config.

    Returns a callable with signature:
        loss_fn(lens_logits, target_logits, labels) -> Tensor
    """
    if config.loss_type == "ce":
        def _ce_loss(lens_logits: torch.Tensor, target_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return cross_entropy_loss(lens_logits, labels)
        return _ce_loss

    elif config.loss_type == "combined":
        def _combined_loss(lens_logits: torch.Tensor, target_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return combined_loss(lens_logits, target_logits, labels, config.temperature, config.ce_weight)
        return _combined_loss

    else:  # kld (default)
        def _kld_loss(lens_logits: torch.Tensor, target_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return kl_divergence_loss(lens_logits, target_logits, config.temperature)
        return _kld_loss
