"""Loss functions for direct_lens training.

Self-contained copy — intentionally not imported from tuned_lens so this module
stays readable and changeable in isolation.

The target is always the backbone's own output distribution (soft labels).
No ground truth labels are used.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import TrainingConfig


def kld_loss(lens_logits: Tensor, target_logits: Tensor, temperature: float = 1.0) -> Tensor:
    """KL(target || lens): how well the lens approximates the backbone distribution.

    Both inputs are raw logits. Temperature-scaled softmax applied internally.
    Uses log_target=True for numerical stability.
    """
    target_log_probs = F.log_softmax(target_logits / temperature, dim=-1)
    lens_log_probs = F.log_softmax(lens_logits / temperature, dim=-1)
    return F.kl_div(lens_log_probs, target_log_probs, log_target=True, reduction="batchmean")


def ce_loss(lens_logits: Tensor, target_logits: Tensor) -> Tensor:
    """CE between lens predictions and the backbone's argmax decision."""
    return F.cross_entropy(lens_logits, target_logits.argmax(dim=-1))


def best_ce_loss(lens_logits: Tensor, target_logits: Tensor, best_fraction: float = 0.2) -> Tensor:
    """CE averaged over the best-performing fraction of samples in the batch.

    Computes per-sample CE, selects the `best_fraction` proportion with the
    lowest loss (best-predicted tokens), and returns their mean.
    """
    per_sample = F.cross_entropy(lens_logits, target_logits.argmax(dim=-1), reduction="none")
    k = max(1, int(len(per_sample) * best_fraction))
    best_losses, _ = torch.topk(per_sample, k, largest=False, sorted=False)
    return best_losses.mean()


def combined_loss(
    lens_logits: Tensor,
    target_logits: Tensor,
    temperature: float = 1.0,
    ce_weight: float = 0.1,
) -> Tensor:
    """KLD + ce_weight * CE."""
    return kld_loss(lens_logits, target_logits, temperature) + ce_weight * ce_loss(lens_logits, target_logits)


def get_loss_fn(config: TrainingConfig) -> Callable[[Tensor, Tensor], Tensor]:
    """Return a loss function (lens_logits, target_logits) → scalar."""
    if config.loss_type == "ce":
        return ce_loss

    if config.loss_type == "combined":
        t, w = config.temperature, config.ce_weight

        def _combined(logits: Tensor, targets: Tensor) -> Tensor:
            return combined_loss(logits, targets, t, w)

        return _combined

    if config.loss_type == "best_ce":
        frac = config.best_fraction

        def _best_ce(logits: Tensor, targets: Tensor) -> Tensor:
            return best_ce_loss(logits, targets, frac)

        return _best_ce

    # default: kld
    t = config.temperature

    def _kld(logits: Tensor, targets: Tensor) -> Tensor:
        return kld_loss(logits, targets, t)

    return _kld
