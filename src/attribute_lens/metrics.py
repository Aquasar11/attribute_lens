"""Insertion and deletion perturbation metrics for attribution evaluation.

The insertion and deletion curves measure how well a patch ranking captures
the model's decision:

- **Insertion** (higher AUC = better): start from a Gaussian-blurred baseline
  and progressively restore patches from the original image, ordered by
  descending importance score.  A good ranking quickly recovers prediction
  confidence.

- **Deletion** (lower AUC = better): start from the original image and
  progressively mask patches with the blurred baseline, ordered by descending
  importance score.  A good ranking causes a rapid drop in confidence.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc as sklearn_auc
from torchvision.transforms.functional import gaussian_blur


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_gaussian_blur(
    image: torch.Tensor,
    kernel_size: int = 55,
    sigma: float = 10.0,
) -> torch.Tensor:
    """Apply Gaussian blur to a pre-processed image tensor.

    Args:
        image: Shape ``[1, C, H, W]``, already normalised (any mean/std).
        kernel_size: Blur kernel size (must be odd).
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        Blurred tensor of the same shape, on the same device.
    """
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=sigma)
    return blurred


def _patch_coords(patch_idx: int, grid_w: int, patch_size: int) -> tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1) pixel coordinates for a flat patch index."""
    r = patch_idx // grid_w
    c = patch_idx % grid_w
    y0, y1 = r * patch_size, (r + 1) * patch_size
    x0, x1 = c * patch_size, (c + 1) * patch_size
    return y0, y1, x0, x1


def _rank_patches(score_map: torch.Tensor) -> list[int]:
    """Return flat patch indices sorted by descending score.

    NaN scores are placed last (treated as least important).
    """
    flat = score_map.flatten()
    nan_mask = torch.isnan(flat)

    valid_idx = (~nan_mask).nonzero(as_tuple=True)[0]
    nan_idx = nan_mask.nonzero(as_tuple=True)[0]

    # Sort valid indices by descending score
    valid_scores = flat[valid_idx]
    order = torch.argsort(valid_scores, descending=True)
    sorted_valid = valid_idx[order].tolist()
    sorted_nan = nan_idx.tolist()

    return sorted_valid + sorted_nan


# ---------------------------------------------------------------------------
# Insertion curve
# ---------------------------------------------------------------------------

@torch.no_grad()
def insertion_curve(
    model: torch.nn.Module,
    original: torch.Tensor,
    blurred: torch.Tensor,
    score_map: torch.Tensor,
    y_hat: int,
    patch_size: int,
    device: str = "cpu",
    eval_batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute insertion curve and AUC with batched model inference.

    Starts from ``blurred`` and sequentially restores the ``original`` image
    patch by patch, ordered by descending score.  Perturbed images are
    accumulated into a batch of size ``eval_batch_size`` before a single
    forward pass, reducing GPU launches from ``n_patches`` to
    ``ceil(n_patches / eval_batch_size)``.

    Args:
        model: The frozen backbone (``VisionModelWrapper.model``).
        original: Pre-processed original image ``[1, C, H, W]``.
        blurred: Gaussian-blurred version ``[1, C, H, W]`` (same normalisation).
        score_map: ``[H_patches, W_patches]`` patch importance scores.  NaN =
            treat as least important.
        y_hat: Class index to track.
        patch_size: Pixel side length of each patch (e.g. 14 for ViT-L/14).
        device: Device for model inference.
        eval_batch_size: Number of perturbed images per forward pass.

    Returns:
        ``(fractions, probabilities, auc_value)`` — fraction of patches inserted,
        model probability at that step, and the area under the curve.
    """
    H_p, W_p = score_map.shape
    n_patches = H_p * W_p
    rank_order = _rank_patches(score_map)

    current = blurred.clone().to(device)
    orig_dev = original.to(device)

    fractions: list[float] = []
    probabilities: list[float] = []

    step = 0
    while step < n_patches:
        batch_imgs: list[torch.Tensor] = []
        batch_steps: list[int] = []

        # Apply patches sequentially and snapshot each intermediate state
        for _ in range(eval_batch_size):
            if step >= n_patches:
                break
            pidx = rank_order[step]
            y0, y1, x0, x1 = _patch_coords(pidx, W_p, patch_size)
            current[:, :, y0:y1, x0:x1] = orig_dev[:, :, y0:y1, x0:x1]
            batch_imgs.append(current.clone())
            batch_steps.append(step)
            step += 1

        # Single forward pass for the whole chunk
        batch_tensor = torch.cat(batch_imgs, dim=0)        # [B, C, H, W]
        logits = model(batch_tensor)
        probs = F.softmax(logits, dim=-1)[:, y_hat].cpu().tolist()

        for s, prob in zip(batch_steps, probs):
            fractions.append((s + 1) / n_patches)
            probabilities.append(float(prob))

    x = np.array(fractions)
    y = np.array(probabilities)
    return x, y, float(sklearn_auc(x, y))


# ---------------------------------------------------------------------------
# Deletion curve
# ---------------------------------------------------------------------------

@torch.no_grad()
def deletion_curve(
    model: torch.nn.Module,
    original: torch.Tensor,
    blurred: torch.Tensor,
    score_map: torch.Tensor,
    y_hat: int,
    patch_size: int,
    device: str = "cpu",
    eval_batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute deletion curve and AUC with batched model inference.

    Starts from ``original`` and sequentially replaces patches with the
    ``blurred`` baseline, ordered by descending score.  Perturbed images are
    accumulated into a batch of size ``eval_batch_size`` before a single
    forward pass.

    Args:
        model: The frozen backbone.
        original: Pre-processed original image ``[1, C, H, W]``.
        blurred: Gaussian-blurred version (same normalisation).
        score_map: ``[H_patches, W_patches]`` patch importance scores.
        y_hat: Class index to track.
        patch_size: Pixel side length of each patch.
        device: Device for model inference.
        eval_batch_size: Number of perturbed images per forward pass.

    Returns:
        ``(fractions, probabilities, auc_value)``.
    """
    H_p, W_p = score_map.shape
    n_patches = H_p * W_p
    rank_order = _rank_patches(score_map)

    current = original.clone().to(device)
    blur_dev = blurred.to(device)

    fractions: list[float] = []
    probabilities: list[float] = []

    step = 0
    while step < n_patches:
        batch_imgs: list[torch.Tensor] = []
        batch_steps: list[int] = []

        for _ in range(eval_batch_size):
            if step >= n_patches:
                break
            pidx = rank_order[step]
            y0, y1, x0, x1 = _patch_coords(pidx, W_p, patch_size)
            current[:, :, y0:y1, x0:x1] = blur_dev[:, :, y0:y1, x0:x1]
            batch_imgs.append(current.clone())
            batch_steps.append(step)
            step += 1

        batch_tensor = torch.cat(batch_imgs, dim=0)        # [B, C, H, W]
        logits = model(batch_tensor)
        probs = F.softmax(logits, dim=-1)[:, y_hat].cpu().tolist()

        for s, prob in zip(batch_steps, probs):
            fractions.append((s + 1) / n_patches)
            probabilities.append(float(prob))

    x = np.array(fractions)
    y = np.array(probabilities)
    return x, y, float(sklearn_auc(x, y))
