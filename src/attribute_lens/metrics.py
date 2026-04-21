"""Insertion and deletion perturbation metrics for attribution evaluation.

The insertion and deletion curves measure how well a patch ranking captures
the model's decision:

- **Insertion** (higher AUC = better): start from a Gaussian-blurred baseline
  and progressively restore patches from the original image, ordered by
  descending importance score.  A good ranking quickly recovers prediction
  confidence.

- **Deletion** (lower AUC = better): start from the original image and
  progressively mask patches with zeros (normalized mean-color fill), ordered
  by descending importance score.  A good ranking causes a rapid drop in
  confidence.
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

    valid_scores = flat[valid_idx]
    order = torch.argsort(valid_scores, descending=True)
    sorted_valid = valid_idx[order].tolist()
    sorted_nan = nan_idx.tolist()

    return sorted_valid + sorted_nan


def _build_perturbed_images(
    base: torch.Tensor,
    patch_src: torch.Tensor,
    rank_order: list[int],
    grid_w: int,
    patch_size: int,
    device: str,
) -> torch.Tensor:
    """Pre-build all N cumulative perturbed images as ``[N, C, H, W]``.

    Image *k* has the first *k+1* patches (``rank_order[0..k]``) taken from
    ``patch_src``; all other pixels come from ``base``.

    Fully vectorized — no Python loop.  A boolean patch-rank mask is
    upsampled to pixel space and passed to ``torch.where``:

    1. ``rank_positions[r, c]`` = step at which patch (r, c) is applied.
    2. ``mask[k, r, c]`` = (rank_positions[r, c] <= k) — True once applied.
    3. Upsample with ``repeat_interleave`` to pixel resolution.
    4. ``imgs = where(mask, patch_src, base)`` — single fused op.
    """
    n_patches = len(rank_order)
    H_p = n_patches // grid_w

    rank_order_t = torch.tensor(rank_order, dtype=torch.long)
    rank_positions = torch.empty(n_patches, dtype=torch.long)
    rank_positions[rank_order_t] = torch.arange(n_patches)
    rank_positions = rank_positions.reshape(H_p, grid_w).to(device)  # [H_p, W_p]

    steps = torch.arange(n_patches, device=device).view(n_patches, 1, 1)  # [N, 1, 1]
    mask = (rank_positions.unsqueeze(0) <= steps)  # [N, H_p, W_p] bool

    mask = mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    mask = mask.unsqueeze(1)  # [N, 1, H, W]

    C = patch_src.shape[1]
    imgs = torch.where(
        mask.expand(-1, C, -1, -1),
        patch_src.to(device).expand(n_patches, -1, -1, -1),
        base.to(device).expand(n_patches, -1, -1, -1),
    )  # [N, C, H, W]
    return imgs


def _run_batched_forward(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    y_hat: int,
    batch_size: int,
    use_fp16: bool = False,
    device: str | None = None,
) -> list[float]:
    """Run *model* on *imgs* in chunks of *batch_size*, return P(y_hat) list.

    If *device* is provided each chunk is moved there before the forward pass,
    allowing *imgs* to live on CPU to avoid pre-allocating large GPU buffers.
    """
    probs: list[float] = []
    device_type = device.split(":")[0] if device else imgs.device.type
    for i in range(0, len(imgs), batch_size):
        chunk = imgs[i: i + batch_size]
        if device:
            chunk = chunk.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_fp16):
            logits = model(chunk)
        probs.extend(F.softmax(logits.float(), dim=-1)[:, y_hat].cpu().tolist())
    return probs


def _run_batched_forward_multi_yhat(
    model: torch.nn.Module,
    imgs: torch.Tensor,       # [N_total, C, H, W]
    y_hats: torch.Tensor,     # [N_total] long — one y_hat per image
    batch_size: int,
) -> list[float]:
    """Like ``_run_batched_forward`` but each image has its own y_hat."""
    probs: list[float] = []
    for i in range(0, len(imgs), batch_size):
        chunk = imgs[i: i + batch_size]
        y_chunk = y_hats[i: i + batch_size]
        logits = model(chunk)
        p = F.softmax(logits, dim=-1)[torch.arange(len(chunk), device=logits.device), y_chunk]
        probs.extend(p.cpu().tolist())
    return probs


# ---------------------------------------------------------------------------
# Single-image curves (for external / one-off use)
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
    """Compute insertion curve and AUC for a single image / single layer.

    Returns:
        ``(fractions, probabilities, auc_value)``
    """
    H_p, W_p = score_map.shape
    n_patches = H_p * W_p
    rank_order = _rank_patches(score_map)

    imgs = _build_perturbed_images(blurred, original, rank_order, W_p, patch_size, device)
    probs = _run_batched_forward(model, imgs, y_hat, eval_batch_size)
    del imgs

    x = np.linspace(1 / n_patches, 1.0, n_patches)
    y = np.array(probs)
    return x, y, float(sklearn_auc(x, y))


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
    """Compute deletion curve and AUC for a single image / single layer.

    Returns:
        ``(fractions, probabilities, auc_value)``
    """
    H_p, W_p = score_map.shape
    n_patches = H_p * W_p
    rank_order = _rank_patches(score_map)

    imgs = _build_perturbed_images(original, torch.zeros_like(original), rank_order, W_p, patch_size, device)
    probs = _run_batched_forward(model, imgs, y_hat, eval_batch_size)
    del imgs

    x = np.linspace(1 / n_patches, 1.0, n_patches)
    y = np.array(probs)
    return x, y, float(sklearn_auc(x, y))


@torch.no_grad()
def insertion_deletion_curves(
    model: torch.nn.Module,
    original: torch.Tensor,
    blurred: torch.Tensor,
    score_map: torch.Tensor,
    y_hat: int,
    patch_size: int,
    device: str = "cpu",
    eval_batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
    """Compute both curves for a single image / single layer.

    Returns:
        ``(ins_x, ins_y, ins_auc, del_x, del_y, del_auc)``
    """
    H_p, W_p = score_map.shape
    n_patches = H_p * W_p
    rank_order = _rank_patches(score_map)
    x = np.linspace(1 / n_patches, 1.0, n_patches)

    ins_imgs = _build_perturbed_images(blurred, original, rank_order, W_p, patch_size, device)
    ins_probs = _run_batched_forward(model, ins_imgs, y_hat, eval_batch_size)
    del ins_imgs

    del_imgs = _build_perturbed_images(original, torch.zeros_like(original), rank_order, W_p, patch_size, device)
    del_probs = _run_batched_forward(model, del_imgs, y_hat, eval_batch_size)
    del del_imgs

    ins_y = np.array(ins_probs)
    del_y = np.array(del_probs)
    return (
        x, ins_y, float(sklearn_auc(x, ins_y)),
        x, del_y, float(sklearn_auc(x, del_y)),
    )


# ---------------------------------------------------------------------------
# All-layers combined: two forward passes per image
# ---------------------------------------------------------------------------

@torch.no_grad()
def insertion_deletion_curves_all_layers(
    model: torch.nn.Module,
    original: torch.Tensor,                        # [1, C, H, W]
    blurred: torch.Tensor,                         # [1, C, H, W]
    score_maps_by_layer: dict[int, torch.Tensor],  # {layer_idx: [H_p, W_p]}
    y_hat: int,
    patch_size: int,
    device: str = "cpu",
    eval_batch_size: int = 128,
    use_fp16: bool = False,
) -> dict[int, tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]]:
    """Compute insertion+deletion curves for ALL layers in two forward passes.

    All L layers' perturbed sequences are concatenated into a single
    ``[L * n_patches, C, H, W]`` tensor.  The model is run once over this
    tensor for insertion and once for deletion — eliminating the per-layer
    forward-pass loop entirely.

    Peak GPU memory: ``[L * n_patches, C, H, W]`` float32 at a time.
    For ViT-L/14 (224×224, 256 patches, 24 layers): 24 × 256 × ~0.6 MB ≈ 3.7 GB.

    Args:
        original:            ``[1, C, H, W]`` image tensor on any device.
        blurred:             ``[1, C, H, W]`` blurred baseline.
        score_maps_by_layer: Per-layer attribution maps ``{layer: [H_p, W_p]}``.
        y_hat:               Predicted class index.
        patch_size:          Pixel side length of each patch.
        device:              Target device for model inference.
        eval_batch_size:     Forward-pass chunk size (controls VRAM usage).

    Returns:
        ``{layer_idx: (ins_x, ins_y, ins_auc, del_x, del_y, del_auc)}``
    """
    layers = sorted(score_maps_by_layer.keys())
    L = len(layers)
    H_p, W_p = next(iter(score_maps_by_layer.values())).shape
    n_patches = H_p * W_p
    C, H, W = original.shape[1], original.shape[2], original.shape[3]

    rank_orders = {l: _rank_patches(score_maps_by_layer[l]) for l in layers}

    # ---- Pass 1: insertion (blurred → original) ----
    all_ins = torch.empty(L * n_patches, C, H, W, device=device)
    for k, l in enumerate(layers):
        all_ins[k * n_patches: (k + 1) * n_patches] = _build_perturbed_images(
            blurred, original, rank_orders[l], W_p, patch_size, device
        )
    ins_probs = _run_batched_forward(model, all_ins, y_hat, eval_batch_size, use_fp16)
    del all_ins

    # ---- Pass 2: deletion (original → zeros) ----
    del_fill = torch.zeros_like(original)
    all_del = torch.empty(L * n_patches, C, H, W, device=device)
    for k, l in enumerate(layers):
        all_del[k * n_patches: (k + 1) * n_patches] = _build_perturbed_images(
            original, del_fill, rank_orders[l], W_p, patch_size, device
        )
    del_probs = _run_batched_forward(model, all_del, y_hat, eval_batch_size, use_fp16)
    del all_del

    x = np.linspace(1 / n_patches, 1.0, n_patches)
    results: dict[int, tuple] = {}
    for k, l in enumerate(layers):
        ins_y = np.array(ins_probs[k * n_patches: (k + 1) * n_patches])
        del_y = np.array(del_probs[k * n_patches: (k + 1) * n_patches])
        results[l] = (
            x, ins_y, float(sklearn_auc(x, ins_y)),
            x, del_y, float(sklearn_auc(x, del_y)),
        )
    return results


# ---------------------------------------------------------------------------
# Legacy: batched across multiple images (one layer at a time)
# ---------------------------------------------------------------------------

@torch.no_grad()
def insertion_deletion_curves_batch(
    model: torch.nn.Module,
    originals: torch.Tensor,           # [N, C, H, W]
    blurreds: torch.Tensor,            # [N, C, H, W]
    score_maps: list[torch.Tensor],    # N × [H_p, W_p]
    y_hats: list[int],                 # length N
    patch_size: int,
    device: str = "cpu",
    eval_batch_size: int = 128,
) -> list[tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]]:
    """Compute insertion+deletion for N images at a single layer.

    All N images' perturbed states are stacked into ``[N * n_patches, C, H, W]``
    and processed together.  Kept for external / single-layer use.

    Returns:
        List of N tuples ``(ins_x, ins_y, ins_auc, del_x, del_y, del_auc)``.
    """
    N = originals.shape[0]
    H_p, W_p = score_maps[0].shape
    n_patches = H_p * W_p

    rank_orders = [_rank_patches(sm) for sm in score_maps]

    C, H, W = originals.shape[1], originals.shape[2], originals.shape[3]
    all_ins = torch.empty(N * n_patches, C, H, W, device=device)
    for i in range(N):
        all_ins[i * n_patches: (i + 1) * n_patches] = _build_perturbed_images(
            blurreds[i: i + 1], originals[i: i + 1], rank_orders[i], W_p, patch_size, device
        )

    y_hats_t = torch.tensor(
        [yh for yh in y_hats for _ in range(n_patches)],
        dtype=torch.long, device=device,
    )
    ins_probs = _run_batched_forward_multi_yhat(model, all_ins, y_hats_t, eval_batch_size)
    del all_ins

    all_del = torch.empty(N * n_patches, C, H, W, device=device)
    for i in range(N):
        all_del[i * n_patches: (i + 1) * n_patches] = _build_perturbed_images(
            originals[i: i + 1], torch.zeros_like(originals[i: i + 1]), rank_orders[i], W_p, patch_size, device
        )

    y_hats_t2 = torch.tensor(
        [yh for yh in y_hats for _ in range(n_patches)],
        dtype=torch.long, device=device,
    )
    del_probs = _run_batched_forward_multi_yhat(model, all_del, y_hats_t2, eval_batch_size)
    del all_del

    x = np.linspace(1 / n_patches, 1.0, n_patches)
    results = []
    for i in range(N):
        ins_y = np.array(ins_probs[i * n_patches: (i + 1) * n_patches])
        del_y = np.array(del_probs[i * n_patches: (i + 1) * n_patches])
        results.append((
            x, ins_y, float(sklearn_auc(x, ins_y)),
            x, del_y, float(sklearn_auc(x, del_y)),
        ))
    return results
