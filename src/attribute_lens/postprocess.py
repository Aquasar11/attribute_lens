"""Post-processing utilities for score maps and patch embeddings.

Three optional modes that can be combined:
- Neighbor embedding average: smooth patch embeddings before scoring
- Neighbor score average: smooth score maps after scoring
- Layer average: average score maps across a configurable layer range
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


_EPS = 1e-8


# ---------------------------------------------------------------------------
# Neighbor averaging on embeddings
# ---------------------------------------------------------------------------

def neighbor_avg_embeddings(
    patch_states: dict[int, torch.Tensor],
    size: int = 3,
    center_weight: int = 2,
) -> dict[int, torch.Tensor]:
    """Return patch_states where each position is a weighted neighborhood average.

    For each layer tensor ``[1, H, W, d_model]``, replace each patch embedding
    with a weighted average of its N×N spatial neighborhood:
    - center patch: weight ``center_weight``
    - each neighbor:  weight 1
    - boundary positions: zero-padded (equivalent to treating out-of-grid as zeros)

    Uses grouped conv2d for efficient vectorized computation over d_model.

    Args:
        patch_states: ``{layer_idx: Tensor[1, H, W, d_model]}``
        size: Neighborhood window side length N (must be odd, ≥ 1).
        center_weight: Weight given to the center patch (neighbors get weight 1).

    Returns:
        New dict with same keys and shapes; values are smoothed embeddings.
    """
    if size == 1:
        return {l: t.clone() for l, t in patch_states.items()}

    pad = size // 2
    result: dict[int, torch.Tensor] = {}

    for layer, states in patch_states.items():
        # states: [1, H, W, d]
        _, H, W, d = states.shape
        device = states.device
        dtype = states.dtype

        # Build normalized kernel: uniform 1s with center boosted
        kernel = torch.ones(size, size, device=device, dtype=dtype)
        kernel[pad, pad] = float(center_weight)
        kernel = kernel / kernel.sum()           # [size, size]
        # Expand to [d, 1, size, size] for groups=d convolution
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0).expand(d, 1, size, size)

        # Reshape: [1, H, W, d] → [1, d, H, W]  (batch=1, channels=d)
        x = states[0].permute(2, 0, 1).unsqueeze(0)  # [1, d, H, W]

        smoothed = F.conv2d(x, kernel_4d, padding=pad, groups=d)  # [1, d, H, W]

        # Reshape back: [1, d, H, W] → [1, H, W, d]
        out = smoothed.squeeze(0).permute(1, 2, 0).unsqueeze(0)   # [1, H, W, d]
        result[layer] = out

    return result


# ---------------------------------------------------------------------------
# Neighbor averaging on scores
# ---------------------------------------------------------------------------

def neighbor_avg_scores(
    score_map: torch.Tensor,
    size: int = 3,
    center_weight: int = 2,
) -> torch.Tensor:
    """Return a score map where each position is a weighted neighborhood average.

    NaN positions (e.g. border patches from PatchLensScorer) are excluded from
    neighbor contributions — they contribute weight 0 to their neighbors.
    Positions that were NaN in the input remain NaN in the output.

    Args:
        score_map: ``Tensor[H, W]``, may contain NaN.
        size: Window side length N (must be odd, ≥ 1).
        center_weight: Weight given to the center position (neighbors get 1).

    Returns:
        ``Tensor[H, W]`` with NaN positions preserved; others are averages.
    """
    if size == 1:
        return score_map.clone()

    H, W = score_map.shape
    device = score_map.device
    dtype = score_map.dtype
    pad = size // 2

    nan_mask = torch.isnan(score_map)           # [H, W] bool
    filled = score_map.nan_to_num(0.0)          # [H, W], NaN → 0

    # valid_mask float: 1.0 where not NaN, 0.0 where NaN
    valid = (~nan_mask).to(dtype)               # [H, W]

    # Unnormalized kernel (int weights, applied as float)
    kernel = torch.ones(size, size, device=device, dtype=dtype)
    kernel[pad, pad] = float(center_weight)
    # Shape expected by conv2d: [out_ch, in_ch, kH, kW]
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, size, size]

    # Compute weighted sum of valid filled values
    filled_v = (filled * valid).unsqueeze(0).unsqueeze(0)    # [1, 1, H, W]
    numerator = F.conv2d(filled_v, kernel_4d, padding=pad)[0, 0]  # [H, W]

    # Compute effective weight sum (excludes NaN neighbors)
    valid_v = valid.unsqueeze(0).unsqueeze(0)                 # [1, 1, H, W]
    # Weight of center = center_weight (already in kernel); neighbors = 1
    # We need: for each position, sum of kernel[i,j] for valid neighbors
    # This is exactly conv2d(valid * kernel_weights_map) but since kernel is
    # spatially uniform except center, conv2d(valid, kernel) gives the right answer
    denom = F.conv2d(valid_v, kernel_4d, padding=pad)[0, 0]   # [H, W]

    result = numerator / (denom + _EPS)

    # Restore NaN mask
    result[nan_mask] = float("nan")
    return result


# ---------------------------------------------------------------------------
# Layer-wise score map averaging
# ---------------------------------------------------------------------------

def layer_avg_score_maps(
    score_maps: dict[int, torch.Tensor],
    min_layer: int | None,
    max_layer: int | None,
    weights: dict[int, float] | None = None,
) -> torch.Tensor:
    """Compute a weighted average of score maps across layers in [min_layer, max_layer].

    Args:
        score_maps: ``{layer_idx: Tensor[H, W]}``
        min_layer: Lower bound (inclusive). None = use first available.
        max_layer: Upper bound (inclusive). None = use last available.
        weights: ``{layer_idx: float}`` pre-computed per-layer weights.
            None or all-zero → falls back to uniform weighting.

    Returns:
        ``Tensor[H, W]``: weighted average map.

    Raises:
        ValueError: if no layers fall within [min_layer, max_layer].
    """
    available = sorted(score_maps.keys())
    lo = min_layer if min_layer is not None else available[0]
    hi = max_layer if max_layer is not None else available[-1]
    layers_in_range = [l for l in available if lo <= l <= hi]

    if not layers_in_range:
        raise ValueError(
            f"No score maps in layer range [{lo}, {hi}]. Available: {available}"
        )

    if weights is None:
        w_list = [1.0] * len(layers_in_range)
    else:
        w_list = [float(weights.get(l, 0.0)) for l in layers_in_range]

    total_w = sum(w_list)
    if total_w == 0.0:
        # All weights were zero → fall back to uniform
        w_list = [1.0] * len(layers_in_range)
        total_w = float(len(layers_in_range))

    stacked = torch.stack([score_maps[l] for l in layers_in_range], dim=0)  # [L, H, W]
    w_tensor = torch.tensor(w_list, dtype=stacked.dtype, device=stacked.device).view(-1, 1, 1)
    return (stacked * w_tensor).sum(dim=0) / total_w


# ---------------------------------------------------------------------------
# Load pre-computed layer weights
# ---------------------------------------------------------------------------

def load_layer_weights(path: str) -> dict[int, float]:
    """Load layer weights from a .pt file saved by the fg_bg_layer_weights notebook.

    Expected format::

        {"layers": [int, ...], "weights": [float, ...], "metadata": {...}}

    Args:
        path: Path to the ``.pt`` file. Empty string returns empty dict.

    Returns:
        ``{layer_idx: weight}``
    """
    if not path:
        return {}
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {int(l): float(w) for l, w in zip(data["layers"], data["weights"])}
