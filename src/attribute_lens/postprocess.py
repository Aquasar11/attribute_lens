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
# Gaussian kernel helper
# ---------------------------------------------------------------------------

def _make_gaussian_kernel(
    size: int,
    std: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return an unnormalized Gaussian kernel of shape ``[size, size]``.

    The kernel is intentionally *not* normalized — per-position normalization
    is done dynamically via a separate denominator convolution, which correctly
    handles image boundaries (corners/edges) where some neighbors are absent.

    Args:
        size: Kernel side length (must be odd, ≥ 1).
        std: Standard deviation of the Gaussian.
        device: Target device.
        dtype: Target dtype.

    Returns:
        ``Tensor[size, size]`` of non-negative weights.
    """
    if size == 1:
        return torch.ones(1, 1, device=device, dtype=dtype)
    half = size // 2
    coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2.0 * std ** 2))
    return kernel  # [size, size], unnormalized


# ---------------------------------------------------------------------------
# Rank helper for rank_sum aggregation
# ---------------------------------------------------------------------------

def _rank_map(
    score_map: torch.Tensor,
    nan_mask: torch.Tensor,
) -> torch.Tensor:
    """Return a rank map ``[H, W]`` (rank 1 = highest score).

    Valid positions are ranked 1..N_valid in descending score order.
    NaN positions receive rank N_total + 1.

    Args:
        score_map: ``Tensor[H, W]``, may contain NaN.
        nan_mask:  ``BoolTensor[H, W]``, True where score_map is NaN.

    Returns:
        ``FloatTensor[H, W]`` of ranks (float for arithmetic convenience).
    """
    H, W = score_map.shape
    n_total = H * W
    flat = score_map.flatten()
    nan_flat = nan_mask.flatten()

    valid_idx = (~nan_flat).nonzero(as_tuple=True)[0]
    valid_scores = flat[valid_idx]
    order = torch.argsort(valid_scores, descending=True)  # best → first

    ranks = torch.full((n_total,), float(n_total + 1), device=score_map.device)
    ranks[valid_idx[order]] = torch.arange(
        1, len(valid_idx) + 1, dtype=torch.float32, device=score_map.device
    )
    return ranks.view(H, W)


# ---------------------------------------------------------------------------
# Neighbor averaging on embeddings
# ---------------------------------------------------------------------------

def neighbor_avg_embeddings(
    patch_states: dict[int, torch.Tensor],
    kernel_sizes: list[int],
    gaussian_std: float,
    aggregation_weights: list[float],
) -> dict[int, torch.Tensor]:
    """Return patch_states where each position is a multi-kernel Gaussian average.

    For each kernel size in ``kernel_sizes``, a Gaussian-weighted average of the
    N×N spatial neighbourhood is computed.  The per-kernel results are then
    combined by normalized weighted average (rank-sum is not defined for vector
    embeddings).

    Boundary positions (corners/edges) are handled correctly: the denominator
    convolution sums only the Gaussian weights of in-bounds neighbours, so the
    result is always a valid weighted average of the available patches.

    Uses grouped conv2d for efficient vectorized computation over d_model.

    Args:
        patch_states: ``{layer_idx: Tensor[B, H, W, d_model]}``
        kernel_sizes: List of odd kernel side lengths, e.g. ``[1, 3, 5]``.
        gaussian_std: Standard deviation shared by all Gaussian kernels.
        aggregation_weights: One weight per kernel size (will be normalized to
            sum=1 internally).  Length must equal ``len(kernel_sizes)``.

    Returns:
        New dict with same keys and shapes; values are smoothed embeddings.
    """
    # Normalize aggregation weights
    w_total = sum(aggregation_weights)
    w_norm = [w / w_total for w in aggregation_weights]

    result: dict[int, torch.Tensor] = {}

    for layer, states in patch_states.items():
        B, H, W, d = states.shape
        device = states.device
        dtype = states.dtype

        # [B, d, H, W]
        x = states.permute(0, 3, 1, 2).contiguous()

        # ones map for denom computation (shared across kernels)
        ones = torch.ones(B, 1, H, W, device=device, dtype=dtype)

        accumulated: torch.Tensor | None = None

        for size, w in zip(kernel_sizes, w_norm):
            pad = size // 2
            kernel = _make_gaussian_kernel(size, gaussian_std, device, dtype)  # [size, size]

            # Grouped conv2d: one filter per d_model channel
            kernel_4d = kernel.unsqueeze(0).unsqueeze(0).expand(d, 1, size, size).contiguous()
            numerator = F.conv2d(x, kernel_4d, padding=pad, groups=d)   # [B, d, H, W]

            # Per-position denominator: sum of in-bounds Gaussian weights
            k1 = kernel.unsqueeze(0).unsqueeze(0)                        # [1, 1, size, size]
            denom = F.conv2d(ones, k1, padding=pad)                      # [B, 1, H, W]

            smoothed = numerator / (denom + _EPS)                        # [B, d, H, W]

            contrib = smoothed * w
            accumulated = contrib if accumulated is None else accumulated + contrib

        # [B, d, H, W] → [B, H, W, d]
        result[layer] = accumulated.permute(0, 2, 3, 1)  # type: ignore[union-attr]

    return result


# ---------------------------------------------------------------------------
# Neighbor averaging on scores
# ---------------------------------------------------------------------------

def neighbor_avg_scores(
    score_map: torch.Tensor,
    kernel_sizes: list[int],
    gaussian_std: float,
    aggregation: str,
    aggregation_weights: list[float],
) -> torch.Tensor:
    """Return a score map where each position is a multi-kernel Gaussian aggregate.

    NaN positions (e.g. border patches from PatchLensScorer) are excluded from
    neighbour contributions — they contribute weight 0 to their neighbours.
    Positions that were NaN in the input remain NaN in the output.

    Boundary positions (corners/edges) are handled correctly: the denominator
    convolution sums only the Gaussian weights of in-bounds valid neighbours,
    so the result is always a properly normalized average.

    Args:
        score_map: ``Tensor[H, W]``, may contain NaN.
        kernel_sizes: List of odd kernel side lengths, e.g. ``[1, 3, 5]``.
        gaussian_std: Standard deviation shared by all Gaussian kernels.
        aggregation: ``"weighted_avg"`` — normalized weighted average of
            per-kernel smoothed maps; ``"rank_sum"`` — for each kernel rank
            positions by smoothed score (descending), sum ranks, re-rank
            (tie-break by 1×1 kernel ordering).
        aggregation_weights: One weight per kernel size for ``"weighted_avg"``
            (normalized internally).  Ignored for ``"rank_sum"``.

    Returns:
        ``Tensor[H, W]`` with NaN positions preserved.
    """
    H, W = score_map.shape
    device = score_map.device
    dtype = score_map.dtype

    nan_mask = torch.isnan(score_map)
    filled = score_map.nan_to_num(0.0)
    valid = (~nan_mask).to(dtype)  # 1.0 where valid, 0.0 where NaN

    filled_v = (filled * valid).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    valid_v = valid.unsqueeze(0).unsqueeze(0)              # [1, 1, H, W]

    def _smooth_one(size: int) -> torch.Tensor:
        """Gaussian-smooth score_map with a kernel of given size."""
        pad = size // 2
        kernel = _make_gaussian_kernel(size, gaussian_std, device, dtype)
        k4 = kernel.unsqueeze(0).unsqueeze(0)                              # [1,1,s,s]
        num = F.conv2d(filled_v, k4, padding=pad)[0, 0]                   # [H, W]
        den = F.conv2d(valid_v, k4, padding=pad)[0, 0]                    # [H, W]
        out = num / (den + _EPS)
        out[nan_mask] = float("nan")
        return out

    if aggregation == "weighted_avg":
        w_total = sum(aggregation_weights)
        w_norm = [w / w_total for w in aggregation_weights]

        accumulated: torch.Tensor | None = None
        for size, w in zip(kernel_sizes, w_norm):
            smoothed = _smooth_one(size)
            # Treat NaN as 0 for accumulation (restored at the end)
            contrib = smoothed.nan_to_num(0.0) * w
            accumulated = contrib if accumulated is None else accumulated + contrib

        result = accumulated  # type: ignore[assignment]
        result[nan_mask] = float("nan")
        return result

    elif aggregation == "rank_sum":
        # Compute per-kernel smoothed maps and rank them
        total_rank = torch.zeros(H, W, device=device, dtype=torch.float32)
        for size in kernel_sizes:
            smoothed = _smooth_one(size)
            total_rank += _rank_map(smoothed, nan_mask)

        # Tie-break: rank from the 1×1 kernel (= original score, no smoothing)
        tiebreak_rank = _rank_map(score_map, nan_mask)

        # Encode final ordering as a score (higher = more important)
        # tiebreak contribution < 1 so it only resolves ties in total_rank
        n = H * W
        final_score = -(total_rank + tiebreak_rank / (n + 1))
        final_score[nan_mask] = float("nan")
        return final_score

    else:
        raise ValueError(
            f"Unknown aggregation '{aggregation}'. Use 'weighted_avg' or 'rank_sum'."
        )


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
