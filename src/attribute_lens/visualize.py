"""Visualization utilities for attribution heatmaps and perturbation curves."""

from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_map_to_rgba(
    score_map_np: np.ndarray,
    colormap: str,
    alpha: float,
) -> np.ndarray:
    """Convert a [H, W] score map to an RGBA image [H, W, 4].

    Alpha is **proportional to the normalised score**, so low-scoring patches
    fade to transparent and only high-scoring patches are prominently coloured.
    NaN positions are fully transparent.

    Args:
        score_map_np: Float array, NaN allowed.
        colormap: Matplotlib colormap name.
        alpha: Maximum opacity applied to the highest-scoring patch.
    """
    cmap = plt.get_cmap(colormap)
    nan_mask = np.isnan(score_map_np)

    # Normalise to [0, 1], ignoring NaN
    valid = score_map_np[~nan_mask]
    if valid.size > 0:
        vmin, vmax = float(valid.min()), float(valid.max())
        if vmax > vmin:
            normalised = (score_map_np - vmin) / (vmax - vmin)
        else:
            normalised = np.zeros_like(score_map_np)
    else:
        normalised = np.zeros_like(score_map_np)

    rgba = cmap(normalised)                    # [H, W, 4], colours from colormap
    rgba[..., 3] = normalised * alpha          # alpha ∝ score: low score → transparent
    rgba[nan_mask, 3] = 0.0                    # NaN (border) → fully transparent
    return (rgba * 255).astype(np.uint8)


def _upsample_score_map(
    score_map: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Bilinearly upsample ``score_map`` from patch-grid to image resolution.

    NaN values are filled with the nearest non-NaN value before upsampling
    and restored as NaN where the original was NaN (expanded to patch blocks).
    """
    nan_mask = np.isnan(score_map)
    filled = score_map.copy()
    if nan_mask.any():
        # Simple fill: replace NaN with 0.0 for interpolation
        filled[nan_mask] = 0.0

    t = torch.tensor(filled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    up = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    result = up[0, 0].numpy()

    # Mark upsampled NaN regions
    if nan_mask.any():
        nm_t = torch.tensor(nan_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        nm_up = F.interpolate(nm_t, size=(target_h, target_w), mode="nearest")
        result[nm_up[0, 0].numpy() > 0.5] = float("nan")

    return result


# ---------------------------------------------------------------------------
# Single-image heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    original_pil: Image.Image,
    score_map: np.ndarray,
    output_path: str,
    title: str = "",
    colormap: str = "hot",
    alpha: float = 0.6,
    dpi: int = 150,
) -> None:
    """Overlay a patch score map on the original image and save as PNG.

    Args:
        original_pil: Original (un-transformed) PIL image.
        score_map: ``[H_patches, W_patches]`` float array, NaN allowed.
        output_path: Destination path for the PNG file.
        title: Optional figure title.
        colormap: Matplotlib colormap name.
        alpha: Opacity of the heatmap overlay (0 = transparent, 1 = opaque).
        dpi: Figure DPI.
    """
    W_img, H_img = original_pil.size
    up = _upsample_score_map(score_map, H_img, W_img)
    overlay = _score_map_to_rgba(up, colormap, alpha)  # [H, W, 4] RGBA

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)
    ax.imshow(original_pil)
    ax.imshow(overlay)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-layer heatmap grid
# ---------------------------------------------------------------------------

def plot_heatmaps_grid(
    original_pil: Image.Image,
    score_maps: dict[int, np.ndarray],
    output_path: str,
    title: str = "",
    colormap: str = "hot",
    alpha: float = 0.6,
    dpi: int = 150,
    ncols: int = 6,
) -> None:
    """Plot one heatmap per layer arranged in a grid, with the original image first.

    Args:
        original_pil: Original PIL image shown in the first cell.
        score_maps: ``{layer_idx: score_map [H_p, W_p]}``.
        output_path: Destination PNG path.
        ncols: Number of columns in the grid.
        Other args: same as :func:`plot_heatmap`.
    """
    W_img, H_img = original_pil.size
    layers = sorted(score_maps.keys())
    n = 1 + len(layers)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3), dpi=dpi)
    axes = np.array(axes).flatten()

    # First cell: original image
    axes[0].imshow(original_pil)
    axes[0].set_title("Original", fontsize=8)
    axes[0].axis("off")

    for col, layer_idx in enumerate(layers, start=1):
        ax = axes[col]
        up = _upsample_score_map(score_maps[layer_idx], H_img, W_img)
        overlay = _score_map_to_rgba(up, colormap, alpha)
        ax.imshow(original_pil)
        ax.imshow(overlay)
        ax.set_title(f"Layer {layer_idx}", fontsize=8)
        ax.axis("off")

    # Hide unused subplots
    for ax in axes[n:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Insertion / deletion curves
# ---------------------------------------------------------------------------

def plot_curves(
    ins_x: np.ndarray,
    ins_y: np.ndarray,
    ins_auc: float,
    del_x: np.ndarray,
    del_y: np.ndarray,
    del_auc: float,
    output_path: str,
    title: str = "",
    dpi: int = 150,
) -> None:
    """Plot insertion and deletion curves side by side.

    Args:
        ins_x / ins_y: Fraction of patches inserted and model probability.
        ins_auc: AUC of insertion curve (higher = better).
        del_x / del_y: Fraction of patches deleted and model probability.
        del_auc: AUC of deletion curve (lower = better).
        output_path: Destination PNG path.
        title: Optional suptitle.
        dpi: Figure DPI.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)

    ax1.plot(ins_x, ins_y, color="steelblue", linewidth=1.5,
             label=f"Insertion (AUC={ins_auc:.4f})")
    ax1.set_xlabel("Fraction of patches inserted")
    ax1.set_ylabel("Model probability of y_hat")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=9)
    ax1.set_title("Insertion ↑")

    ax2.plot(del_x, del_y, color="tomato", linewidth=1.5,
             label=f"Deletion (AUC={del_auc:.4f})")
    ax2.set_xlabel("Fraction of patches deleted")
    ax2.set_ylabel("Model probability of y_hat")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9)
    ax2.set_title("Deletion ↓")

    if title:
        fig.suptitle(title, fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined 4-panel report
# ---------------------------------------------------------------------------

def plot_combined_report(
    original_pil: Image.Image,
    score_map: np.ndarray,
    ins_x: np.ndarray,
    ins_y: np.ndarray,
    ins_auc: float,
    del_x: np.ndarray,
    del_y: np.ndarray,
    del_auc: float,
    output_path: str,
    title: str = "",
    colormap: str = "hot",
    alpha: float = 0.6,
    dpi: int = 150,
) -> None:
    """Single figure: original | heatmap | insertion curve | deletion curve.

    Args:
        original_pil: Original PIL image.
        score_map: ``[H_p, W_p]`` patch scores.
        Other args: same as :func:`plot_curves` and :func:`plot_heatmap`.
    """
    W_img, H_img = original_pil.size
    up = _upsample_score_map(score_map, H_img, W_img)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=dpi)

    # Panel 1 – original
    axes[0].imshow(original_pil)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    # Panel 2 – heatmap overlay
    overlay = _score_map_to_rgba(up, colormap, alpha)
    axes[1].imshow(original_pil)
    axes[1].imshow(overlay)
    axes[1].set_title("Attribution Heatmap", fontsize=10)
    axes[1].axis("off")

    # Panel 3 – insertion
    axes[2].plot(ins_x, ins_y, color="steelblue", linewidth=1.5,
                 label=f"AUC={ins_auc:.4f}")
    axes[2].set_xlabel("Fraction inserted")
    axes[2].set_ylabel("Probability")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=9)
    axes[2].set_title("Insertion ↑", fontsize=10)

    # Panel 4 – deletion
    axes[3].plot(del_x, del_y, color="tomato", linewidth=1.5,
                 label=f"AUC={del_auc:.4f}")
    axes[3].set_xlabel("Fraction deleted")
    axes[3].set_ylabel("Probability")
    axes[3].set_xlim(0, 1)
    axes[3].set_ylim(0, 1)
    axes[3].legend(fontsize=9)
    axes[3].set_title("Deletion ↓", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregate curves (mean ± std across images)
# ---------------------------------------------------------------------------

def plot_aggregate_curves(
    ins_x: np.ndarray,
    ins_y_list: list[np.ndarray],
    del_x: np.ndarray,
    del_y_list: list[np.ndarray],
    output_path: str,
    title: str = "",
    dpi: int = 150,
) -> None:
    """Plot mean ± 1 std insertion and deletion curves across multiple images.

    All arrays in ``ins_y_list`` / ``del_y_list`` must have the same length as
    ``ins_x`` / ``del_x`` respectively.

    Args:
        ins_x: Shared x-axis for insertion (fraction of patches).
        ins_y_list: List of per-image insertion probability arrays.
        del_x: Shared x-axis for deletion.
        del_y_list: List of per-image deletion probability arrays.
        output_path: Destination PNG path.
        title: Optional suptitle.
        dpi: Figure DPI.
    """
    from sklearn.metrics import auc as sklearn_auc

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)

    def _draw(ax, x, y_list, color, label_prefix, higher_is_better: bool):
        y_arr = np.stack(y_list, axis=0)   # [N, steps]
        mean = y_arr.mean(axis=0)
        std = y_arr.std(axis=0)
        aucs = [float(sklearn_auc(x, y)) for y in y_list]
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))

        ax.plot(x, mean, color=color, linewidth=2,
                label=f"{label_prefix} (AUC={mean_auc:.4f} ± {std_auc:.4f})")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_xlabel("Fraction of patches")
        ax.set_ylabel("Model probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(1.0, float((mean + std).max()) + 0.05))
        arrow = "↑" if higher_is_better else "↓"
        ax.set_title(f"{label_prefix} {arrow}", fontsize=11)
        ax.legend(fontsize=9)
        ax.text(
            0.98, 0.02, f"n={len(y_list)}",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=8, color="gray",
        )

    _draw(ax1, ins_x, ins_y_list, "steelblue", "Insertion", higher_is_better=True)
    _draw(ax2, del_x, del_y_list, "tomato",    "Deletion",  higher_is_better=False)

    if title:
        fig.suptitle(title, fontsize=13)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
