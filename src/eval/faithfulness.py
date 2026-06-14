"""
eval.faithfulness
===========================
Insertion / deletion faithfulness evaluation, driven by a YAML config.

Per image: score every target layer's patches with the trained lens (one forward pass),
optionally neighbour-average (ring-sigma or multi-kernel, plain or edge-aware bilateral)
in the chosen score space, optionally fuse layers into one map, then rank patches and
measure insertion (blurred baseline, higher=better) and deletion (zeroed, lower=better)
AUC. Shared model/lens plumbing lives in :mod:`eval.common`.

Run:
    python -m eval.faithfulness --config configs/eval/faithfulness.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from tqdm.auto import tqdm
from PIL import Image

from . import common
from .common import EMBED_DIM, GRID
from .config import FaithfulnessConfig


# ============================================================
# NEIGHBOUR-AVERAGING KERNELS
# ============================================================
def build_ring_sigma_kernel(size, sigmas, device, dtype):
    """Single ``size×size`` Gaussian kernel with one sigma per Chebyshev ring.

    Closer rings get a LARGER sigma (gentler falloff, more weight); farther rings get
    smaller sigmas. Returned UNNORMALISED — per-position normalisation happens at apply
    time via a denominator convolution so edges/corners remain correct weighted averages.
    """
    half = size // 2
    coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    dist2 = xx ** 2 + yy ** 2
    ring = torch.maximum(xx.abs(), yy.abs()).long()
    kernel = torch.ones(size, size, device=device, dtype=dtype)
    for r in range(1, half + 1):
        sigma = float(sigmas[r - 1])
        m = ring == r
        kernel[m] = torch.exp(-dist2[m] / (2.0 * sigma ** 2))
    return kernel


def _make_single_sigma_kernel(size, sigma, device, dtype):
    """Unnormalized 2-D Gaussian kernel with a single global sigma."""
    if size == 1:
        return torch.ones(1, 1, device=device, dtype=dtype)
    half = size // 2
    coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))


def smooth_scores_2d(score_2d, kernel):
    """Boundary-aware Gaussian average of a ``[H, W]`` score map."""
    H, W = score_2d.shape
    size = kernel.shape[0]
    pad = size // 2
    k4 = kernel.view(1, 1, size, size)
    x = score_2d.view(1, 1, H, W)
    num = F.conv2d(x, k4, padding=pad)
    den = F.conv2d(torch.ones_like(x), k4, padding=pad)
    return (num / (den + 1e-8)).view(H, W)


def smooth_scores_2d_multi(score_2d, kernels, weights):
    """Boundary-aware multi-kernel Gaussian average of a ``[H, W]`` score map."""
    H, W = score_2d.shape
    x = score_2d.view(1, 1, H, W)
    ones = torch.ones_like(x)
    accumulated = None
    for kernel, w in zip(kernels, weights):
        size = kernel.shape[0]
        pad = size // 2
        k4 = kernel.view(1, 1, size, size)
        smoothed = (F.conv2d(x, k4, padding=pad) / (F.conv2d(ones, k4, padding=pad) + 1e-8)).view(H, W)
        accumulated = smoothed * w if accumulated is None else accumulated + smoothed * w
    return accumulated


def smooth_embeddings(patch_tokens, kernel, grid):
    """Boundary-aware Gaussian average of patch embeddings BEFORE the lens."""
    B, N, D = patch_tokens.shape
    size = kernel.shape[0]
    pad = size // 2
    x = patch_tokens.view(B, grid, grid, D).permute(0, 3, 1, 2).contiguous()
    k_dw = kernel.view(1, 1, size, size).expand(D, 1, size, size).contiguous()
    num = F.conv2d(x, k_dw, padding=pad, groups=D)
    ones = torch.ones(B, 1, grid, grid, device=x.device, dtype=x.dtype)
    den = F.conv2d(ones, kernel.view(1, 1, size, size), padding=pad)
    out = num / (den + 1e-8)
    return out.permute(0, 2, 3, 1).reshape(B, N, D)


def smooth_embeddings_multi(patch_tokens, kernels, weights, grid):
    """Boundary-aware multi-kernel Gaussian average of patch embeddings."""
    B, N, D = patch_tokens.shape
    x = patch_tokens.view(B, grid, grid, D).permute(0, 3, 1, 2).contiguous()
    ones = torch.ones(B, 1, grid, grid, device=x.device, dtype=x.dtype)
    accumulated = None
    for kernel, w in zip(kernels, weights):
        size = kernel.shape[0]
        pad = size // 2
        k_dw = kernel.view(1, 1, size, size).expand(D, 1, size, size).contiguous()
        num = F.conv2d(x, k_dw, padding=pad, groups=D)
        den = F.conv2d(ones, kernel.view(1, 1, size, size), padding=pad)
        smoothed = num / (den + 1e-8)
        accumulated = smoothed * w if accumulated is None else accumulated + smoothed * w
    return accumulated.permute(0, 2, 3, 1).reshape(B, N, D)


def _bilateral_weights(emb_grid, kernel, sigma_range_mult):
    """Per-window bilateral weights ``[B, k*k, H*W]`` (spatial × feature similarity)."""
    B, D, H, W = emb_grid.shape
    k = kernel.shape[0]
    pad = k // 2
    L = H * W
    nb = F.unfold(emb_grid, kernel_size=k, padding=pad).view(B, D, k * k, L)
    center = emb_grid.view(B, D, 1, L)
    dist2 = ((nb - center) ** 2).sum(dim=1)
    ones = torch.ones(B, 1, H, W, device=emb_grid.device, dtype=emb_grid.dtype)
    valid = F.unfold(ones, kernel_size=k, padding=pad)
    center_idx = (k * k) // 2
    nbmask = valid.clone()
    nbmask[:, center_idx, :] = 0.0
    mean_dist2 = (dist2 * nbmask).sum() / (nbmask.sum() + 1e-8)
    sigma_r2 = sigma_range_mult * mean_dist2 + 1e-8
    wr = torch.exp(-dist2 / (2.0 * sigma_r2))
    ws = kernel.view(1, k * k, 1)
    return ws * wr * valid


def smooth_scores_bilateral(score_2d, emb_grid, kernel, sigma_range_mult):
    """Edge-aware Gaussian average of a ``[H, W]`` score map."""
    H, W = score_2d.shape
    k = kernel.shape[0]
    pad = k // 2
    w = _bilateral_weights(emb_grid, kernel, sigma_range_mult)
    s_nb = F.unfold(score_2d.view(1, 1, H, W), kernel_size=k, padding=pad)
    out = (w * s_nb).sum(dim=1) / (w.sum(dim=1) + 1e-8)
    return out.view(H, W)


def smooth_embeddings_bilateral(patch_tokens, kernel, sigma_range_mult, grid):
    """Edge-aware (self-guided) Gaussian average of patch embeddings BEFORE the lens."""
    B, N, D = patch_tokens.shape
    k = kernel.shape[0]
    pad = k // 2
    emb_grid = patch_tokens.view(B, grid, grid, D).permute(0, 3, 1, 2).contiguous()
    w = _bilateral_weights(emb_grid, kernel, sigma_range_mult)
    nb = F.unfold(emb_grid, kernel_size=k, padding=pad).view(B, D, k * k, N)
    out = (w.unsqueeze(1) * nb).sum(dim=2) / (w.sum(dim=1, keepdim=True) + 1e-8)
    return out.permute(0, 2, 1).reshape(B, N, D)


def smooth_scores_bilateral_multi(score_2d, emb_grid, kernels, weights, sigma_range_mult):
    accumulated = None
    for kernel, w in zip(kernels, weights):
        smoothed = smooth_scores_bilateral(score_2d, emb_grid, kernel, sigma_range_mult)
        accumulated = smoothed * w if accumulated is None else accumulated + smoothed * w
    return accumulated


def smooth_embeddings_bilateral_multi(patch_tokens, kernels, weights, sigma_range_mult, grid):
    accumulated = None
    for kernel, w in zip(kernels, weights):
        smoothed = smooth_embeddings_bilateral(patch_tokens, kernel, sigma_range_mult, grid)
        accumulated = smoothed * w if accumulated is None else accumulated + smoothed * w
    return accumulated


# ============================================================
# SCORE SPACE & LAYER AGGREGATION
# ============================================================
def class_score_field(logits, class_id, space):
    """Per-patch score ``[N]`` for the target class in ``prob`` | ``logprob`` | ``logit``."""
    z = logits[0]
    if space == "prob":
        return F.softmax(z, dim=-1)[:, class_id]
    if space == "logprob":
        return F.log_softmax(z, dim=-1)[:, class_id]
    if space == "logit":
        return z[:, class_id]
    raise ValueError(f"Unknown score_space '{space}'. Use 'prob', 'logprob', or 'logit'.")


def _rank_normalize(field):
    """``[N]`` scores -> per-patch goodness in ``[0,1]`` by rank (scale-free)."""
    n = field.shape[0]
    order = np.argsort(field)[::-1]
    goodness = np.empty(n, dtype=np.float64)
    goodness[order] = np.linspace(1.0, 0.0, n)
    return goodness


def _minmax_normalize(field):
    f = field.astype(np.float64)
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return np.zeros_like(f)
    return (f - lo) / (hi - lo)


def layer_confidence(prob_field, topk):
    """Label-free per-layer confidence: mean of the top-K patch P(y_hat)."""
    k = int(min(topk, prob_field.shape[0]))
    return float(np.sort(prob_field)[::-1][:k].mean())


def fuse_layer_fields(layer_fields, layer_confs, agg_mode, norm_mode, gamma):
    """Fuse ``{name: score_field[N]}`` into one ``[N]`` attribution. Returns (fused, weights)."""
    names = list(layer_fields.keys())
    if norm_mode == "rank":
        normed = {nm: _rank_normalize(layer_fields[nm]) for nm in names}
    elif norm_mode == "minmax":
        normed = {nm: _minmax_normalize(layer_fields[nm]) for nm in names}
    else:
        raise ValueError(f"Unknown layer_norm '{norm_mode}'. Use 'rank' or 'minmax'.")

    if agg_mode == "uniform":
        w = {nm: 1.0 for nm in names}
    elif agg_mode == "confidence":
        w = {nm: float(layer_confs[nm]) ** gamma for nm in names}
    elif agg_mode == "ramp":
        idx = {nm: int("".join(filter(str.isdigit, nm))) for nm in names}
        lo = min(idx.values())
        w = {nm: float(idx[nm] - lo + 1) for nm in names}
    else:
        raise ValueError(f"Unknown layer_agg '{agg_mode}'. Use uniform|confidence|ramp.")

    tot = sum(w.values()) or 1.0
    w = {nm: w[nm] / tot for nm in names}
    fused = np.zeros(layer_fields[names[0]].shape[0], dtype=np.float64)
    for nm in names:
        fused += w[nm] * normed[nm]
    return fused, w


# ============================================================
# PLOTTING
# ============================================================
def save_insertion_deletion_plots(ins_f, ins_p, ins_auc, del_f, del_p, del_auc,
                                  image_name, layer_name, out_dir):
    """Two separate plots (insertion, deletion) for one image/layer pair."""
    os.makedirs(out_dir, exist_ok=True)
    safe_img = os.path.splitext(os.path.basename(str(image_name)))[0]
    base = f"{safe_img}__{layer_name}"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ins_f, ins_p, color="#1f77b4", label=f"AUC={ins_auc:.4f}")
    ax.set_title("Insertion ↑"); ax.set_xlabel("Fraction inserted"); ax.set_ylabel("Probability")
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{base}__insertion.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(del_f, del_p, color="#ff7f0e", label=f"AUC={del_auc:.4f}")
    ax.set_title("Deletion ↓"); ax.set_xlabel("Fraction deleted"); ax.set_ylabel("Probability")
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{base}__deletion.png"), dpi=150)
    plt.close(fig)


def save_aggregated_plots_per_layer(per_layer_curves, out_dir, grid_points=257):
    """One aggregated insertion + one deletion plot per layer (mean over images)."""
    if not per_layer_curves:
        return
    os.makedirs(out_dir, exist_ok=True)
    grid = np.linspace(0.0, 1.0, grid_points)
    for layer_name, curves in per_layer_curves.items():
        for kind, color, title, fname in (
            ("ins", "#1f77b4", "Insertion ↑  (mean over samples)", "insertion"),
            ("del", "#ff7f0e", "Deletion ↓  (mean over samples)", "deletion"),
        ):
            curve_list = curves.get(kind, [])
            if not curve_list:
                continue
            interp = np.stack([
                np.interp(grid, np.asarray(f, dtype=float), np.asarray(p, dtype=float))
                for f, p in curve_list
            ], axis=0)
            mean_p = interp.mean(axis=0)
            agg_auc = float(auc(grid, mean_p))
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(grid, mean_p, color=color, label=f"AUC={agg_auc:.4f}  (n={len(curve_list)})")
            ax.set_title(title); ax.set_xlabel(f"Fraction {fname}d"); ax.set_ylabel("Probability")
            ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"AGG__{layer_name}__{fname}.png"), dpi=150)
            plt.close(fig)


# ============================================================
# RUN
# ============================================================
def run(cfg: FaithfulnessConfig, clip_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  mapper_type={cfg.mapper_type}")

    if clip_model is None:
        print("Loading ViT model...")
        clip_model = common.load_clip_model(cfg.model_name, cfg.timm_weights_path, device)

    lens_model, logits_fn, target_layers = common.build_lens_model(
        clip_model, cfg.mapper_type, cfg.target_layers, device,
        mapper_dir=cfg.mapper_weights_dir, lens_dir=cfg.lens_weights_dir,
        unified_dir=cfg.unified_weights_dir, direct_lens_path=cfg.direct_lens_path,
        direct_lens_layer=cfg.direct_lens_layer, lens_out_dim_override=cfg.lens_out_dim,
    )

    # --- build kernel(s) once ---
    kernel = multi_kernels = multi_weights_norm = None
    if cfg.use_kernel:
        assert cfg.kernel_mode in ("score", "embedding")
        assert cfg.kernel_approach in ("ring_sigma", "multi_kernel")
        if cfg.kernel_approach == "ring_sigma":
            assert cfg.kernel_size % 2 == 1, "kernel_size must be odd."
            n_rings = cfg.kernel_size // 2
            assert len(cfg.kernel_sigmas) == n_rings, (
                f"kernel_size={cfg.kernel_size} has {n_rings} rings; "
                f"provide {n_rings} sigmas (got {len(cfg.kernel_sigmas)})."
            )
            kernel = build_ring_sigma_kernel(cfg.kernel_size, cfg.kernel_sigmas, device, torch.float32)
            print(f"[kernel] ring_sigma mode={cfg.kernel_mode} size={cfg.kernel_size} "
                  f"sigmas={cfg.kernel_sigmas}")
        else:
            assert len(cfg.multi_kernel_sizes) == len(cfg.multi_kernel_sigmas) == len(cfg.multi_kernel_weights)
            assert all(s % 2 == 1 for s in cfg.multi_kernel_sizes), "multi_kernel_sizes must be odd."
            w_total = sum(cfg.multi_kernel_weights)
            multi_weights_norm = [w / w_total for w in cfg.multi_kernel_weights]
            multi_kernels = [
                _make_single_sigma_kernel(s, sig, device, torch.float32)
                for s, sig in zip(cfg.multi_kernel_sizes, cfg.multi_kernel_sigmas)
            ]
            print(f"[kernel] multi_kernel mode={cfg.kernel_mode} sizes={cfg.multi_kernel_sizes} "
                  f"sigmas={cfg.multi_kernel_sigmas} weights(norm)={multi_weights_norm}")
        if cfg.bilateral:
            assert cfg.bilateral_sigma_range > 0
            print(f"[kernel] bilateral=ON sigma_range_mult={cfg.bilateral_sigma_range}")
        else:
            print("[kernel] bilateral=OFF (plain spatial Gaussian)")
    else:
        print("[kernel] disabled")

    assert cfg.score_space in ("prob", "logprob", "logit")
    assert cfg.layer_agg in ("off", "uniform", "confidence", "ramp")
    assert cfg.layer_norm in ("rank", "minmax")
    print(f"[score] ranking/smoothing space={cfg.score_space} (curves use true P(y_hat))")
    print(f"[layer-agg] {cfg.layer_agg}" + ("" if cfg.layer_agg == "off"
          else f" norm={cfg.layer_norm} -> fused '{cfg.agg_layer_name}'"))

    # --- hook all target layers (one forward pass populates all) ---
    activations = {}
    hook_handles = []
    for layer_name in target_layers:
        layer_idx = int("".join(filter(str.isdigit, layer_name)))
        h = clip_model.blocks[layer_idx].register_forward_hook(
            common.make_activation_hook(activations, layer_name))
        hook_handles.append(h)
    print(f"Hooked {len(hook_handles)} layers ({target_layers[0]}..{target_layers[-1]})")

    # --- data ---
    df = pd.read_csv(cfg.y_hat_csv)
    col_img = "image_name" if "image_name" in df.columns else "image_id"
    col_cls = "y_hat" if "y_hat" in df.columns else "class_id"
    unique_images = df[[col_img, col_cls]].drop_duplicates().reset_index(drop=True)

    do_per_layer = cfg.eval_per_layer or cfg.layer_agg == "off"
    do_agg = cfg.layer_agg != "off"
    curve_names = (list(target_layers) if do_per_layer else []) \
        + ([cfg.agg_layer_name] if do_agg else [])

    all_scores_dict, results = {}, []
    per_layer_curves = {ln: {"ins": [], "del": []} for ln in curve_names}

    def evaluate_and_record(name, ranked, scores_np, image_name, pil_img, class_id):
        all_scores_dict[f"{image_name}__{name}"] = scores_np
        del_f, del_p = common.deletion_curve_batched(
            clip_model, pil_img, class_id, ranked, cfg.eval_batch, device)
        del_auc = auc(del_f, del_p)
        ins_f, ins_p = common.insertion_curve_ublur_batched(
            clip_model, pil_img, class_id, ranked, cfg.eval_batch, device)
        ins_auc = auc(ins_f, ins_p)
        save_insertion_deletion_plots(ins_f, ins_p, ins_auc, del_f, del_p, del_auc,
                                      image_name=image_name, layer_name=name,
                                      out_dir=cfg.plots_out_dir)
        per_layer_curves[name]["ins"].append((ins_f, ins_p))
        per_layer_curves[name]["del"].append((del_f, del_p))
        results.append({
            "image_name": image_name, "y_hat": class_id, "layer": name,
            "deletion_auc": del_auc, "insertion_auc": ins_auc,
            "ins_minus_del": ins_auc - del_auc,
        })

    print(f"\nEvaluating {len(target_layers)} layers on {len(unique_images)} images "
          f"(per-layer={do_per_layer}, fused={do_agg})...")
    for img_i, row in enumerate(tqdm(unique_images.itertuples(index=False), total=len(unique_images))):
        image_name = str(getattr(row, col_img))
        class_id = int(getattr(row, col_cls))
        image_path = common.resolve_image_path(cfg.image_dir, image_name)
        if image_path is None:
            print(f"Skipping (missing): {image_name}")
            continue

        pil_img = Image.open(image_path).convert("RGB").resize(common.IMAGE_SIZE)
        img_tensor = common.transform_image(pil_img, device)

        with torch.no_grad():
            _ = clip_model(img_tensor)   # populates `activations` for all hooked layers

        # --- Phase A: score every layer from the CLEAN activations ---
        layer_fields, layer_confs = {}, {}
        for layer_name in target_layers:
            with torch.no_grad():
                layer_patches = activations[layer_name][:, 1:, :]
                raw_emb = layer_patches

                if cfg.use_kernel and cfg.kernel_mode == "embedding":
                    if cfg.kernel_approach == "ring_sigma":
                        layer_patches = (smooth_embeddings_bilateral(layer_patches, kernel,
                                         cfg.bilateral_sigma_range, GRID) if cfg.bilateral
                                         else smooth_embeddings(layer_patches, kernel, GRID))
                    else:
                        layer_patches = (smooth_embeddings_bilateral_multi(layer_patches, multi_kernels,
                                         multi_weights_norm, cfg.bilateral_sigma_range, GRID) if cfg.bilateral
                                         else smooth_embeddings_multi(layer_patches, multi_kernels,
                                         multi_weights_norm, GRID))

                mapped = lens_model.projections[layer_name](layer_patches)
                logits = logits_fn(mapped)
                score_field = class_score_field(logits, class_id, cfg.score_space)

                if cfg.use_kernel and cfg.kernel_mode == "score":
                    emb_grid = raw_emb.view(1, GRID, GRID, EMBED_DIM).permute(0, 3, 1, 2)
                    if cfg.kernel_approach == "ring_sigma":
                        score_field = (smooth_scores_bilateral(score_field.view(GRID, GRID), emb_grid,
                                       kernel, cfg.bilateral_sigma_range) if cfg.bilateral
                                       else smooth_scores_2d(score_field.view(GRID, GRID), kernel)).reshape(-1)
                    else:
                        score_field = (smooth_scores_bilateral_multi(score_field.view(GRID, GRID), emb_grid,
                                       multi_kernels, multi_weights_norm, cfg.bilateral_sigma_range)
                                       if cfg.bilateral else smooth_scores_2d_multi(
                                       score_field.view(GRID, GRID), multi_kernels, multi_weights_norm)).reshape(-1)

                scores_np = score_field.cpu().numpy()
                prob_field = F.softmax(logits[0], dim=-1)[:, class_id].cpu().numpy()

            layer_fields[layer_name] = scores_np
            layer_confs[layer_name] = layer_confidence(prob_field, cfg.layer_conf_topk)

        # --- Phase B: evaluate (these forward passes overwrite `activations`) ---
        if do_per_layer:
            for layer_name in target_layers:
                scores_np = layer_fields[layer_name]
                ranked = np.argsort(scores_np)[::-1]
                evaluate_and_record(layer_name, ranked, scores_np, image_name, pil_img, class_id)

        if do_agg:
            fused, _w = fuse_layer_fields(layer_fields, layer_confs, cfg.layer_agg,
                                          cfg.layer_norm, cfg.layer_conf_gamma)
            ranked = np.argsort(fused)[::-1]
            evaluate_and_record(cfg.agg_layer_name, ranked, fused, image_name, pil_img, class_id)

        pd.DataFrame(results).to_csv(cfg.all_metrics_out_csv, index=False)
        if (img_i + 1) % 20 == 0:
            np.savez_compressed(cfg.all_scores_out_file, **all_scores_dict)

    for h in hook_handles:
        h.remove()

    np.savez_compressed(cfg.all_scores_out_file, **all_scores_dict)
    results_df = pd.DataFrame(results)
    results_df.to_csv(cfg.all_metrics_out_csv, index=False)
    save_aggregated_plots_per_layer(per_layer_curves, out_dir=cfg.plots_out_dir)
    print(f"\nSaved scores -> {cfg.all_scores_out_file}; metrics -> {cfg.all_metrics_out_csv}; "
          f"plots -> {cfg.plots_out_dir}")

    print("\n" + "=" * 60)
    print(f"PER-LAYER SUMMARY  (mapper_type={cfg.mapper_type})")
    print("=" * 60)
    summary = (results_df.groupby("layer")[["deletion_auc", "insertion_auc", "ins_minus_del"]]
               .mean().reindex(curve_names))
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 60)
    return results_df


def main():
    p = argparse.ArgumentParser(description="Insertion/deletion faithfulness evaluation.")
    p.add_argument("--config", required=True, help="path to a faithfulness YAML config")
    p.add_argument("--mapper-type", default=None,
                   help="override mapper_type (unified|full|low_rank|direct_lens)")
    p.add_argument("--image-dir", default=None, help="override image_dir")
    args = p.parse_args()
    cfg = FaithfulnessConfig.from_yaml(
        args.config, mapper_type=args.mapper_type, image_dir=args.image_dir)
    run(cfg)


if __name__ == "__main__":
    main()
