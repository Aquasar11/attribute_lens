"""
eval.localization
===========================
CAAP localization evaluation (Pointing Game, AUPR1, AUPR0), driven by a YAML config.

Per annotated image: score patches with the trained lens, upsample the attribution map to
pixels, and compare against the ground-truth FG/BG mask. Shared model/lens plumbing lives
in :mod:`eval.common`.

Run:
    python -m eval.localization --config configs/eval/localization.yaml
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import auc, precision_recall_curve
from tqdm.auto import tqdm

from . import common
from .common import GRID, IMG_SIZE, IMAGE_SIZE
from .config import LocalizationConfig


def calculate_caap_localization_metrics(attr_map_1d, gt_mask_pil):
    """Pointing Game + AUPR for FG (1) and BG (0) from a (256,) patch-score map."""
    attr_grid = attr_map_1d.reshape(GRID, GRID)
    attr_resized = cv2.resize(attr_grid, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

    gt_mask = np.array(gt_mask_pil.resize(IMAGE_SIZE, Image.NEAREST))
    y_true = (gt_mask > 0).astype(int).flatten()
    y_scores = attr_resized.flatten()

    max_idx = np.argmax(y_scores)
    pg_score = 1.0 if y_true[max_idx] == 1 else 0.0

    if len(np.unique(y_true)) < 2:
        return pg_score, np.nan, np.nan

    prec1, rec1, _ = precision_recall_curve(y_true, y_scores)
    aupr1_score = auc(rec1, prec1)

    prec0, rec0, _ = precision_recall_curve(1 - y_true, -y_scores)
    aupr0_score = auc(rec0, prec0)

    return pg_score, aupr1_score, aupr0_score


def run(cfg: LocalizationConfig, clip_model=None):
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

    activations = {}
    hook_handles = []
    for layer_name in target_layers:
        layer_idx = int("".join(filter(str.isdigit, layer_name)))
        h = clip_model.blocks[layer_idx].register_forward_hook(
            common.make_activation_hook(activations, layer_name))
        hook_handles.append(h)
    print(f"Hooked {len(hook_handles)} layers ({target_layers[0]}..{target_layers[-1]})")

    img_dir_path = Path(cfg.image_dir)
    mask_dir_path = Path(cfg.mask_dir)
    all_imgs = sorted(p.stem for p in img_dir_path.glob("*.JPEG"))
    val_names = [n for n in all_imgs if (mask_dir_path / f"{n}.png").exists()]

    results = []
    print(f"\nEvaluating localization on {len(val_names)} images from `{cfg.image_dir}`...")
    for base_name in tqdm(val_names):
        img_path = os.path.join(cfg.image_dir, f"{base_name}.JPEG")
        mask_path = os.path.join(cfg.mask_dir, f"{base_name}.png")
        if not os.path.exists(mask_path):
            continue

        pil_img = Image.open(img_path).convert("RGB")
        gt_mask_pil = Image.open(mask_path).convert("L")
        img_tensor = common.transform_image(pil_img, device)

        with torch.no_grad():
            logits_base = clip_model(img_tensor)
            pred_class_id = logits_base.argmax(dim=-1).item()

        for layer_name in target_layers:
            with torch.no_grad():
                layer_patches = activations[layer_name][:, 1:, :]
                mapped = lens_model.projections[layer_name](layer_patches)
                logits = logits_fn(mapped)
                probs = F.softmax(logits, dim=-1)
                attr_1d = probs[0, :, pred_class_id].cpu().numpy()

            pg, aupr1, aupr0 = calculate_caap_localization_metrics(attr_1d, gt_mask_pil)
            results.append({
                "image_name": base_name, "target_class": pred_class_id, "layer": layer_name,
                "Pointing_Game": pg, "AUPR1": aupr1, "AUPR0": aupr0,
            })

    for h in hook_handles:
        h.remove()

    results_df = pd.DataFrame(results)
    results_df.to_csv(cfg.all_metrics_out_csv, index=False)
    print(f"\nSaved localization metrics to: {cfg.all_metrics_out_csv}")

    print("\n" + "=" * 60)
    print(f"PER-LAYER LOCALIZATION SUMMARY  (mapper_type={cfg.mapper_type})")
    print("=" * 60)
    summary = (results_df.groupby("layer")[["Pointing_Game", "AUPR1", "AUPR0"]]
               .mean().reindex(target_layers))
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 60)
    return results_df


def main():
    p = argparse.ArgumentParser(description="CAAP localization evaluation (PG / AUPR1 / AUPR0).")
    p.add_argument("--config", required=True, help="path to a localization YAML config")
    p.add_argument("--mapper-type", default=None,
                   help="override mapper_type (unified|full|low_rank|direct_lens)")
    p.add_argument("--image-dir", default=None, help="override image_dir")
    p.add_argument("--mask-dir", default=None, help="override mask_dir")
    args = p.parse_args()
    cfg = LocalizationConfig.from_yaml(
        args.config, mapper_type=args.mapper_type, image_dir=args.image_dir, mask_dir=args.mask_dir)
    run(cfg)


if __name__ == "__main__":
    main()
