"""Attribution evaluation script.

CLI usage::

    python -m attribute_lens.evaluate --config configs/attribution_default.yaml

    # Override specific settings:
    python -m attribute_lens.evaluate \\
        --config configs/attribution_default.yaml \\
        --image /path/to/image.JPEG \\
        --output-dir outputs/my_eval \\
        --device cuda \\
        --scorer-type cls \\
        --layer 6 --layer 12 --layer 18 \\
        --num-images 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from tuned_lens.config import ModelConfig
from tuned_lens.model import VisionModelWrapper

from .config import AttributionConfig, EvalSection
from .scorer import CLSLensScorer, PatchLensScorer, PatchMapCLSLensScorer, discover_lens_files
from .metrics import apply_gaussian_blur, insertion_deletion_curves_batch
from .visualize import (
    plot_heatmap,
    plot_heatmaps_grid,
    plot_curves,
    plot_combined_report,
    plot_aggregate_curves,
)


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}


def _collect_images(cfg: EvalSection, seed: int) -> list[str]:
    """Collect image paths according to config, apply num_images limit."""
    if cfg.image_paths:
        paths = list(cfg.image_paths)
    elif cfg.image_dir:
        paths = []
        for ext in _IMAGE_EXTENSIONS:
            paths.extend(glob(os.path.join(cfg.image_dir, "**", f"*{ext}"), recursive=True))
        paths = sorted(set(paths))
    else:
        raise ValueError("Provide either eval.image_paths or eval.image_dir in config.")

    if not paths:
        raise ValueError("No images found. Check eval.image_paths / eval.image_dir.")

    if cfg.num_images is not None and cfg.num_images < len(paths):
        rng = random.Random(seed)
        paths = rng.sample(paths, cfg.num_images)
        paths = sorted(paths)   # deterministic order after sampling

    return paths


# ---------------------------------------------------------------------------
# Layer resolution
# ---------------------------------------------------------------------------

def _resolve_target_layers(
    config: AttributionConfig,
    scorer_type: str,
) -> list[int]:
    """Determine target layers from available lens files and config override."""
    layers: set[int] = set()

    if scorer_type in ("cls", "both", "all") and config.lens.cls_lens_dir:
        layers |= set(discover_lens_files(config.lens.cls_lens_dir).keys())

    if scorer_type in ("patch", "both", "all") and config.lens.patch_lens_dir:
        available = set(discover_lens_files(config.lens.patch_lens_dir).keys())
        if layers:
            layers &= available   # intersection when both scorers active
        else:
            layers = available

    if scorer_type in ("patch_map_cls", "all") and config.lens.patch_map_dir:
        # Require intersection of patch_map layers and CLS lens layers
        pm_available = set(discover_lens_files(config.lens.patch_map_dir).keys())
        cls_available = (
            set(discover_lens_files(config.lens.cls_lens_dir).keys())
            if config.lens.cls_lens_dir
            else pm_available
        )
        available = pm_available & cls_available
        if layers:
            layers &= available
        else:
            layers = available

    if not layers:
        raise ValueError(
            "No lens checkpoint files found. "
            "Check lens.cls_lens_dir / lens.patch_lens_dir / lens.patch_map_dir in config."
        )

    if config.model.target_layers is not None:
        layers &= set(config.model.target_layers)

    if not layers:
        raise ValueError(
            "target_layers filter left no layers. "
            "Check model.target_layers in config."
        )

    return sorted(layers)


# ---------------------------------------------------------------------------
# Model wrapper factory
# ---------------------------------------------------------------------------

def _build_wrapper(config: AttributionConfig, device: str) -> VisionModelWrapper:
    """Create a VisionModelWrapper in patch_mode=True on the target device."""
    model_cfg = ModelConfig(
        model_name=config.model.model_name,
        weights_path=config.model.weights_path,
        pretrained=True,
        target_layers=None,   # hook all layers; we filter in scorer
        freeze_model=True,
        patch_mode=True,
    )
    wrapper = VisionModelWrapper(model_cfg, device=device)
    return wrapper


# ---------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------

def _infer_patch_size(wrapper: VisionModelWrapper) -> int:
    """Return the pixel side length of each patch from the model's patch_embed."""
    return wrapper.model.patch_embed.patch_size[0]


# ---------------------------------------------------------------------------
# Batch helpers: score map computation and perturbation curves
# ---------------------------------------------------------------------------

def _compute_batch_score_maps(
    batch_patch_states: dict[int, torch.Tensor],
    y_hats: list[int],
    active_scorers_list: list[tuple[str, object]],
    n_images: int,
) -> list[dict[str, dict[int, torch.Tensor]]]:
    """Compute score maps for all images in the extraction batch.

    Returns a list (length ``n_images``) of
    ``{scorer_name: {layer_idx: score_map [H_p, W_p]}}``.
    """
    result: list[dict] = [{} for _ in range(n_images)]
    for i in range(n_images):
        patch_states_i = {l: batch_patch_states[l][i: i + 1] for l in batch_patch_states}
        for scorer_name, scorer in active_scorers_list:
            result[i][scorer_name] = scorer.score_all_layers(patch_states_i, y_hats[i])  # type: ignore[attr-defined]
    return result


def _run_batch_perturbations(
    model: torch.nn.Module,
    batch_tensor: torch.Tensor,         # [N, C, H, W]
    batch_blurred: torch.Tensor,        # [N, C, H, W]
    batch_score_maps: list[dict],       # [{scorer: {layer: score_map}}] × N
    y_hats: list[int],
    active_scorer_names: list[str],
    target_layers: list[int],
    patch_size: int,
    device: str,
    eval_batch_size: int,
    image_batch_size: int,
) -> list[dict[str, dict[int, tuple]]]:
    """Run insertion+deletion curves for all images, batched across images.

    For each (scorer, layer), images are grouped into sub-batches of
    ``image_batch_size`` and processed together via
    ``insertion_deletion_curves_batch``.

    Returns a list (length N) of
    ``{scorer_name: {layer_idx: (ins_x, ins_y, ins_auc, del_x, del_y, del_auc)}}``.
    """
    N = len(y_hats)
    results: list[dict] = [{sn: {} for sn in active_scorer_names} for _ in range(N)]

    combos = [(sn, li) for sn in active_scorer_names for li in target_layers]
    pbar = tqdm(combos, desc="perturbations", unit="layer", leave=True)

    for scorer_name, layer_idx in pbar:
        pbar.set_postfix(scorer=scorer_name, layer=layer_idx)

        # Indices of images that have a score map for this (scorer, layer)
        valid_idx = [
            i for i in range(N)
            if layer_idx in batch_score_maps[i].get(scorer_name, {})
        ]
        if not valid_idx:
            continue

        # Sub-batch across images
        for sub_start in range(0, len(valid_idx), image_batch_size):
            sub = valid_idx[sub_start: sub_start + image_batch_size]

            sub_results = insertion_deletion_curves_batch(
                model=model,
                originals=batch_tensor[sub],
                blurreds=batch_blurred[sub],
                score_maps=[batch_score_maps[i][scorer_name][layer_idx] for i in sub],
                y_hats=[y_hats[i] for i in sub],
                patch_size=patch_size,
                device=device,
                eval_batch_size=eval_batch_size,
            )

            for j, img_i in enumerate(sub):
                results[img_i][scorer_name][layer_idx] = sub_results[j]

    return results


# ---------------------------------------------------------------------------
# Per-image output saving (visualization + metrics JSON)
# ---------------------------------------------------------------------------

def _evaluate_image(
    image_path: str,
    pil_img: Image.Image,
    y_hat: int,
    score_maps_by_scorer: dict[str, dict[int, torch.Tensor]],
    curves_by_scorer: dict[str, dict[int, tuple]],
    active_scorer_names: list[str],
    target_layers: list[int],
    config: AttributionConfig,
    save_outputs: bool = True,
) -> dict:
    """Save per-image outputs and collect metrics from pre-computed scores/curves.

    Args:
        score_maps_by_scorer: ``{scorer_name: {layer_idx: Tensor[H_p, W_p]}}``.
        curves_by_scorer: ``{scorer_name: {layer_idx: (ins_x, ins_y, ins_auc,
            del_x, del_y, del_auc)}}``.
        save_outputs: If False, skip all file I/O; metrics are still returned.

    Returns:
        Dict with per-scorer, per-layer AUC values for aggregate summary.
    """
    stem = Path(image_path).stem
    out_root = config.eval.output_dir

    image_metrics: dict = {"image_path": image_path, "y_hat": y_hat, "scorers": {}}

    for scorer_name in active_scorer_names:
        scorer_maps = score_maps_by_scorer.get(scorer_name, {})
        scorer_curves = curves_by_scorer.get(scorer_name, {})

        if save_outputs:
            scorer_dir = os.path.join(out_root, stem, scorer_name)
            os.makedirs(scorer_dir, exist_ok=True)

        layer_metrics: dict[int, dict] = {}
        all_score_maps_np: dict[int, np.ndarray] = {}

        for layer_idx in target_layers:
            if layer_idx not in scorer_maps or layer_idx not in scorer_curves:
                continue

            sm = scorer_maps[layer_idx].cpu().numpy()
            ins_x, ins_y, ins_auc, del_x, del_y, del_auc = scorer_curves[layer_idx]

            layer_metrics[layer_idx] = {
                "insertion_auc": ins_auc,
                "deletion_auc": del_auc,
                "insertion_x": ins_x.tolist(),
                "insertion_y": ins_y.tolist(),
                "deletion_x": del_x.tolist(),
                "deletion_y": del_y.tolist(),
            }

            if save_outputs:
                all_score_maps_np[layer_idx] = sm

                plot_heatmap(
                    pil_img, sm,
                    output_path=os.path.join(scorer_dir, f"layer_{layer_idx}_heatmap.png"),
                    title=f"{scorer_name} | layer {layer_idx} | y_hat={y_hat}",
                    colormap=config.eval.heatmap_colormap,
                    alpha=config.eval.heatmap_alpha,
                    dpi=config.eval.plot_dpi,
                )
                plot_curves(
                    ins_x, ins_y, ins_auc, del_x, del_y, del_auc,
                    output_path=os.path.join(scorer_dir, f"layer_{layer_idx}_curves.png"),
                    title=f"{scorer_name} | layer {layer_idx} | y_hat={y_hat}",
                    dpi=config.eval.plot_dpi,
                )
                plot_combined_report(
                    pil_img, sm, ins_x, ins_y, ins_auc, del_x, del_y, del_auc,
                    output_path=os.path.join(scorer_dir, f"layer_{layer_idx}_report.png"),
                    title=f"{stem} | {scorer_name} | layer {layer_idx}",
                    colormap=config.eval.heatmap_colormap,
                    alpha=config.eval.heatmap_alpha,
                    dpi=config.eval.plot_dpi,
                )

        if save_outputs and all_score_maps_np:
            plot_heatmaps_grid(
                pil_img, all_score_maps_np,
                output_path=os.path.join(scorer_dir, "all_layers_heatmaps.png"),
                title=f"{stem} | {scorer_name} scorer",
                colormap=config.eval.heatmap_colormap,
                alpha=config.eval.heatmap_alpha,
                dpi=config.eval.plot_dpi,
            )

        image_metrics["scorers"][scorer_name] = layer_metrics

    if save_outputs:
        metrics_path = os.path.join(out_root, stem, "metrics.json")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            compact = {
                "image_path": image_metrics["image_path"],
                "y_hat": image_metrics["y_hat"],
                "scorers": {
                    sname: {
                        str(lidx): {
                            "insertion_auc": ldata["insertion_auc"],
                            "deletion_auc": ldata["deletion_auc"],
                        }
                        for lidx, ldata in layers_data.items()
                    }
                    for sname, layers_data in image_metrics["scorers"].items()
                },
            }
            json.dump(compact, f, indent=2)

    return image_metrics


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def _build_summary(
    all_image_metrics: list[dict],
    target_layers: list[int],
    scorer_names: list[str],
    output_dir: str,
    plot_dpi: int,
) -> None:
    """Compute aggregate stats and save summary.json + aggregate_curves.png."""
    summary: dict = {"images": [], "aggregate": {}}

    # Compact per-image entries
    for m in all_image_metrics:
        entry = {
            "image_path": m["image_path"],
            "y_hat": m["y_hat"],
            "scorers": {
                sname: {
                    str(lidx): {
                        "insertion_auc": ldata["insertion_auc"],
                        "deletion_auc": ldata["deletion_auc"],
                    }
                    for lidx, ldata in layers_data.items()
                }
                for sname, layers_data in m["scorers"].items()
            },
        }
        summary["images"].append(entry)

    # Per-scorer, per-layer aggregate stats
    for scorer_name in scorer_names:
        summary["aggregate"][scorer_name] = {}
        for layer_idx in target_layers:
            ins_aucs, del_aucs = [], []
            for m in all_image_metrics:
                layer_data = m["scorers"].get(scorer_name, {}).get(layer_idx)
                if layer_data is not None:
                    ins_aucs.append(layer_data["insertion_auc"])
                    del_aucs.append(layer_data["deletion_auc"])
            if not ins_aucs:
                continue
            summary["aggregate"][scorer_name][str(layer_idx)] = {
                "n": len(ins_aucs),
                "insertion_auc_mean": float(np.mean(ins_aucs)),
                "insertion_auc_std": float(np.std(ins_aucs)),
                "insertion_auc_median": float(np.median(ins_aucs)),
                "deletion_auc_mean": float(np.mean(del_aucs)),
                "deletion_auc_std": float(np.std(del_aucs)),
                "deletion_auc_median": float(np.median(del_aucs)),
            }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Aggregate curve plots (one per scorer × layer)
    for scorer_name in scorer_names:
        for layer_idx in target_layers:
            ins_ys, del_ys = [], []
            ref_ins_x = ref_del_x = None
            for m in all_image_metrics:
                layer_data = m["scorers"].get(scorer_name, {}).get(layer_idx)
                if layer_data is None:
                    continue
                ins_ys.append(np.array(layer_data["insertion_y"]))
                del_ys.append(np.array(layer_data["deletion_y"]))
                if ref_ins_x is None:
                    ref_ins_x = np.array(layer_data["insertion_x"])
                    ref_del_x = np.array(layer_data["deletion_x"])

            if not ins_ys or ref_ins_x is None:
                continue

            plot_aggregate_curves(
                ins_x=ref_ins_x,
                ins_y_list=ins_ys,
                del_x=ref_del_x,
                del_y_list=del_ys,
                output_path=os.path.join(
                    output_dir, f"aggregate_{scorer_name}_layer{layer_idx}_curves.png"
                ),
                title=f"Aggregate ({len(ins_ys)} images) | {scorer_name} | layer {layer_idx}",
                dpi=plot_dpi,
            )


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lens-based patch attribution evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to attribution YAML config.")
    p.add_argument(
        "--image", dest="images", action="append", default=[],
        metavar="PATH",
        help="Image path(s) to evaluate (overrides eval.image_paths).",
    )
    p.add_argument("--image-dir", default=None, help="Override eval.image_dir.")
    p.add_argument("--output-dir", default=None, help="Override eval.output_dir.")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"],
                   help="Override eval.device.")
    p.add_argument(
        "--scorer-type", default=None,
        choices=["cls", "patch", "both", "patch_map_cls", "all"],
        help="Override eval.scorer_type.",
    )
    p.add_argument(
        "--layer", dest="layers", action="append", type=int, default=[],
        metavar="IDX",
        help="Target layer index (repeatable; overrides model.target_layers).",
    )
    p.add_argument("--num-images", type=int, default=None,
                   help="Override eval.num_images.")
    p.add_argument(
        "--num-save-images", type=int, default=None,
        metavar="N",
        help="Override eval.num_save_images: save PNGs/metrics.json for N randomly "
             "selected images; all images still contribute to aggregate stats.",
    )
    p.add_argument(
        "--perturbation-image-batch-size", type=int, default=None,
        metavar="N",
        help="Override eval.perturbation_image_batch_size: number of images batched "
             "together for insertion/deletion inference.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    config = AttributionConfig.from_yaml(args.config)

    # Apply CLI overrides
    if args.images:
        config.eval.image_paths = args.images
    if args.image_dir is not None:
        config.eval.image_dir = args.image_dir
    if args.output_dir is not None:
        config.eval.output_dir = args.output_dir
    if args.device is not None:
        config.eval.device = args.device
    if args.scorer_type is not None:
        config.eval.scorer_type = args.scorer_type
    if args.layers:
        config.model.target_layers = args.layers
    if args.num_images is not None:
        config.eval.num_images = args.num_images
    if args.num_save_images is not None:
        config.eval.num_save_images = args.num_save_images
    if args.perturbation_image_batch_size is not None:
        config.eval.perturbation_image_batch_size = args.perturbation_image_batch_size

    # Seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Device
    device = config.eval.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warning] CUDA not available, falling back to CPU.")
        device = "cpu"

    scorer_type = config.eval.scorer_type

    # Resolve target layers
    target_layers = _resolve_target_layers(config, scorer_type)
    print(f"Target layers: {target_layers}")

    # Build model wrapper (patch_mode=True for all extraction)
    print("Loading model...")
    wrapper = _build_wrapper(config, device)
    # Limit hooks to only the target layers we need
    wrapper.target_layers = target_layers
    wrapper._register_hooks()

    patch_size = _infer_patch_size(wrapper)
    print(f"Patch size: {patch_size}px | Grid: {wrapper.patch_grid_size}")

    # Build scorers
    cls_scorer: CLSLensScorer | None = None
    patch_scorer: PatchLensScorer | None = None
    patch_map_cls_scorer: PatchMapCLSLensScorer | None = None

    if scorer_type in ("cls", "both", "all"):
        print("Loading CLS lens scorer...")
        cls_scorer = CLSLensScorer(
            cls_lens_dir=config.lens.cls_lens_dir,
            means_path=config.lens.means_path,
            target_layers=target_layers,
            device=device,
        )

    if scorer_type in ("patch", "both", "all"):
        print("Loading patch lens scorer...")
        patch_scorer = PatchLensScorer(
            patch_lens_dir=config.lens.patch_lens_dir,
            patch_neighbor_size=config.lens.patch_neighbor_size,
            patch_border=config.lens.patch_border,
            target_layers=target_layers,
            device=device,
        )

    if scorer_type in ("patch_map_cls", "all"):
        print("Loading patch map CLS lens scorer...")
        patch_map_cls_scorer = PatchMapCLSLensScorer(
            cls_lens_dir=config.lens.cls_lens_dir,
            patch_map_dir=config.lens.patch_map_dir,
            target_layers=target_layers,
            device=device,
        )

    # Collect images
    image_paths = _collect_images(config.eval, config.seed)
    print(f"Evaluating {len(image_paths)} image(s)...")

    # Determine which images will have per-image outputs (PNGs + metrics.json) saved
    n_save = config.eval.num_save_images
    if n_save is None or n_save >= len(image_paths):
        save_paths: set[str] = set(image_paths)
    else:
        rng = random.Random(config.seed)
        save_paths = set(rng.sample(image_paths, n_save))
    print(
        f"Per-image outputs (PNGs/JSON) will be saved for "
        f"{len(save_paths)}/{len(image_paths)} image(s)."
    )

    # Build ordered list of active scorers (name + object) for batch helpers
    active_scorers_list: list[tuple[str, object]] = []
    if cls_scorer is not None:
        active_scorers_list.append(("cls", cls_scorer))
    if patch_scorer is not None:
        active_scorers_list.append(("patch", patch_scorer))
    if patch_map_cls_scorer is not None:
        active_scorers_list.append(("patch_map_cls", patch_map_cls_scorer))

    active_scorer_names: list[str] = [name for name, _ in active_scorers_list]

    all_image_metrics: list[dict] = []
    transform = wrapper.get_transform()
    extr_bs = config.eval.extraction_batch_size
    n_total = len(image_paths)

    # Process images in extraction batches to amortise the feature-extraction
    # forward pass; insertion/deletion curves are still per-image.
    processed = 0
    batch_pbar = tqdm(range(0, n_total, extr_bs), desc="extraction batches", unit="batch")
    for batch_start in batch_pbar:
        batch_paths = image_paths[batch_start: batch_start + extr_bs]

        # Load and transform images
        pil_imgs: list[Image.Image] = []
        tensors: list[torch.Tensor] = []
        valid_paths: list[str] = []
        for path in batch_paths:
            try:
                pil = Image.open(path).convert("RGB")
                t = transform(pil).to(device)
                pil_imgs.append(pil)
                tensors.append(t)
                valid_paths.append(path)
            except Exception as e:
                print(f"  [load error] {path}: {e}")

        if not tensors:
            continue

        # Single extraction forward pass for the whole batch (hooks active)
        batch_tensor = torch.stack(tensors, dim=0)   # [B, C, H, W]
        with torch.no_grad():
            batch_patch_states, batch_logits = wrapper.extract_patches(batch_tensor)

        # Detach hooks before per-image perturbation loops so that the many
        # model(batch_tensor) calls in insertion/deletion don't capture and
        # store intermediate tensors for all 24 target layers.
        wrapper._remove_hooks()

        # Blurred baselines for the whole batch (cheap, no model needed)
        batch_blurred = apply_gaussian_blur(
            batch_tensor,
            kernel_size=config.eval.blur_kernel_size,
            sigma=config.eval.blur_sigma,
        )

        n_imgs = len(valid_paths)
        y_hats = [int(batch_logits[i].argmax().item()) for i in range(n_imgs)]

        # Compute score maps for all images (fast: no model call)
        batch_score_maps = _compute_batch_score_maps(
            batch_patch_states, y_hats, active_scorers_list, n_imgs
        )

        # Batch insertion/deletion across images (the slow part)
        try:
            batch_curves = _run_batch_perturbations(
                model=wrapper.model,
                batch_tensor=batch_tensor,
                batch_blurred=batch_blurred,
                batch_score_maps=batch_score_maps,
                y_hats=y_hats,
                active_scorer_names=active_scorer_names,
                target_layers=target_layers,
                patch_size=patch_size,
                device=device,
                eval_batch_size=config.eval.perturbation_batch_size,
                image_batch_size=config.eval.perturbation_image_batch_size,
            )
        except Exception as e:
            print(f"  [batch perturbation error] {e}")
            wrapper._register_hooks()
            continue

        # Per-image output saving and metrics collection
        for i, (image_path, pil_img) in enumerate(zip(valid_paths, pil_imgs)):
            processed += 1
            tqdm.write(f"[{processed}/{n_total}] {image_path}")

            try:
                metrics = _evaluate_image(
                    image_path=image_path,
                    pil_img=pil_img,
                    y_hat=y_hats[i],
                    score_maps_by_scorer=batch_score_maps[i],
                    curves_by_scorer=batch_curves[i],
                    active_scorer_names=active_scorer_names,
                    target_layers=target_layers,
                    config=config,
                    save_outputs=image_path in save_paths,
                )
                all_image_metrics.append(metrics)

                for sname, layers_data in metrics["scorers"].items():
                    for lidx, ldata in sorted(layers_data.items()):
                        tqdm.write(
                            f"  {sname} layer {lidx}: "
                            f"insertion={ldata['insertion_auc']:.4f}  "
                            f"deletion={ldata['deletion_auc']:.4f}"
                        )
            except Exception as e:
                print(f"  [error] {e}")
                continue

        # Re-attach hooks for the next batch's extraction pass
        wrapper._register_hooks()

    if not all_image_metrics:
        print("No images successfully evaluated.")
        return

    # Save aggregate summary and curves
    print("Saving aggregate summary...")
    _build_summary(
        all_image_metrics=all_image_metrics,
        target_layers=target_layers,
        scorer_names=active_scorer_names,
        output_dir=config.eval.output_dir,
        plot_dpi=config.eval.plot_dpi,
    )

    print(f"\nDone. Results saved to: {config.eval.output_dir}")
    print(f"  summary.json — aggregate AUC statistics")
    print(f"  aggregate_*_curves.png — mean ± std curves per scorer × layer")


if __name__ == "__main__":
    main()
