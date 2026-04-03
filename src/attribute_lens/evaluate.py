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

from tuned_lens.config import ModelConfig
from tuned_lens.model import VisionModelWrapper

from .config import AttributionConfig, EvalSection
from .scorer import CLSLensScorer, PatchLensScorer, discover_lens_files
from .metrics import apply_gaussian_blur, insertion_curve, deletion_curve
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

    if scorer_type in ("cls", "both") and config.lens.cls_lens_dir:
        layers |= set(discover_lens_files(config.lens.cls_lens_dir).keys())

    if scorer_type in ("patch", "both") and config.lens.patch_lens_dir:
        available = set(discover_lens_files(config.lens.patch_lens_dir).keys())
        if layers:
            layers &= available   # intersection when both scorers active
        else:
            layers = available

    if not layers:
        raise ValueError(
            "No lens checkpoint files found. "
            "Check lens.cls_lens_dir / lens.patch_lens_dir in config."
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


def _evaluate_image(
    image_path: str,
    pil_img: Image.Image,
    image_tensor: torch.Tensor,
    patch_states: dict[int, torch.Tensor],
    y_hat: int,
    blurred: torch.Tensor,
    cls_scorer: CLSLensScorer | None,
    patch_scorer: PatchLensScorer | None,
    target_layers: list[int],
    config: AttributionConfig,
    patch_size: int,
    device: str,
    model: torch.nn.Module,
) -> dict:
    """Run full attribution pipeline on a single image.

    Pre-computed inputs (``patch_states``, ``y_hat``, ``blurred``) are passed
    in so that the caller can amortise the extraction forward pass across a
    batch of images.

    Returns a dict with per-scorer, per-layer AUC values.
    """
    stem = Path(image_path).stem
    out_root = config.eval.output_dir

    image_metrics: dict = {
        "image_path": image_path,
        "y_hat": y_hat,
        "scorers": {},
    }

    active_scorers: list[tuple[str, CLSLensScorer | PatchLensScorer]] = []
    if cls_scorer is not None:
        active_scorers.append(("cls", cls_scorer))
    if patch_scorer is not None:
        active_scorers.append(("patch", patch_scorer))

    for scorer_name, scorer in active_scorers:
        scorer_dir = os.path.join(out_root, stem, scorer_name)
        os.makedirs(scorer_dir, exist_ok=True)

        score_maps = scorer.score_all_layers(patch_states, y_hat)

        layer_metrics: dict[int, dict] = {}
        all_score_maps_np: dict[int, np.ndarray] = {}

        for layer_idx in target_layers:
            if layer_idx not in score_maps:
                continue
            sm = score_maps[layer_idx].cpu().numpy()
            all_score_maps_np[layer_idx] = sm

            # Insertion curve
            ins_x, ins_y, ins_auc = insertion_curve(
                model=model,
                original=image_tensor,
                blurred=blurred,
                score_map=score_maps[layer_idx],
                y_hat=y_hat,
                patch_size=patch_size,
                device=device,
                eval_batch_size=config.eval.perturbation_batch_size,
            )

            # Deletion curve
            del_x, del_y, del_auc = deletion_curve(
                model=model,
                original=image_tensor,
                blurred=blurred,
                score_map=score_maps[layer_idx],
                y_hat=y_hat,
                patch_size=patch_size,
                device=device,
                eval_batch_size=config.eval.perturbation_batch_size,
            )

            layer_metrics[layer_idx] = {
                "insertion_auc": ins_auc,
                "deletion_auc": del_auc,
                "insertion_x": ins_x.tolist(),
                "insertion_y": ins_y.tolist(),
                "deletion_x": del_x.tolist(),
                "deletion_y": del_y.tolist(),
            }

            # Save per-layer heatmap
            plot_heatmap(
                pil_img, sm,
                output_path=os.path.join(scorer_dir, f"layer_{layer_idx}_heatmap.png"),
                title=f"{scorer_name} | layer {layer_idx} | y_hat={y_hat}",
                colormap=config.eval.heatmap_colormap,
                alpha=config.eval.heatmap_alpha,
                dpi=config.eval.plot_dpi,
            )

            # Save per-layer insertion/deletion curves
            plot_curves(
                ins_x, ins_y, ins_auc,
                del_x, del_y, del_auc,
                output_path=os.path.join(scorer_dir, f"layer_{layer_idx}_curves.png"),
                title=f"{scorer_name} | layer {layer_idx} | y_hat={y_hat}",
                dpi=config.eval.plot_dpi,
            )

            # Save combined 4-panel report
            plot_combined_report(
                pil_img, sm,
                ins_x, ins_y, ins_auc,
                del_x, del_y, del_auc,
                output_path=os.path.join(scorer_dir, f"layer_{layer_idx}_report.png"),
                title=f"{stem} | {scorer_name} | layer {layer_idx}",
                colormap=config.eval.heatmap_colormap,
                alpha=config.eval.heatmap_alpha,
                dpi=config.eval.plot_dpi,
            )

        # Save multi-layer heatmap grid
        if all_score_maps_np:
            plot_heatmaps_grid(
                pil_img,
                all_score_maps_np,
                output_path=os.path.join(scorer_dir, "all_layers_heatmaps.png"),
                title=f"{stem} | {scorer_name} scorer",
                colormap=config.eval.heatmap_colormap,
                alpha=config.eval.heatmap_alpha,
                dpi=config.eval.plot_dpi,
            )

        image_metrics["scorers"][scorer_name] = layer_metrics

    # Save per-image metrics JSON
    metrics_path = os.path.join(out_root, stem, "metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        # Exclude large curve arrays from JSON for compactness
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
    p.add_argument("--scorer-type", default=None, choices=["cls", "patch", "both"],
                   help="Override eval.scorer_type.")
    p.add_argument(
        "--layer", dest="layers", action="append", type=int, default=[],
        metavar="IDX",
        help="Target layer index (repeatable; overrides model.target_layers).",
    )
    p.add_argument("--num-images", type=int, default=None,
                   help="Override eval.num_images.")
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

    if scorer_type in ("cls", "both"):
        print("Loading CLS lens scorer...")
        cls_scorer = CLSLensScorer(
            cls_lens_dir=config.lens.cls_lens_dir,
            means_path=config.lens.means_path,
            target_layers=target_layers,
            device=device,
        )

    if scorer_type in ("patch", "both"):
        print("Loading patch lens scorer...")
        patch_scorer = PatchLensScorer(
            patch_lens_dir=config.lens.patch_lens_dir,
            patch_neighbor_size=config.lens.patch_neighbor_size,
            patch_border=config.lens.patch_border,
            target_layers=target_layers,
            device=device,
        )

    # Collect images
    image_paths = _collect_images(config.eval, config.seed)
    print(f"Evaluating {len(image_paths)} image(s)...")

    # Determine which scorer names are active (for summary)
    active_scorer_names: list[str] = []
    if cls_scorer is not None:
        active_scorer_names.append("cls")
    if patch_scorer is not None:
        active_scorer_names.append("patch")

    all_image_metrics: list[dict] = []
    transform = wrapper.get_transform()
    extr_bs = config.eval.extraction_batch_size
    n_total = len(image_paths)

    # Process images in extraction batches to amortise the feature-extraction
    # forward pass; insertion/deletion curves are still per-image.
    processed = 0
    for batch_start in range(0, n_total, extr_bs):
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

        for i, (image_path, pil_img) in enumerate(zip(valid_paths, pil_imgs)):
            processed += 1
            print(f"[{processed}/{n_total}] {image_path}")

            # Slice this image's data out of the batch
            image_tensor = batch_tensor[i: i + 1]                        # [1, C, H, W]
            patch_states_i = {
                l: batch_patch_states[l][i: i + 1]
                for l in batch_patch_states
            }
            y_hat = int(batch_logits[i].argmax().item())
            blurred_i = batch_blurred[i: i + 1]                          # [1, C, H, W]

            try:
                metrics = _evaluate_image(
                    image_path=image_path,
                    pil_img=pil_img,
                    image_tensor=image_tensor,
                    patch_states=patch_states_i,
                    y_hat=y_hat,
                    blurred=blurred_i,
                    cls_scorer=cls_scorer,
                    patch_scorer=patch_scorer,
                    target_layers=target_layers,
                    config=config,
                    patch_size=patch_size,
                    device=device,
                    model=wrapper.model,
                )
                all_image_metrics.append(metrics)

                for sname, layers_data in metrics["scorers"].items():
                    for lidx, ldata in sorted(layers_data.items()):
                        print(
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
