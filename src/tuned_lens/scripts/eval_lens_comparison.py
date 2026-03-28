"""Compare tuned-lens vs logit-lens accuracy per layer on ImageNet val.

Terminology
-----------
- **Tuned lens**: learned affine/MLP probe trained on intermediate CLS tokens.
- **Logit lens**: the frozen model's own classification head (norm → fc_norm → head)
  applied directly to intermediate CLS tokens — no learning, just the head re-used.

Two accuracy metrics are plotted side by side for each layer:
  1. vs GT    : argmax(lens logits) == ground-truth ImageNet label
  2. vs y_hat : argmax(lens logits) == argmax(final model logits)

Usage
-----
  python -m tuned_lens.scripts.eval_lens_comparison \\
      --lens-dir outputs/affine_kld/best_lenses \\
      --imagenet-root /data/imagenet/extracted \\
      --model-name vit_large_patch14_clip_224.openai_ft_in1k

  # Or load model/data settings from a saved config
  python -m tuned_lens.scripts.eval_lens_comparison \\
      --lens-dir outputs/affine_kld/best_lenses \\
      --config outputs/affine_kld/config.yaml
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from tuned_lens.config import ModelConfig
from tuned_lens.lens import AffineLens, MLPLens, LensBank, BaseLens
from tuned_lens.model import VisionModelWrapper


# ---------------------------------------------------------------------------
# Lens loading (reused from eval_lens.py)
# ---------------------------------------------------------------------------

def discover_lenses(lens_dir: str) -> dict[int, str]:
    """Return {layer_idx: path} for all layer_*.pt files in lens_dir."""
    pattern = re.compile(r"^layer_(\d+)\.pt$")
    lenses = {}
    for fname in os.listdir(lens_dir):
        m = pattern.match(fname)
        if m:
            lenses[int(m.group(1))] = os.path.join(lens_dir, fname)
    if not lenses:
        raise FileNotFoundError(f"No layer_*.pt files found in {lens_dir}")
    return lenses


def load_lens_from_file(path: str) -> BaseLens:
    """Load a lens from a .pt file, inferring its type from the state_dict keys."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload["state_dict"]

    if any(k.startswith("net.") for k in state_dict):
        linear_keys = [k for k in state_dict if k.endswith(".weight")]
        in_features = state_dict[linear_keys[0]].shape[1]
        out_features = state_dict[linear_keys[-1]].shape[0]
        hidden_dim = state_dict[linear_keys[0]].shape[0]
        num_layers = len(linear_keys)
        lens = MLPLens(in_features, out_features, hidden_dim=hidden_dim, num_layers=num_layers)
    else:
        in_features = state_dict["linear.weight"].shape[1]
        out_features = state_dict["linear.weight"].shape[0]
        bias = "linear.bias" in state_dict
        lens = AffineLens(in_features, out_features, bias=bias)

    lens.load_state_dict(state_dict)
    lens.eval()
    return lens


# ---------------------------------------------------------------------------
# Logit lens: model's own head applied to intermediate CLS tokens
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_logit_lens(wrapper: VisionModelWrapper, cls_tokens: torch.Tensor) -> torch.Tensor:
    """Apply the frozen model's final head to intermediate CLS tokens.

    Replicates the post-block path: norm → fc_norm (if present) → head.

    Args:
        wrapper: VisionModelWrapper with the loaded timm model.
        cls_tokens: [B, d_model] intermediate CLS token representations.

    Returns:
        logits: [B, num_classes]
    """
    model = wrapper.model
    x = model.norm(cls_tokens)
    fc_norm = getattr(model, "fc_norm", None)
    if fc_norm is not None:
        x = fc_norm(x)
    return model.head(x)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class LayerResult:
    tuned_correct_gt: int = 0
    tuned_correct_yhat: int = 0
    logit_correct_gt: int = 0
    logit_correct_yhat: int = 0


@torch.no_grad()
def evaluate(
    wrapper: VisionModelWrapper,
    lenses: dict[int, BaseLens],
    val_loader: DataLoader,
    device: str,
) -> tuple[dict[int, LayerResult], int]:
    """Evaluate both tuned lens and logit lens on the full val set.

    Returns:
        results: {layer_idx: LayerResult} with per-layer counts.
        total: total number of images processed.
    """
    layer_indices = sorted(lenses.keys())
    results = {i: LayerResult() for i in layer_indices}
    total = 0

    for batch_idx, (images, gt_labels) in enumerate(val_loader):
        images = images.to(device)
        gt_labels = gt_labels.to(device)

        hidden_states, final_logits = wrapper.extract(images)
        model_preds = final_logits.argmax(dim=-1)  # y_hat: [B]

        for layer_idx in layer_indices:
            cls_token = hidden_states[layer_idx].to(device)

            # --- Tuned lens ---
            tuned_logits = lenses[layer_idx](cls_token)
            tuned_preds = tuned_logits.argmax(dim=-1)

            # --- Logit lens ---
            logit_logits = apply_logit_lens(wrapper, cls_token)
            logit_preds = logit_logits.argmax(dim=-1)

            r = results[layer_idx]
            r.tuned_correct_gt   += (tuned_preds == gt_labels).sum().item()
            r.tuned_correct_yhat += (tuned_preds == model_preds).sum().item()
            r.logit_correct_gt   += (logit_preds == gt_labels).sum().item()
            r.logit_correct_yhat += (logit_preds == model_preds).sum().item()

        total += images.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"  [{batch_idx + 1}/{len(val_loader)}] processed {total} images")

    return results, total


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    results: dict[int, LayerResult],
    total: int,
    output_path: str,
    model_name: str,
) -> None:
    """Two side-by-side bar charts: accuracy vs GT and vs y_hat."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    layers = sorted(results.keys())
    x = np.arange(len(layers))
    width = 0.35

    def pct(n: int) -> float:
        return n / total * 100

    tuned_gt   = [pct(results[i].tuned_correct_gt)   for i in layers]
    logit_gt   = [pct(results[i].logit_correct_gt)   for i in layers]
    tuned_yhat = [pct(results[i].tuned_correct_yhat) for i in layers]
    logit_yhat = [pct(results[i].logit_correct_yhat) for i in layers]

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(layers) * 1.1), 6), sharey=False)
    fig.suptitle(f"Tuned lens vs Logit lens — {model_name}", fontsize=12)

    for ax, tuned_vals, logit_vals, title, ylabel in [
        (axes[0], tuned_gt,   logit_gt,   "Accuracy vs Ground Truth",      "Accuracy (%)"),
        (axes[1], tuned_yhat, logit_yhat, "Accuracy vs Model Prediction (y_hat)", "Accuracy (%)"),
    ]:
        bars_tuned = ax.bar(x - width / 2, tuned_vals, width, label="Tuned lens",
                            color="steelblue", edgecolor="white", linewidth=0.5)
        bars_logit = ax.bar(x + width / 2, logit_vals, width, label="Logit lens",
                            color="darkorange", edgecolor="white", linewidth=0.5)

        # Value labels on bars (only if enough space)
        if len(layers) <= 16:
            for bar, v in zip(bars_tuned, tuned_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)
            for bar, v in zip(bars_logit, logit_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Layer index", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(layers, fontsize=8)
        all_vals = tuned_vals + logit_vals
        ax.set_ylim(0, min(100, max(all_vals) * 1.18) if all_vals else 100)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare tuned-lens vs logit-lens accuracy per layer on ImageNet val.\n"
            "Produces a two-panel plot: accuracy vs ground truth and vs model prediction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lens-dir", type=str, required=True,
                        help="Directory containing layer_*.pt files (tuned lenses)")
    parser.add_argument("--imagenet-root", type=str, default=None,
                        help="Path to extracted ImageNet (with val/ subdir)")
    parser.add_argument("--model-name", type=str, default=None,
                        help="timm model name")
    parser.add_argument("--weights-path", type=str, default=None,
                        help="Local model weights (.safetensors or .pt)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml saved during training (sets model/data defaults)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None,
                        help="Output plot path (default: <lens-dir>/comparison.png)")
    args = parser.parse_args()

    model_name = args.model_name
    imagenet_root = args.imagenet_root
    weights_path = args.weights_path

    if args.config is not None:
        from tuned_lens.config import TunedLensConfig
        cfg = TunedLensConfig.from_yaml(args.config)
        if model_name is None:
            model_name = cfg.model.model_name
        if imagenet_root is None:
            imagenet_root = cfg.data.imagenet_root
        if weights_path is None:
            weights_path = cfg.model.weights_path

    if model_name is None:
        parser.error("--model-name is required (or provide --config)")
    if not imagenet_root:
        parser.error("--imagenet-root is required (or provide --config with imagenet_root set)")

    output_path = args.output or os.path.join(args.lens_dir, "comparison.png")

    # Discover tuned lenses
    print(f"Loading tuned lenses from {args.lens_dir} ...")
    lens_paths = discover_lenses(args.lens_dir)
    target_layers = sorted(lens_paths.keys())
    print(f"Found lenses for layers: {target_layers}")

    # Load model
    print(f"Loading model: {model_name} ...")
    model_config = ModelConfig(
        model_name=model_name,
        pretrained=(weights_path is None),
        weights_path=weights_path,
        target_layers=target_layers,
        freeze_model=True,
    )
    wrapper = VisionModelWrapper(model_config, device=args.device)
    print(f"  d_model={wrapper.d_model}, num_classes={wrapper.num_classes}")

    # Load tuned lenses onto device
    lenses: dict[int, BaseLens] = {}
    for layer_idx, path in lens_paths.items():
        lenses[layer_idx] = load_lens_from_file(path).to(args.device)
    print(f"Loaded {len(lenses)} tuned lenses")

    # Val dataloader
    val_dir = os.path.join(imagenet_root, "val")
    val_dataset = ImageFolder(val_dir, transform=wrapper.get_transform())
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Val set: {len(val_dataset)} images, {len(val_loader)} batches")

    # Evaluate
    print("\nEvaluating ...")
    results, total = evaluate(wrapper, lenses, val_loader, args.device)

    # Print table
    print(f"\n{'Layer':>6}  {'Tuned vs GT':>12}  {'Logit vs GT':>12}  "
          f"{'Tuned vs ŷ':>12}  {'Logit vs ŷ':>12}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")
    for layer_idx in sorted(results):
        r = results[layer_idx]
        print(
            f"  {layer_idx:>6}  "
            f"{r.tuned_correct_gt / total * 100:>11.2f}%  "
            f"{r.logit_correct_gt / total * 100:>11.2f}%  "
            f"{r.tuned_correct_yhat / total * 100:>11.2f}%  "
            f"{r.logit_correct_yhat / total * 100:>11.2f}%"
        )

    # Plot
    plot_comparison(results, total, output_path, model_name)
    wrapper.cleanup()


if __name__ == "__main__":
    main()
