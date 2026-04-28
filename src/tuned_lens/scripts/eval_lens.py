"""Evaluate trained lenses on the ImageNet validation set and plot per-layer accuracy.

Accuracy is defined as the fraction of images where the lens prediction (argmax of
lens logits) matches the model's own final-layer prediction (argmax of final logits).
This measures how faithfully each layer's lens mimics the full model's decisions.
"""

from __future__ import annotations

import argparse
import os
import re

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from tuned_lens.config import ModelConfig, DataConfig
from tuned_lens.lens import AffineLens, MLPLens, LensBank, BaseLens
from tuned_lens.model import VisionModelWrapper


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

    # Infer architecture from keys
    if any(k.startswith("net.") for k in state_dict):
        # MLPLens — infer dims from weights
        # Find first and last Linear layers
        linear_keys = [k for k in state_dict if k.endswith(".weight")]
        in_features = state_dict[linear_keys[0]].shape[1]
        out_features = state_dict[linear_keys[-1]].shape[0]
        hidden_dim = state_dict[linear_keys[0]].shape[0]
        num_layers = len(linear_keys)
        lens = MLPLens(in_features, out_features, hidden_dim=hidden_dim, num_layers=num_layers)
    else:
        # AffineLens
        in_features = state_dict["linear.weight"].shape[1]
        out_features = state_dict["linear.weight"].shape[0]
        bias = "linear.bias" in state_dict
        lens = AffineLens(in_features, out_features, bias=bias)

    lens.load_state_dict(state_dict)
    lens.eval()
    return lens


@torch.no_grad()
def evaluate(
    wrapper: VisionModelWrapper,
    lenses: dict[int, BaseLens],
    val_loader: DataLoader,
    device: str,
) -> dict[int, float]:
    """Compute per-layer accuracy over the full val set.

    Accuracy = fraction of images where lens_i argmax == final model argmax.
    """
    layer_indices = sorted(lenses.keys())
    correct = {i: 0 for i in layer_indices}
    total = 0

    for batch_idx, (images, _) in enumerate(val_loader):
        images = images.to(device)
        hidden_states, target_logits = wrapper.extract(images)
        model_preds = target_logits.argmax(dim=-1)  # [B]

        for layer_idx in layer_indices:
            cls_token = hidden_states[layer_idx].to(device)
            lens_embedding = lenses[layer_idx](cls_token)            # [B, d_model]
            lens_logits = wrapper.apply_head(lens_embedding)         # [B, num_classes]
            lens_preds = lens_logits.argmax(dim=-1)  # [B]
            correct[layer_idx] += (lens_preds == model_preds).sum().item()

        total += images.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"  [{batch_idx + 1}/{len(val_loader)}] processed {total} images")

    return {i: correct[i] / total for i in layer_indices}


def plot_accuracy(
    accuracies: dict[int, float],
    output_path: str,
    model_name: str,
) -> None:
    """Bar chart of per-layer lens accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(accuracies.keys())
    accs = [accuracies[i] * 100 for i in layers]

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.6), 5))
    bars = ax.bar(layers, accs, color="steelblue", edgecolor="white", linewidth=0.5)

    # Annotate bars with value
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel("Layer index", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"Tuned lens accuracy per layer\n"
        f"(lens prediction vs. final model prediction)\n"
        f"{model_name}",
        fontsize=11,
    )
    ax.set_xticks(layers)
    ax.set_ylim(0, min(100, max(accs) * 1.15))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate tuned lenses on ImageNet val and plot per-layer accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Evaluate best lenses from a training run\n"
            "  python -m tuned_lens.scripts.eval_lens \\\n"
            "    --lens-dir outputs/dry_run/best_lenses \\\n"
            "    --imagenet-root /data/imagenet/extracted \\\n"
            "    --model-name vit_large_patch14_clip_224.openai_ft_in1k\n\n"
            "  # Use a saved config from the run\n"
            "  python -m tuned_lens.scripts.eval_lens \\\n"
            "    --lens-dir outputs/dry_run/best_lenses \\\n"
            "    --config outputs/dry_run/config.yaml"
        ),
    )
    parser.add_argument("--lens-dir", type=str, required=True,
                        help="Directory containing layer_*.pt files")
    parser.add_argument("--imagenet-root", type=str, default=None,
                        help="Path to extracted ImageNet (with val/ subdir)")
    parser.add_argument("--model-name", type=str, default=None,
                        help="timm model name")
    parser.add_argument("--weights-path", type=str, default=None,
                        help="Local model weights (.safetensors or .pt)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml saved during training (sets model and data defaults)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for output plot (default: <lens-dir>/accuracy.png)")
    args = parser.parse_args()

    # Load defaults from config if provided
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

    output_path = args.output or os.path.join(args.lens_dir, "accuracy.png")

    # Discover lens files
    print(f"Loading lenses from {args.lens_dir} ...")
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

    # Load lenses
    lenses: dict[int, BaseLens] = {}
    for layer_idx, path in lens_paths.items():
        lenses[layer_idx] = load_lens_from_file(path).to(args.device)
    print(f"Loaded {len(lenses)} lenses")

    # Build val dataloader
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
    accuracies = evaluate(wrapper, lenses, val_loader, args.device)

    # Print results
    print("\nPer-layer accuracy (lens vs. final model prediction):")
    print(f"  {'Layer':>6}  {'Accuracy':>10}")
    print(f"  {'-'*6}  {'-'*10}")
    for layer_idx in sorted(accuracies):
        print(f"  {layer_idx:>6}  {accuracies[layer_idx]*100:>9.2f}%")

    # Plot
    plot_accuracy(accuracies, output_path, model_name)
    wrapper.cleanup()


if __name__ == "__main__":
    main()
