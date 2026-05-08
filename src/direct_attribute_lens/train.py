"""Training script for direct_attribute_lens.

Two phases:
  1. Pre-computation  — run backbone once on all training images, save patch embeddings to disk.
  2. Training         — load cached embeddings chunk by chunk, train a linear lens on them.

The backbone is never touched during phase 2, so training is pure GPU linear algebra.

Usage:
    python -m direct_attribute_lens.train --config configs/direct_lens/default.yaml
    python -m direct_attribute_lens.train --config configs/direct_lens/default.yaml \
        --target-layer 12 --output-dir outputs/direct_lens/layer12
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from .config import DirectLensConfig, TrainingConfig
from .lens import EmbeddingLens, DirectLens
from .loss import get_loss_fn
from .precompute import build_model_wrapper, precompute_train, precompute_val


# ── Core training functions ───────────────────────────────────────────────────

def train_one_chunk(
    lens: nn.Module,
    apply_head_fn: Callable[[Tensor], Tensor] | None,
    patch_embs: Tensor,    # [N_tokens, d_model] float16, on GPU
    img_targets: Tensor,   # [N_images, num_classes] float16, on GPU
    image_ids: Tensor,     # [N_tokens] int32, on GPU
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    lens_batch_size: int,
    grad_clip_norm: float,
) -> float:
    """Train the lens on one pre-computed chunk.

    Shuffles all tokens in the chunk at GPU level (torch.randperm), then iterates
    over mini-batches. Cross-image token mixing is automatic because all tokens
    from all images in the chunk are shuffled together.

    Returns the mean loss over all mini-batch steps.
    """
    lens.train()
    N_tokens = patch_embs.shape[0]
    perm = torch.randperm(N_tokens, device=patch_embs.device)

    total_loss = 0.0
    n_steps = 0

    for start in range(0, N_tokens, lens_batch_size):
        idx = perm[start : start + lens_batch_size]

        x = patch_embs[idx].float()              # float16 → float32
        y = img_targets[image_ids[idx]].float()  # image-level target for each token

        if apply_head_fn is not None:
            logits = apply_head_fn(lens(x))      # embedding type: lens then head
        else:
            logits = lens(x)                     # direct type: lens outputs logits

        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(lens.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        n_steps += 1

    return total_loss / n_steps if n_steps > 0 else 0.0


@torch.no_grad()
def validate(
    lens: nn.Module,
    apply_head_fn: Callable[[Tensor], Tensor] | None,
    val_patch_embs: Tensor,   # [N_val_tokens, d_model] float16, on GPU
    val_img_targets: Tensor,  # [N_val_images, num_classes] float16, on GPU
    val_image_ids: Tensor,    # [N_val_tokens] int32, on GPU
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    lens_batch_size: int,
) -> float:
    """Evaluate the lens on the pre-computed validation set. Returns mean loss."""
    lens.eval()
    N_tokens = val_patch_embs.shape[0]

    total_loss = 0.0
    n_steps = 0

    for start in range(0, N_tokens, lens_batch_size):
        end = min(start + lens_batch_size, N_tokens)
        idx = torch.arange(start, end, device=val_patch_embs.device)

        x = val_patch_embs[idx].float()
        y = val_img_targets[val_image_ids[idx]].float()

        if apply_head_fn is not None:
            logits = apply_head_fn(lens(x))
        else:
            logits = lens(x)

        total_loss += loss_fn(logits, y).item()
        n_steps += 1

    return total_loss / n_steps if n_steps > 0 else 0.0


# ── Optimizer factory ─────────────────────────────────────────────────────────

def build_optimizer(
    lens: nn.Module, config: TrainingConfig
) -> torch.optim.Optimizer:
    params = lens.parameters()
    if config.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(params, lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)
    return torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a direct_attribute_lens")
    parser.add_argument("--config", default="configs/direct_lens/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--target-layer", type=int, default=None,
                        help="Override config.model.target_layer")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override config.output_dir")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override config.training.lr")
    parser.add_argument("--num-epochs", type=int, default=None,
                        help="Override config.training.num_epochs")
    parser.add_argument("--lens-type", choices=["embedding", "direct"], default=None,
                        help="Override config.lens.lens_type")
    parser.add_argument("--loss-type", choices=["kld", "ce", "combined"], default=None,
                        help="Override config.training.loss_type")
    args = parser.parse_args()

    # ── Load and patch config ─────────────────────────────────────────────────
    config = DirectLensConfig.from_yaml(args.config)
    if args.target_layer is not None:
        config.model.target_layer = args.target_layer
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.lr is not None:
        config.training.lr = args.lr
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    if args.lens_type is not None:
        config.lens.lens_type = args.lens_type
    if args.loss_type is not None:
        config.training.loss_type = args.loss_type

    torch.manual_seed(config.seed)
    random.seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: layer={config.model.target_layer}, lens_type={config.lens.lens_type}, "
          f"loss={config.training.loss_type}, lr={config.training.lr}")

    # ── Phase 1: Pre-computation ──────────────────────────────────────────────
    print("\n=== Phase 1: Pre-computation ===")
    model_wrapper = build_model_wrapper(config, device)
    d_model = model_wrapper.d_model
    num_classes = model_wrapper.num_classes

    chunk_paths = precompute_train(model_wrapper, config, device)
    val_patch_embs, val_img_targets, val_image_ids = precompute_val(model_wrapper, config, device)
    # Val tensors stay on GPU for the entire training run (they're small).
    val_patch_embs = val_patch_embs.to(device)
    val_img_targets = val_img_targets.to(device)
    val_image_ids = val_image_ids.to(device)

    print(f"\nPre-computation done: {len(chunk_paths)} training chunk(s), "
          f"{val_patch_embs.shape[0]} val tokens.")

    # ── Build lens ────────────────────────────────────────────────────────────
    if config.lens.lens_type == "direct":
        head_w, head_b = model_wrapper.get_head_parameters()
        lens: nn.Module = DirectLens(d_model, num_classes, config.lens.bias, head_w, head_b)
        apply_head_fn = None
    else:
        lens = EmbeddingLens(d_model, config.lens.bias)
        apply_head_fn = model_wrapper.apply_head

    lens = lens.to(device)
    n_params = sum(p.numel() for p in lens.parameters())
    print(f"Lens: {config.lens.lens_type}, {n_params:,} parameters")

    # ── Setup training ────────────────────────────────────────────────────────
    loss_fn = get_loss_fn(config.training)
    optimizer = build_optimizer(lens, config.training)

    best_val_loss = float("inf")
    # Use a separate Generator so chunk shuffling is reproducible and independent
    # of any other torch random state.
    rng = torch.Generator()
    rng.manual_seed(config.seed)

    # ── Phase 2: Training ─────────────────────────────────────────────────────
    print("\n=== Phase 2: Training ===")
    for epoch in range(1, config.training.num_epochs + 1):

        # Shuffle chunk order each epoch so the lens sees data in different sequence.
        chunk_order = torch.randperm(len(chunk_paths), generator=rng).tolist()

        epoch_losses: list[float] = []
        for ci in chunk_order:
            chunk_path = chunk_paths[ci]
            # Load chunk from disk → GPU. Each chunk is ~26 GB for 50K images in float16.
            data = torch.load(chunk_path, map_location=device, weights_only=True)
            chunk_patch_embs = data["patch_embs"]
            chunk_img_targets = data["img_targets"]
            chunk_image_ids = data["image_ids"]

            chunk_loss = train_one_chunk(
                lens=lens,
                apply_head_fn=apply_head_fn,
                patch_embs=chunk_patch_embs,
                img_targets=chunk_img_targets,
                image_ids=chunk_image_ids,
                optimizer=optimizer,
                loss_fn=loss_fn,
                lens_batch_size=config.training.lens_batch_size,
                grad_clip_norm=config.training.grad_clip_norm,
            )
            epoch_losses.append(chunk_loss)

            # Free chunk VRAM before loading the next one.
            del data, chunk_patch_embs, chunk_img_targets, chunk_image_ids
            torch.cuda.empty_cache()

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)

        do_val = (epoch % config.training.val_interval == 0) or (epoch == config.training.num_epochs)
        if do_val:
            val_loss = validate(
                lens=lens,
                apply_head_fn=apply_head_fn,
                val_patch_embs=val_patch_embs,
                val_img_targets=val_img_targets,
                val_image_ids=val_image_ids,
                loss_fn=loss_fn,
                lens_batch_size=config.training.lens_batch_size,
            )
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                if config.lens.lens_type == "direct":
                    lens.save(output_dir / "best_lens.pt")  # type: ignore[union-attr]
                else:
                    lens.save(output_dir / "best_lens.pt")  # type: ignore[union-attr]
            marker = " *" if is_best else ""
            print(f"Epoch {epoch:4d}/{config.training.num_epochs}  "
                  f"train={avg_train_loss:.4f}  val={val_loss:.4f}{marker}")
        else:
            print(f"Epoch {epoch:4d}/{config.training.num_epochs}  train={avg_train_loss:.4f}")

    # ── Save final artifacts ──────────────────────────────────────────────────
    lens.save(output_dir / "final_lens.pt")  # type: ignore[union-attr]
    config.to_yaml(str(output_dir / "config.yaml"))
    print(f"\nDone. Best val loss: {best_val_loss:.4f}. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
