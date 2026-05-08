"""Training script for direct_attribute_lens.

Two phases:
  1. Pre-computation  — run backbone once on all training images, save patch embeddings to disk.
  2. Training         — load cached embeddings chunk by chunk, train a linear lens on them.

After pre-computation the full backbone is offloaded to CPU. Only the tiny classification
head (~2 MB) stays on GPU for the embedding lens type; for the direct lens nothing is needed.

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
from tqdm import tqdm

from .config import DirectLensConfig, TrainingConfig
from .lens import EmbeddingLens, DirectLens
from .loss import get_loss_fn
from .precompute import build_model_wrapper, make_standalone_head, precompute_train, precompute_val


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
    n_steps = (N_tokens + lens_batch_size - 1) // lens_batch_size

    total_loss = 0.0
    steps_done = 0

    with tqdm(total=n_steps, desc="  batches", unit="step", leave=False) as pbar:
        for start in range(0, N_tokens, lens_batch_size):
            idx = perm[start : start + lens_batch_size]

            x = patch_embs[idx].float()              # float16 → float32
            y = img_targets[image_ids[idx]].float()  # image-level target for each token

            if apply_head_fn is not None:
                logits = apply_head_fn(lens(x))      # embedding type: lens then frozen head
            else:
                logits = lens(x)                     # direct type: lens outputs logits directly

            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(lens.parameters(), grad_clip_norm)
            optimizer.step()

            total_loss += loss.item()
            steps_done += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / steps_done if steps_done > 0 else 0.0


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


# ── Optimizer and scheduler factories ────────────────────────────────────────

def build_optimizer(lens: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    params = lens.parameters()
    if config.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(params, lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)
    return torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=config.scheduler_patience,
            factor=config.scheduler_factor,
            min_lr=config.scheduler_min_lr,
        )
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a direct_attribute_lens")
    parser.add_argument("--config", default="configs/direct_lens/default.yaml")
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--lens-type", choices=["embedding", "direct"], default=None)
    parser.add_argument("--loss-type", choices=["kld", "ce", "combined"], default=None)
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
          f"loss={config.training.loss_type}, lr={config.training.lr}, "
          f"scheduler={config.training.scheduler}")

    # ── Phase 1: Pre-computation ──────────────────────────────────────────────
    print("\n=== Phase 1: Pre-computation ===")
    model_wrapper = build_model_wrapper(config, device)
    d_model = model_wrapper.d_model
    num_classes = model_wrapper.num_classes

    chunk_paths = precompute_train(model_wrapper, config, device)
    val_patch_embs, val_img_targets, val_image_ids = precompute_val(model_wrapper, config, device)
    val_patch_embs = val_patch_embs.to(device)
    val_img_targets = val_img_targets.to(device)
    val_image_ids = val_image_ids.to(device)

    print(f"\nPre-computation done: {len(chunk_paths)} chunk(s), {val_patch_embs.shape[0]:,} val tokens.")

    # ── Build lens and apply_head_fn, then offload backbone ──────────────────
    if config.lens.lens_type == "direct":
        head_w, head_b = model_wrapper.get_head_parameters()
        lens: nn.Module = DirectLens(d_model, num_classes, config.lens.bias, head_w, head_b)
        apply_head_fn: Callable[[Tensor], Tensor] | None = None
    else:
        # Extract the tiny head (fc_norm + linear) before offloading the backbone.
        standalone_head = make_standalone_head(model_wrapper).to(device)
        lens = EmbeddingLens(d_model, config.lens.bias)
        apply_head_fn = standalone_head

    # Offload backbone transformer blocks to CPU — not needed during training.
    print("Offloading backbone to CPU...")
    model_wrapper.to("cpu")
    torch.cuda.empty_cache()

    lens = lens.to(device)
    n_params = sum(p.numel() for p in lens.parameters())
    print(f"Lens: {config.lens.lens_type}, {n_params:,} parameters")

    # ── Setup training ────────────────────────────────────────────────────────
    loss_fn = get_loss_fn(config.training)
    optimizer = build_optimizer(lens, config.training)
    scheduler = build_scheduler(optimizer, config.training)

    best_val_loss = float("inf")
    rng = torch.Generator()
    rng.manual_seed(config.seed)

    # ── Phase 2: Training ─────────────────────────────────────────────────────
    print("\n=== Phase 2: Training ===")
    epoch_bar = tqdm(range(1, config.training.num_epochs + 1), desc="Epochs", unit="epoch")

    for epoch in epoch_bar:
        # Shuffle chunk order each epoch so the lens sees data in different sequence.
        chunk_order = torch.randperm(len(chunk_paths), generator=rng).tolist()

        epoch_losses: list[float] = []
        chunk_bar = tqdm(chunk_order, desc=f"Epoch {epoch:3d} chunks", unit="chunk", leave=False)

        for ci in chunk_bar:
            chunk_path = chunk_paths[ci]
            # Load chunk from disk → GPU. Each chunk is ~26 GB for 50K images in float16.
            data = torch.load(chunk_path, map_location=device, weights_only=True)

            chunk_loss = train_one_chunk(
                lens=lens,
                apply_head_fn=apply_head_fn,
                patch_embs=data["patch_embs"],
                img_targets=data["img_targets"],
                image_ids=data["image_ids"],
                optimizer=optimizer,
                loss_fn=loss_fn,
                lens_batch_size=config.training.lens_batch_size,
                grad_clip_norm=config.training.grad_clip_norm,
            )
            epoch_losses.append(chunk_loss)
            chunk_bar.set_postfix(chunk_loss=f"{chunk_loss:.4f}")

            # Free chunk VRAM before loading the next one.
            del data
            torch.cuda.empty_cache()

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)

        # Step scheduler on training loss — reduces LR when train loss stops falling.
        if scheduler is not None:
            scheduler.step(avg_train_loss)
        current_lr = optimizer.param_groups[0]["lr"]

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
                lens.save(output_dir / "best_lens.pt")  # type: ignore[union-attr]
            marker = " *" if is_best else ""
            epoch_bar.write(
                f"Epoch {epoch:4d}/{config.training.num_epochs}  "
                f"train={avg_train_loss:.4f}  val={val_loss:.4f}  "
                f"lr={current_lr:.2e}{marker}"
            )
        else:
            epoch_bar.write(
                f"Epoch {epoch:4d}/{config.training.num_epochs}  "
                f"train={avg_train_loss:.4f}  lr={current_lr:.2e}"
            )

    # ── Save final artifacts ──────────────────────────────────────────────────
    lens.save(output_dir / "final_lens.pt")  # type: ignore[union-attr]
    config.to_yaml(str(output_dir / "config.yaml"))
    print(f"\nDone. Best val loss: {best_val_loss:.4f}. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
