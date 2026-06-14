"""Pre-compute patch embeddings through the frozen backbone.

Run once to fill the disk cache. All subsequent training reads from cache,
keeping the backbone completely out of the training loop.

On-disk format — one file per training chunk, plus one val file:

    cache_dir/
        train_chunk_0000.pt   # dict with keys: patch_embs, img_targets, image_ids
        train_chunk_0001.pt
        ...
        val.pt

Each .pt file contains:
    patch_embs   Tensor[N_tokens, d_model]      float16  (N_tokens = N_images * H * W)
    img_targets  Tensor[N_images, num_classes]  float16  (one per image, shared by all its patches)
    image_ids    Tensor[N_tokens]               int32    (maps each token → its image index within this file)
"""

from __future__ import annotations

import copy
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# Re-use the existing model code unchanged.
from train.tuned_lens.model import VisionModelWrapper
from train.tuned_lens.config import ModelConfig as TunedModelConfig

if TYPE_CHECKING:
    from .config import DirectLensConfig

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ── Standalone head (kept on GPU after backbone is offloaded) ─────────────────

class _NormLinearHead(nn.Module):
    """fc_norm + head copied from a standard timm ViT. Frozen."""

    def __init__(self, fc_norm: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.fc_norm = copy.deepcopy(fc_norm)
        self.head = copy.deepcopy(head)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.fc_norm(x))


class _LinearHead(nn.Module):
    """Custom linear head (e.g. DINOv2): just W @ x + b. Frozen."""

    def __init__(self, weight: Tensor, bias: Tensor | None) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone().detach(), requires_grad=False)
        self.bias = (nn.Parameter(bias.clone().detach(), requires_grad=False)
                     if bias is not None else None)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


def make_standalone_head(model_wrapper: VisionModelWrapper) -> nn.Module:
    """Extract just the classification head as a standalone frozen module.

    After pre-computation the full backbone can be offloaded to CPU while this
    tiny module (<2 MB) stays on GPU to provide apply_head for the embedding lens.

    For standard timm ViT:  copies model.fc_norm + model.head
    For custom-head models: copies the custom linear head (CLS-only portion)
    """
    if model_wrapper._custom_head is not None:
        d = model_wrapper.d_model
        w = model_wrapper._custom_head.weight[:, :d]
        b = model_wrapper._custom_head.bias
        return _LinearHead(w, b)
    return _NormLinearHead(model_wrapper.model.fc_norm, model_wrapper.model.head)


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_model_wrapper(config: DirectLensConfig, device: torch.device) -> VisionModelWrapper:
    """Create a frozen VisionModelWrapper in full-sequence mode for patch extraction."""
    tuned_config = TunedModelConfig(
        model_name=config.model.model_name,
        pretrained=config.model.pretrained,
        weights_path=config.model.weights_path,
        head_weights_path=config.model.head_weights_path,
        target_layers=[config.model.target_layer],
        freeze_model=True,
        patch_mode=False,
    )
    wrapper = VisionModelWrapper(tuned_config, device=str(device))
    wrapper.enable_full_sequence_mode()
    return wrapper


def _subsample_balanced(dataset: ImageFolder, max_per_class: int) -> Subset:
    """Deterministic balanced subset: at most max_per_class images per class."""
    class_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    selected: list[int] = []
    for label in sorted(class_to_indices):
        selected.extend(class_to_indices[label][:max_per_class])
    return Subset(dataset, selected)


def _save_chunk(
    patch_embs_parts: list[torch.Tensor],
    img_targets_parts: list[torch.Tensor],
    image_ids_parts: list[torch.Tensor],
    path: Path,
) -> None:
    data = {
        "patch_embs": torch.cat(patch_embs_parts, dim=0),      # [N_tokens, d_model] float16
        "img_targets": torch.cat(img_targets_parts, dim=0),    # [N_images, num_classes] float16
        "image_ids": torch.cat(image_ids_parts, dim=0).int(),  # [N_tokens] int32
    }
    torch.save(data, path)
    n_images = data["img_targets"].shape[0]
    n_tokens = data["patch_embs"].shape[0]
    size_gb = path.stat().st_size / 1e9
    print(f"  Saved {path.name}: {n_images} images / {n_tokens} tokens / {size_gb:.2f} GB")


# ── Public API ────────────────────────────────────────────────────────────────

@torch.no_grad()
def precompute_train(
    model_wrapper: VisionModelWrapper,
    config: DirectLensConfig,
    device: torch.device,
) -> list[Path]:
    """Pre-compute patch embeddings for the full training split and save to disk.

    Processes images in chunks of ``config.precompute.chunk_size``.
    Existing chunk files are skipped so the run is safe to resume after interruption.
    Returns a sorted list of all chunk file paths (existing + newly written).
    """
    cache_dir = Path(config.precompute.cache_dir) / str(config.seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_layer = config.model.target_layer

    # How many chunks are already on disk?
    existing_chunks = sorted(cache_dir.glob("train_chunk_*.pt"))
    n_existing = len(existing_chunks)
    skip_images = n_existing * config.precompute.chunk_size

    # Build balanced training dataset (deterministic, sorted by class)
    train_dir = os.path.join(config.precompute.imagenet_root, "train")
    full_dataset = ImageFolder(train_dir, transform=model_wrapper.get_transform())
    dataset = _subsample_balanced(full_dataset, config.precompute.max_images_per_class)
    total_images = len(dataset)

    if skip_images >= total_images:
        print(f"All {n_existing} training chunk(s) already on disk — skipping pre-computation.")
        return existing_chunks

    # Deterministic shuffle so each chunk gets a mix of all classes.
    # torch.randperm with a fixed generator is reproducible: same seed → same permutation,
    # so resume (slice shuffled[skip_images:]) lands on exactly the right images.
    g = torch.Generator()
    g.manual_seed(config.seed)
    shuffled = torch.randperm(total_images, generator=g).tolist()

    if n_existing > 0:
        print(f"Resuming from chunk {n_existing}: skipping the first {skip_images} images.")
        shuffled = shuffled[skip_images:]

    dataset = Subset(dataset, shuffled)
    n_remaining = len(dataset)
    n_batches = (n_remaining + config.precompute.precompute_batch_size - 1) // config.precompute.precompute_batch_size
    print(
        f"Pre-computing training embeddings: {n_remaining} images remaining "
        f"(layer {target_layer}, chunk_size={config.precompute.chunk_size})."
    )

    loader = DataLoader(
        dataset,
        batch_size=config.precompute.precompute_batch_size,
        shuffle=False,                                               # must be False for resume
        num_workers=config.precompute.num_workers,
        pin_memory=True,
        persistent_workers=config.precompute.num_workers > 0,
    )

    chunk_paths = list(existing_chunks)
    chunk_idx = n_existing

    # Accumulators for the current in-progress chunk
    patch_embs_parts: list[torch.Tensor] = []
    img_targets_parts: list[torch.Tensor] = []
    image_ids_parts: list[torch.Tensor] = []
    n_in_chunk = 0

    with tqdm(total=n_batches, desc="Pre-compute train", unit="batch") as pbar:
        for batch_images, _ in loader:
            batch_images = batch_images.to(device)

            # fp16 inference — autocast speeds up the backbone forward pass ~2×
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, patch_states, final_logits = model_wrapper.extract_cls_and_patches(batch_images)

            # patch_states[target_layer]: [B, H, W, d_model]
            patch_batch = patch_states[target_layer]
            B, H, W, d_model = patch_batch.shape
            N_patches = H * W

            # For each image, all N_patches tokens share the same image-level target.
            # image_ids maps every token to its image's local index within the current chunk.
            local_image_ids = torch.arange(B) + n_in_chunk        # [B]
            token_image_ids = local_image_ids.repeat_interleave(N_patches)  # [B * N_patches]

            patch_embs_parts.append(patch_batch.reshape(B * N_patches, d_model).cpu().half())
            img_targets_parts.append(final_logits.cpu().half())
            image_ids_parts.append(token_image_ids)
            n_in_chunk += B

            if n_in_chunk >= config.precompute.chunk_size:
                chunk_path = cache_dir / f"train_chunk_{chunk_idx:04d}.pt"
                _save_chunk(patch_embs_parts, img_targets_parts, image_ids_parts, chunk_path)
                chunk_paths.append(chunk_path)
                patch_embs_parts, img_targets_parts, image_ids_parts = [], [], []
                n_in_chunk = 0
                chunk_idx += 1

            pbar.update(1)
            pbar.set_postfix(chunk=chunk_idx, images_in_chunk=n_in_chunk)

    # Save the last (possibly partial) chunk
    if patch_embs_parts:
        chunk_path = cache_dir / f"train_chunk_{chunk_idx:04d}.pt"
        _save_chunk(patch_embs_parts, img_targets_parts, image_ids_parts, chunk_path)
        chunk_paths.append(chunk_path)

    return sorted(chunk_paths)


@torch.no_grad()
def precompute_val(
    model_wrapper: VisionModelWrapper,
    config: DirectLensConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-compute patch embeddings for a balanced subset of the ImageNet val split.

    Saves to ``cache_dir/val.pt`` and loads from there on subsequent calls.
    Returns (patch_embs, img_targets, image_ids) already on ``device``.
    """
    cache_dir = Path(config.precompute.cache_dir) / str(config.seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    val_cache = cache_dir / "val.pt"

    if val_cache.exists():
        print(f"Loading cached val embeddings from {val_cache}.")
        data = torch.load(val_cache, map_location="cpu", weights_only=True)
        return (
            data["patch_embs"].to(device),
            data["img_targets"].to(device),
            data["image_ids"].to(device),
        )

    target_layer = config.model.target_layer
    val_dir = os.path.join(config.precompute.imagenet_root, "val")
    full_val = ImageFolder(val_dir, transform=model_wrapper.get_transform())
    val_dataset = _subsample_balanced(full_val, config.precompute.val_max_images_per_class)

    n_val = len(val_dataset)
    n_batches = (n_val + config.precompute.precompute_batch_size - 1) // config.precompute.precompute_batch_size
    print(
        f"Pre-computing val embeddings: {n_val} images "
        f"(layer {target_layer}, {config.precompute.val_max_images_per_class}/class)."
    )

    loader = DataLoader(
        val_dataset,
        batch_size=config.precompute.precompute_batch_size,
        shuffle=False,
        num_workers=config.precompute.num_workers,
        pin_memory=True,
        persistent_workers=config.precompute.num_workers > 0,
    )

    patch_embs_parts: list[torch.Tensor] = []
    img_targets_parts: list[torch.Tensor] = []
    image_ids_parts: list[torch.Tensor] = []
    n_processed = 0

    with tqdm(total=n_batches, desc="Pre-compute val", unit="batch") as pbar:
        for batch_images, _ in loader:
            batch_images = batch_images.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, patch_states, final_logits = model_wrapper.extract_cls_and_patches(batch_images)

            patch_batch = patch_states[target_layer]
            B, H, W, d_model = patch_batch.shape
            N_patches = H * W

            local_image_ids = torch.arange(B) + n_processed
            token_image_ids = local_image_ids.repeat_interleave(N_patches)

            patch_embs_parts.append(patch_batch.reshape(B * N_patches, d_model).cpu().half())
            img_targets_parts.append(final_logits.cpu().half())
            image_ids_parts.append(token_image_ids)
            n_processed += B

            pbar.update(1)

    patch_embs = torch.cat(patch_embs_parts, dim=0)
    img_targets = torch.cat(img_targets_parts, dim=0)
    image_ids = torch.cat(image_ids_parts, dim=0).int()

    torch.save({"patch_embs": patch_embs, "img_targets": img_targets, "image_ids": image_ids}, val_cache)
    size_gb = val_cache.stat().st_size / 1e9
    print(f"Saved val embeddings to {val_cache} ({size_gb:.2f} GB).")

    return patch_embs.to(device), img_targets.to(device), image_ids.to(device)
