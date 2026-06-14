"""ImageNet data loading utilities."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Callable

from PIL import ImageFile
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from .config import DataConfig

# Allow PIL to load images that are slightly truncated rather than raising OSError.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _subsample_per_class(dataset: ImageFolder, max_per_class: int) -> Subset:
    """Return a Subset keeping at most ``max_per_class`` images per class.

    Images are taken from the front of each class's sorted file list, which
    ensures deterministic and reproducible subsets without needing a fixed seed.
    """
    class_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    selected: list[int] = []
    for label in sorted(class_to_indices):
        selected.extend(class_to_indices[label][:max_per_class])

    return Subset(dataset, selected)


def create_imagenet_dataloaders(
    config: DataConfig,
    train_transform: Callable,
    val_transform: Callable,
) -> tuple[DataLoader, DataLoader]:
    """Create train and val DataLoaders using ImageFolder.

    Expects ``config.imagenet_root`` to contain:
        imagenet_root/train/<synset_id>/*.JPEG
        imagenet_root/val/<synset_id>/*.JPEG

    If ``config.max_images_per_class`` is set, only that many images per class
    are used for training (the full val set is always kept intact).
    """
    train_dir = os.path.join(config.imagenet_root, "train")
    val_dir = os.path.join(config.imagenet_root, "val")

    train_dataset: ImageFolder | Subset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    if config.max_images_per_class is not None:
        original_size = len(train_dataset)
        train_dataset = _subsample_per_class(train_dataset, config.max_images_per_class)
        print(
            f"Subsampled training set: {len(train_dataset)} / {original_size} images "
            f"(max {config.max_images_per_class} per class)"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, val_loader
