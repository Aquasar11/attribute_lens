"""ImageNet data loading utilities."""

from __future__ import annotations

import os
from typing import Callable

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .config import DataConfig


def create_imagenet_dataloaders(
    config: DataConfig,
    train_transform: Callable,
    val_transform: Callable,
) -> tuple[DataLoader, DataLoader]:
    """Create train and val DataLoaders using ImageFolder.

    Expects ``config.imagenet_root`` to contain:
        imagenet_root/train/<synset_id>/*.JPEG
        imagenet_root/val/<synset_id>/*.JPEG
    """
    train_dir = os.path.join(config.imagenet_root, "train")
    val_dir = os.path.join(config.imagenet_root, "val")

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

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
