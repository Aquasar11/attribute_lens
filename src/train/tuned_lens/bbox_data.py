"""Bbox-aware ImageNet dataset for contrastive patch map training.

Only images that have a matching Pascal-VOC XML annotation are included.
Each sample returns (image_tensor, class_idx, fg_mask, bg_mask) where the
masks are pre-computed in the dataloader worker so they arrive at the
training step as stacked tensors — no per-batch CPU loop in the hot path.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

from .config import PatchMapDataConfig

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------------------------
# XML parsing and bbox coordinate helpers
# ---------------------------------------------------------------------------

def parse_bbox_xml(xml_path: Path) -> tuple[list[dict], int, int]:
    """Parse a Pascal-VOC XML annotation file.

    Returns:
        bboxes:  list of {'xmin', 'ymin', 'xmax', 'ymax', 'name'}
        orig_w:  image width recorded in the XML
        orig_h:  image height recorded in the XML
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    orig_w = int(size_elem.find("width").text)
    orig_h = int(size_elem.find("height").text)

    bboxes = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        bboxes.append({
            "xmin": float(bndbox.find("xmin").text),
            "ymin": float(bndbox.find("ymin").text),
            "xmax": float(bndbox.find("xmax").text),
            "ymax": float(bndbox.find("ymax").text),
            "name": obj.find("name").text,
        })

    return bboxes, orig_w, orig_h


def transform_bboxes_224(
    bboxes_orig: list[dict],
    orig_w: int,
    orig_h: int,
) -> list[dict]:
    """Map bboxes from original image space into the 224×224 crop space.

    Applies timm's default eval pre-processing geometry:
      1. Resize so the shorter side = 224 (keeping aspect ratio).
      2. Center-crop to 224×224.

    Boxes that fall entirely outside the crop are dropped.
    """
    scale = 224.0 / min(orig_w, orig_h)
    x_off = (orig_w * scale - 224.0) / 2.0
    y_off = (orig_h * scale - 224.0) / 2.0

    result = []
    for bbox in bboxes_orig:
        xmin = max(0.0, min(224.0, bbox["xmin"] * scale - x_off))
        ymin = max(0.0, min(224.0, bbox["ymin"] * scale - y_off))
        xmax = max(0.0, min(224.0, bbox["xmax"] * scale - x_off))
        ymax = max(0.0, min(224.0, bbox["ymax"] * scale - y_off))

        if xmax <= xmin or ymax <= ymin:
            continue  # box was cropped away

        result.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

    return result


def classify_patches(
    bboxes_224: list[dict],
    grid_size: int,
    patch_size: int,
    fg_threshold: float,
    bg_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify each patch in a grid as foreground, background, or ignored.

    A patch is *foreground* if its overlap with any bbox >= fg_threshold of
    its area.  It is *background* if that overlap <= bg_threshold.  Patches
    in between are ignored during training.

    Uses fully-vectorised numpy operations — no Python loop over patches.

    Args:
        bboxes_224:   Bounding boxes in 224px crop space (xmin/ymin/xmax/ymax).
        grid_size:    Number of patches per spatial side (e.g. 16 for ViT-L/14).
        patch_size:   Pixel width/height of each patch (e.g. 14 for ViT-L/14).
        fg_threshold: Minimum overlap ratio to label a patch as foreground.
        bg_threshold: Maximum overlap ratio to label a patch as background.

    Returns:
        fg_mask: bool array [grid_size, grid_size]
        bg_mask: bool array [grid_size, grid_size]
    """
    if not bboxes_224:
        fg_mask = np.zeros((grid_size, grid_size), dtype=bool)
        bg_mask = np.ones((grid_size, grid_size), dtype=bool)
        return fg_mask, bg_mask

    patch_area = float(patch_size * patch_size)

    # Patch grid edges
    idx = np.arange(grid_size)
    px0 = idx * patch_size          # [W] left edges
    px1 = px0 + patch_size          # [W] right edges
    py0 = idx * patch_size          # [H] top edges
    py1 = py0 + patch_size          # [H] bottom edges

    # Bounding boxes [K, 4]
    boxes = np.array([[b["xmin"], b["ymin"], b["xmax"], b["ymax"]] for b in bboxes_224])
    bx0, by0, bx1, by1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # Intersection extents: [W, K] and [H, K]
    inter_w = np.maximum(
        0.0,
        np.minimum(px1[:, None], bx1[None, :]) - np.maximum(px0[:, None], bx0[None, :]),
    )
    inter_h = np.maximum(
        0.0,
        np.minimum(py1[:, None], by1[None, :]) - np.maximum(py0[:, None], by0[None, :]),
    )

    # Total overlap per (row, col): sum over all K boxes → [H, W]
    overlap = (inter_h[:, None, :] * inter_w[None, :, :]).sum(axis=2)
    ratio = np.minimum(overlap / patch_area, 1.0)

    fg_mask = ratio >= fg_threshold
    bg_mask = ratio <= bg_threshold

    return fg_mask, bg_mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _build_class_to_idx(train_dir: Path) -> dict[str, int]:
    """Sort synset folder names alphabetically → assign 0-based class indices.

    Matches torchvision ImageFolder's convention so indices are consistent.
    """
    synsets = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    return {s: i for i, s in enumerate(synsets)}


class BboxImageNetDataset(Dataset):
    """ImageNet dataset filtered to images that have bbox XML annotations.

    Returns per item: ``(image_tensor, class_idx, fg_mask, bg_mask)``

    - *image_tensor*: transformed image [C, H, W]
    - *class_idx*: integer class label
    - *fg_mask*: bool tensor [H*W] — True for foreground patches
    - *bg_mask*: bool tensor [H*W] — True for background patches

    Patch classification runs in the dataloader worker process, fully
    overlapping with GPU forward passes in the main training loop.
    """

    def __init__(
        self,
        image_dir: str | Path,
        bbox_dir: str | Path,
        transform: Callable,
        split: str,
        class_to_idx: dict[str, int],
        grid_size: int,
        patch_size: int,
        fg_threshold: float,
        bg_threshold: float,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.bbox_dir = Path(bbox_dir)
        self.transform = transform
        self.split = split
        self.class_to_idx = class_to_idx
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.fg_threshold = fg_threshold
        self.bg_threshold = bg_threshold

        # samples: list of (img_path, class_idx, xml_path)
        self.samples: list[tuple[Path, int, Path]] = []

        if split == "train":
            self._build_train_index()
        else:
            self._build_val_index()

    def _build_train_index(self) -> None:
        """Match train XMLs to image files via synset directory structure."""
        print("Building train bbox index...", flush=True)
        for synset_dir in sorted(self.bbox_dir.iterdir()):
            if not synset_dir.is_dir():
                continue
            class_idx = self.class_to_idx.get(synset_dir.name)
            if class_idx is None:
                continue
            for xml_path in sorted(synset_dir.glob("*.xml")):
                img_path = self.image_dir / synset_dir.name / f"{xml_path.stem}.JPEG"
                if img_path.exists():
                    self.samples.append((img_path, class_idx, xml_path))
        print(f"  {len(self.samples)} train samples with bbox annotations", flush=True)

    def _build_val_index(self) -> None:
        """Walk val image dir to build stem→(path, class_idx), then match XMLs."""
        print("Building val bbox index (walking val dir)...", flush=True)
        stem_to_sample: dict[str, tuple[Path, int]] = {}
        for synset_dir in self.image_dir.iterdir():
            if not synset_dir.is_dir():
                continue
            class_idx = self.class_to_idx.get(synset_dir.name)
            if class_idx is None:
                continue
            for img_path in synset_dir.iterdir():
                if img_path.suffix.upper() in {".JPEG", ".JPG", ".PNG"}:
                    stem_to_sample[img_path.stem] = (img_path, class_idx)

        for xml_path in sorted(self.bbox_dir.glob("*.xml")):
            entry = stem_to_sample.get(xml_path.stem)
            if entry is not None:
                img_path, class_idx = entry
                self.samples.append((img_path, class_idx, xml_path))
        print(f"  {len(self.samples)} val samples with bbox annotations", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        img_path, class_idx, xml_path = self.samples[idx]

        bboxes_orig, orig_w, orig_h = parse_bbox_xml(xml_path)
        bboxes_224 = transform_bboxes_224(bboxes_orig, orig_w, orig_h)

        fg, bg = classify_patches(
            bboxes_224, self.grid_size, self.patch_size,
            self.fg_threshold, self.bg_threshold,
        )

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        return (
            img_tensor,
            class_idx,
            torch.from_numpy(fg.reshape(-1)),   # [H*W]
            torch.from_numpy(bg.reshape(-1)),   # [H*W]
        )


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def bbox_collate_fn(
    batch: list[tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate fn that stacks pre-computed fg/bg masks as bool tensors."""
    images   = torch.stack([item[0] for item in batch])
    labels   = torch.tensor([item[1] for item in batch], dtype=torch.long)
    fg_masks = torch.stack([item[2] for item in batch])  # [B, H*W]
    bg_masks = torch.stack([item[3] for item in batch])  # [B, H*W]
    return images, labels, fg_masks, bg_masks


def _subsample_by_class(
    samples: list[tuple], max_per_class: int
) -> list[tuple]:
    """Keep at most *max_per_class* samples per class_idx (element [1])."""
    buckets: dict[int, list] = defaultdict(list)
    for s in samples:
        buckets[s[1]].append(s)
    result = []
    for cls in sorted(buckets):
        result.extend(buckets[cls][:max_per_class])
    return result


def create_bbox_dataloaders(
    config: PatchMapDataConfig,
    train_transform: Callable,
    val_transform: Callable,
    grid_size: int,
    patch_size: int,
    fg_threshold: float,
    bg_threshold: float,
) -> tuple[DataLoader, DataLoader]:
    """Create train and val DataLoaders with pre-computed fg/bg patch masks.

    Patch classification is performed in the dataloader worker processes,
    running in parallel with the GPU forward pass.
    """
    train_dir = Path(config.imagenet_root) / "train"
    class_to_idx = _build_class_to_idx(train_dir)

    classify_kwargs = dict(
        grid_size=grid_size,
        patch_size=patch_size,
        fg_threshold=fg_threshold,
        bg_threshold=bg_threshold,
    )

    train_dataset = BboxImageNetDataset(
        image_dir=train_dir,
        bbox_dir=config.bbox_dir_train,
        transform=train_transform,
        split="train",
        class_to_idx=class_to_idx,
        **classify_kwargs,
    )
    val_dataset = BboxImageNetDataset(
        image_dir=Path(config.imagenet_root) / "val",
        bbox_dir=config.bbox_dir_val,
        transform=val_transform,
        split="val",
        class_to_idx=class_to_idx,
        **classify_kwargs,
    )

    if config.max_images_per_class is not None:
        original_size = len(train_dataset)
        train_dataset.samples = _subsample_by_class(
            train_dataset.samples, config.max_images_per_class
        )
        print(
            f"Subsampled train: {len(train_dataset.samples)} / {original_size} images "
            f"(max {config.max_images_per_class} per class)",
            flush=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
        collate_fn=bbox_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        collate_fn=bbox_collate_fn,
    )

    return train_loader, val_loader
