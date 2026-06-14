"""
unified_lens.data
=================
Foreground/background patch-mask dataset for unified-lens training.

Both supervision sources -- ImageNet-S segmentation masks and ImageNet bounding-box
union masks -- live on disk in the *same* layout::

    <data_dir>/images/<name>.JPEG
    <data_dir>/masks_fg_bg/<name>.png   (FG=255 inside object/box, BG=0 outside)

so a single dataset class serves both; ``mask_source`` only selects which prepared
directory to read (and is overridable via ``data_dir``).
"""

from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from .lens import CLIP_MEAN, CLIP_STD


# Default prepared-data locations per supervision source. Override with cfg.data_dir.
DEFAULT_DATA_DIRS = {
    "imagenet_s": "/home/mll/attr_lens/data/imagenet-s/validation",
    "box": "/home/mll/attr_lens/data/boxes/train",
}


def resolve_data_dir(mask_source, data_dir=None):
    """Pick the data directory: explicit ``data_dir`` wins, else the ``mask_source`` default."""
    if data_dir:
        return data_dir
    if mask_source not in DEFAULT_DATA_DIRS:
        raise ValueError(
            f"Unknown mask_source={mask_source!r}; expected one of {list(DEFAULT_DATA_DIRS)} "
            f"or an explicit data_dir."
        )
    return DEFAULT_DATA_DIRS[mask_source]


class FGBGMaskDataset(Dataset):
    """Image + binary FG/BG patch mask.

    Images are resized to ``img_size`` with BICUBIC + CLIP-normalized (matching the
    eval's geometry/stats); masks are nearest-resized to the patch grid.
    """

    def __init__(self, data_dir, img_size=224, patch_size=14):
        self.img_dir = Path(data_dir) / "images"
        self.mask_dir = Path(data_dir) / "masks_fg_bg"

        all_imgs = sorted(p.stem for p in self.img_dir.glob("*.JPEG"))
        self.names = [n for n in all_imgs if (self.mask_dir / f"{n}.png").exists()]
        if not self.names:
            raise RuntimeError(
                f"No (image, mask) pairs found under {data_dir}. "
                f"Looked in {self.img_dir} and {self.mask_dir}."
            )

        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
        grid = img_size // patch_size
        self.grid = grid
        self.mask_tf = transforms.Compose([
            transforms.Resize(
                (grid, grid),
                interpolation=InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),  # 0 -> 0.0 ; 255 -> 1.0
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = Image.open(self.img_dir / f"{name}.JPEG").convert("RGB")
        msk = Image.open(self.mask_dir / f"{name}.png").convert("L")
        return self.img_tf(img), self.mask_tf(msk).squeeze(0).flatten()  # (3,H,W), (N,)
