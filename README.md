# attribute_lens

Vision transformer interpretability via per-layer probing and contrastive patch alignment.

---

## Overview

**attribute_lens** studies what intermediate layers of a frozen ViT "know" about the final prediction.
It contains two complementary components:

| Component | What it does |
|---|---|
| **Tuned Lens** | Trains a lightweight probe per layer to predict the final model output distribution from each layer's CLS token |
| **Contrastive Patch Map** | Trains a per-layer linear map that pulls foreground patch embeddings toward the CLS token and pushes background patches away, using ImageNet bounding box annotations as weak supervision |

The backbone (CLIP ViT-L/14 or any timm ViT) is always **frozen** — only the probes/maps are trained.

---

## Architecture

```
src/tuned_lens/
├── config.py               # All config dataclasses with YAML serialization
├── model.py                # VisionModelWrapper: frozen timm ViT + hook-based extraction
├── lens.py                 # AffineLens / MLPLens, collected in LensBank (one per layer)
├── patch_map.py            # FullPatchMap / LowRankPatchMap, collected in PatchMapBank
├── loss.py                 # KLD, CE, combined loss functions
├── data.py                 # ImageNet DataLoader (ImageFolder + per-class subsampling)
├── bbox_data.py            # BboxImageNetDataset: ImageNet + Pascal-VOC XML bbox annotations
├── trainer.py              # Lightning module — trains LensBank against frozen backbone
├── contrastive_trainer.py  # Lightning module — trains PatchMapBank with contrastive loss
├── sweep.py                # Optuna hyperparameter search
└── scripts/
    ├── train.py            # CLI: train tuned lens
    ├── train_patch_map.py  # CLI: train contrastive patch map
    └── prepare_imagenet.py # Extract ILSVRC2012 tars into ImageFolder layout
```

---

## Component 1: Tuned Lens

Trains learnable probes (one per target layer) that map the CLS token hidden state to the final
classification distribution, enabling layer-by-layer analysis of model confidence.

**Training objective:** KL divergence between the lens output and the frozen model's final softmax.

### Lens types

| Type | Description |
|---|---|
| `affine` | Single linear layer `d_model → num_classes` |
| `mlp` | Multi-layer MLP with GELU activations |

### Train

```bash
python -m tuned_lens.scripts.train --config configs/default.yaml
```

Common overrides:

```bash
python -m tuned_lens.scripts.train --config configs/default.yaml \
  --imagenet-root /path/to/imagenet \
  --lr 5e-4 --batch-size 128 --epochs 20 \
  --target-layers 0 6 12 18 23
```

Outputs saved to `outputs/tuned_lens/` (configurable via `output_dir`):

```
outputs/tuned_lens/
├── config.yaml
├── best_lenses/layer_N.pt   # best checkpoint per layer (by val loss)
├── final_lenses/layer_N.pt  # weights at end of training
└── checkpoints/             # Lightning checkpoints (top-3 by val/loss_avg)
```

---

## Component 2: Contrastive Patch Map

Trains a per-layer affine map `y = Wx + b` that transforms patch token embeddings so that:
- **Foreground patches** (inside bounding boxes, overlap ≥ 0.80) move **closer** to the CLS token
- **Background patches** (outside bounding boxes, overlap ≤ 0.20) move **further** from the CLS token

The transformed embedding `y` can then be fed into the CLS lens for spatially-aware attribution.

**Training objective (per layer):**
```
loss = MSE(map(fg_patches), cls) − neg_weight × clamp(MSE(map(bg_patches), cls), max=neg_clip)
```

**Data:** ~544K training images with bounding box annotations (≈42% of the ILSVRC 2012 train set) and all 50K val images annotated. Bounding boxes are parsed from Pascal-VOC XML files and transformed into the 224×224 crop coordinate space.

### Map types

| Type | Parameterization | Parameters |
|---|---|---|
| `full` | Square matrix `W ∈ ℝ^{d×d}` + bias `b ∈ ℝ^d` | d² + d |
| `low_rank` | `W = AB`, `A ∈ ℝ^{d×r}`, `B ∈ ℝ^{r×d}` + bias `b ∈ ℝ^d` | 2dr + d |

### Train

```bash
python -m tuned_lens.scripts.train_patch_map --config configs/patch_map.yaml
```

Common overrides:

```bash
python -m tuned_lens.scripts.train_patch_map --config configs/patch_map.yaml \
  --imagenet-root /path/to/imagenet \
  --bbox-dir-train /path/to/boxes/train \
  --bbox-dir-val   /path/to/boxes/val \
  --map-type low_rank --rank 128 \
  --target-layers 12 19 \
  --epochs 20 --batch-size 32
```

Smoke test (small subset):

```bash
python -m tuned_lens.scripts.train_patch_map --config configs/patch_map.yaml \
  --max-images-per-class 5 --epochs 2 --batch-size 8 --target-layers 12 19
```

Outputs saved to `outputs/patch_map/` (configurable via `output_dir`):

```
outputs/patch_map/
├── config.yaml
├── best_maps/layer_N.pt    # best checkpoint per layer (by val loss)
├── final_maps/layer_N.pt   # weights at end of training
└── checkpoints/            # Lightning checkpoints (top-3 by val/loss_avg)
```

---

## Configuration

All config lives in YAML files that map to typed dataclasses.

| File | Dataclass | Purpose |
|---|---|---|
| `configs/default.yaml` | `TunedLensConfig` | Tuned lens training |
| `configs/patch_map.yaml` | `PatchMapFullConfig` | Patch map training |

All YAML fields can be overridden from the CLI. Run any script with `--help` for the full list.

**Key patch map config knobs** (`configs/patch_map.yaml`):

```yaml
patch_map:
  map_type: "low_rank"   # "full" | "low_rank"
  rank: 128
  fg_threshold: 0.80     # patch overlap >= this → foreground
  bg_threshold: 0.20     # patch overlap <= this → background
  neg_weight: 0.5        # repulsion term weight
  neg_clip: 0.3          # clamp on negative MSE
```

---

## Model structure (timm ViT)

```
patch_embed → _pos_embed → norm_pre → blocks[0..N-1] → norm → fc_norm(cls) → head
```

Forward hooks on `model.blocks[i]` capture the full token sequence `[B, 1+H*W, d_model]`.
CLS (position 0) and patches (positions 1:) are split after the hook.

---

## Notebooks

| Notebook | Description |
|---|---|
| `00_extract_embeddings.ipynb` | Extract and save ViT embeddings + bounding boxes for val images |
| `01_patch_lens_scoring.ipynb` | Attribution heatmaps via patch lens (k×k neighborhood scoring) |
| `02_cls_lens_individual_scoring.ipynb` | Attribution heatmaps via CLS lens applied to raw patch tokens |
| `03_cls_lens_mean_adjusted_scoring.ipynb` | CLS lens with mean-shift correction |
| `04_patch_map_scoring.ipynb` | **Compare** raw patch scores vs. patch-map-transformed scores side by side for 10 images; includes difference maps |

---

## Data preparation

### Extract ImageNet tars

```bash
python -m tuned_lens.scripts.prepare_imagenet \
  --train-tar /path/to/ILSVRC2012_img_train.tar \
  --val-tar   /path/to/ILSVRC2012_img_val.tar \
  --output-dir /path/to/imagenet
```

Expected layout after extraction:

```
imagenet/
├── train/<synset_id>/*.JPEG    (1,281,167 images, 1000 classes)
└── val/<synset_id>/*.JPEG      (50,000 images)
```

Bounding box XMLs (separate download, Pascal-VOC format):

```
boxes/
├── train/<synset_id>/<synset_id>_<id>.xml   (~544K files, ~42% of train)
└── val/ILSVRC2012_val_XXXXXXXX.xml           (50K files, 100% of val)
```

---

## Install

```bash
pip install -e ".[dev]"
```
