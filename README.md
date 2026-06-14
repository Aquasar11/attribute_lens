# attribute_lens

Vision-transformer interpretability via per-layer probing and patch attribution.

The backbone ViT is always **frozen** — only the lightweight probes / lenses / maps are trained.

---

## Overview

| Component | What it does |
|---|---|
| **Tuned lens** | One probe per layer mapping the CLS token to the final classification distribution |
| **Contrastive patch map** | A per-layer linear map pulling foreground patch tokens toward CLS, pushing background away (bbox-supervised) |
| **Direct lens** | A single per-layer lens trained directly to class space (no FG/BG masks) |
| **Unified lens** | A single affine per layer mapping a *patch* token straight to classifier-ready space (FG/BG-mask supervised) — v1 proxy, v2 faithfulness-aware |
| **Evaluation** | Two config-driven evals: **faithfulness** (insertion/deletion AUC) and **localization** (CAAP PG / AUPR) |

Everything is configured by a YAML under [configs/](configs/) and run as a Python module.
The repo splits cleanly into **training** (`src/train`, `configs/train`) and **evaluation**
(`src/eval`, `configs/eval`).

---

## Layout

```
src/
├── train/
│   ├── tuned_lens/         # CLS tuned lens + contrastive patch map (Lightning)
│   │   ├── model.py lens.py patch_map.py loss.py data.py bbox_data.py trainer.py ...
│   │   └── scripts/train.py train_patch_map.py prepare_imagenet.py download_dinov2_head.py
│   ├── direct_lens/        # single per-layer lens trained directly to class space
│   │   └── train.py lens.py loss.py precompute.py config.py
│   └── unified_lens/       # single affine per layer: patch token -> classifier space
│       ├── lens.py  data.py  config.py  trainv1.py (proxy)  trainv2.py (faithfulness)
└── eval/
    ├── common.py           # shared: MultiLayerLens, weight loaders, make_logits_fn,
    │                       #         preprocess, hooks, batched insertion/deletion curves
    ├── config.py           # FaithfulnessConfig / LocalizationConfig (+ from_yaml)
    ├── faithfulness.py     # insertion/deletion eval        [run file]
    ├── localization.py     # CAAP PG / AUPR eval            [run file]
    └── y_hat_creator.py    # build the y_hat baseline CSV    [run file]

configs/
├── train/{tuned_lens,patch_map,direct_lens,unified_lens}/*.yaml
└── eval/{faithfulness,localization}.yaml
```

Model structure (timm ViT): `patch_embed → _pos_embed → norm_pre → blocks[0..N-1] → norm → fc_norm(cls) → head`.
Forward hooks on `model.blocks[i]` capture `[B, 1+H*W, d_model]`; CLS (position 0) and patches (1:) are split after the hook.

---

## Supported models

Loaded via [timm](https://github.com/huggingface/pytorch-image-models) with pretrained ImageNet-1K heads.

| Model | timm identifier | Patch | Grid | d_model | Layers |
|---|---|---|---|---|---|
| CLIP ViT-L/14 | `vit_large_patch14_clip_224.openai_ft_in1k` | 14 | 16×16 | 1024 | 24 |
| ViT-L/16 (AugReg) | `vit_large_patch16_224.augreg_in21k_ft_in1k` | 16 | 14×14 | 1024 | 24 |
| DeiT3-L/16 | `deit3_large_patch16_224.fb_in1k` | 16 | 14×14 | 1024 | 24 |
| DINOv2-L/14 | `vit_large_patch14_dinov2.lvd142m` | 14 | 37×37 | 1024 | 24 |

The unified lens and both evaluations target **CLIP ViT-L/14** (geometry constants in
[src/eval/common.py](src/eval/common.py)). Tuned lens / patch map support all four via `configs/train/tuned_lens/default_*.yaml`.
DINOv2 needs a downloaded classification head — see [DINOv2 prerequisite](#dinov2-prerequisite).

---

## Install

```bash
pip install -e ".[dev]"
```

The backbone weights are fetched automatically by timm (`pretrained=True`) the first time, unless
a local checkpoint exists at the `*_weights_path` in the config (then that file is loaded).

> **Data is not committed.** `data/`, `model/`, `outputs/`, `results/` are git-ignored. Point the
> path fields in each YAML config at your own data / checkpoint locations.

---

## DINOv2 prerequisite

DINOv2's timm backbone ships without a classification head. Download it once:

```bash
pip install transformers
python -m train.tuned_lens.scripts.download_dinov2_head --output dinov2_large_imagenet1k_head.pt
```

`configs/train/tuned_lens/default_dinov2_l14.yaml` points to this file via `model.head_weights_path`.

---

## Data

### ImageNet (tuned lens / patch map / direct lens)

```bash
python -m train.tuned_lens.scripts.prepare_imagenet \
  --train-tar /path/ILSVRC2012_img_train.tar \
  --val-tar   /path/ILSVRC2012_img_val.tar \
  --output-dir /path/imagenet            # -> imagenet/{train,val}/<synset>/*.JPEG
```

Bounding boxes (for patch-map training) come from the ILSVRC2012 devkit (`*_bbox_train_v2`, `*_bbox_val_v3`).

### FG/BG masks (unified lens + localization)

Both consume `<dir>/images/*.JPEG` + `<dir>/masks_fg_bg/*.png` (FG=255, BG=0), from either
ImageNet-S segmentation or ImageNet bounding-box unions. The unified-lens `mask_source` field
(`imagenet_s` | `box`) selects which prepared directory to read. (Mask-prep scripts are not bundled
— point `data_dir` at a directory in that layout.)

---

## Training

```bash
# 1) Tuned lens (CLS probe).  Variants: affine_kld.yaml, mlp_kld.yaml; models: default_*.yaml
python -m train.tuned_lens.scripts.train \
  --config configs/train/tuned_lens/default.yaml --imagenet-root /path/imagenet

# 2) Contrastive patch map.  configs/train/patch_map/{full,lowrank}.yaml
python -m train.tuned_lens.scripts.train_patch_map \
  --config configs/train/patch_map/lowrank.yaml \
  --imagenet-root /path/imagenet --bbox-dir-train /path/boxes/train --bbox-dir-val /path/boxes/val

# 3) Direct lens (optional)
python -m train.direct_lens.train --config configs/train/direct_lens/default.yaml

# 4) Unified lens — v1 (proxy) and v2 (faithfulness surrogate)
python -m train.unified_lens.trainv1 --config configs/train/unified_lens/v1.yaml
python -m train.unified_lens.trainv2 --config configs/train/unified_lens/v2.yaml
```

The unified-lens supervision source is set in the config or overridden on the CLI:

```bash
python -m train.unified_lens.trainv2 --config configs/train/unified_lens/v2.yaml --mask-source box
```

| Field | Meaning |
|---|---|
| `mask_source` | `imagenet_s` (segmentation) or `box` (bbox union); selects the default `data_dir` |
| `data_dir` | explicit override for the data directory |
| `layers` | ViT block indices to train (default `[22]`) |
| v2: `faith_weight`, `warmup_epochs`, `faith_fractions/samples/beta`, `del/ins_weight`, `conc_weight` | faithfulness-surrogate knobs |

Both trainers use **CLIP mean/std** + `norm → fc_norm → head` (no train/eval shift).
Outputs: `outputs/unified_lens/v{1,2}/layer_NN_best.pt`.

---

## Evaluation

Both evals are a **YAML config + a run file**, share their plumbing
([src/eval/common.py](src/eval/common.py)), and pick the lens to score via `mapper_type`:

| `mapper_type` | Lens scored | Checkpoints used |
|---|---|---|
| `unified` | unified affine + ViT `norm+fc_norm+head` | `unified_weights_dir` |
| `full` / `low_rank` | patch map + tuned/MLP lens | `mapper_weights_dir`, `lens_weights_dir` |
| `direct_lens` | single direct lens (one layer) | `direct_lens_path`, `direct_lens_layer` |

### Faithfulness (insertion / deletion AUC)

```bash
# first build the y_hat baseline for the image set:
python -m eval.y_hat_creator --image-dir data/first_100/images \
  --output-csv data/first_100/y_hat_baseline.csv

python -m eval.faithfulness --config configs/eval/faithfulness.yaml
```

Ranks patches by the per-patch target-class score (`score_space`, default `logprob`), with optional
edge-aware (bilateral) ring-sigma / multi-kernel neighbour smoothing and optional multi-layer fusion.
Writes per-image curves + AUCs (CSV), the attribution maps (`.npz`), and plots.

| Metric | Direction | Meaning |
|---|---|---|
| Insertion AUC | higher better | top-attributed patches drive the prediction |
| Deletion AUC | lower better | removing top patches collapses confidence fast |

### Localization (CAAP)

```bash
python -m eval.localization --config configs/eval/localization.yaml
```

Compares the upsampled attribution map against ground-truth FG/BG masks: Pointing Game, AUPR1 (FG), AUPR0 (BG).

---

## Workflow summary

```
1. Prepare data    prepare_imagenet (+ bbox XMLs) ; FG/BG masks for unified/localization
2. Tuned lens      python -m train.tuned_lens.scripts.train       --config configs/train/tuned_lens/default.yaml
3. Patch map       python -m train.tuned_lens.scripts.train_patch_map --config configs/train/patch_map/lowrank.yaml
4. Unified lens    python -m train.unified_lens.trainv{1,2}       --config configs/train/unified_lens/v{1,2}.yaml
5. Evaluate        python -m eval.faithfulness --config configs/eval/faithfulness.yaml
                   python -m eval.localization --config configs/eval/localization.yaml
```
