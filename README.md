# attribute_lens

Vision transformer interpretability via per-layer probing and contrastive patch alignment.

The backbone ViT is always **frozen** — only the lightweight probes and maps are trained.

---

## Overview

| Component | What it does |
|---|---|
| **Tuned Lens** | Trains one probe per layer that maps each layer's CLS token to the final classification distribution |
| **Contrastive Patch Map** | Trains a per-layer linear map that pulls foreground patch tokens toward the CLS token and pushes background tokens away, using bounding box annotations |
| **Evaluation** | Generates attribution heatmaps and computes insertion/deletion AUC across 9 experiment variants |

---

## Supported models

All models are loaded via [timm](https://github.com/huggingface/pytorch-image-models) with pretrained ImageNet-1K classification heads.

| Model | timm identifier | Patch size | Patch grid | d_model | Layers |
|---|---|---|---|---|---|
| CLIP ViT-L/14 | `vit_large_patch14_clip_224.openai_ft_in1k` | 14 px | 16×16 | 1024 | 24 |
| ViT-L/16 (AugReg) | `vit_large_patch16_224.augreg_in21k_ft_in1k` | 16 px | 14×14 | 1024 | 24 |
| DeiT3-L/16 | `deit3_large_patch16_224.fb_in1k` | 16 px | 14×14 | 1024 | 24 |

Each model has a corresponding default training config in `configs/`:

| Config file | Model |
|---|---|
| `configs/default.yaml` | CLIP ViT-L/14 |
| `configs/default_vit_l16.yaml` | ViT-L/16 |
| `configs/default_deit3_l16.yaml` | DeiT3-L/16 |

---

## Architecture

```
src/tuned_lens/
├── config.py               # All config dataclasses with YAML serialization
├── model.py                # VisionModelWrapper: frozen timm ViT + hook-based extraction
├── lens.py                 # AffineLens / MLPLens collected in LensBank (one per layer)
├── patch_map.py            # FullPatchMap / LowRankPatchMap collected in PatchMapBank
├── loss.py                 # KLD, CE, combined loss functions
├── data.py                 # ImageNet DataLoader (ImageFolder + per-class subsampling)
├── bbox_data.py            # BboxImageNetDataset: ImageNet + Pascal-VOC bbox annotations
├── trainer.py              # Lightning module — trains LensBank against frozen backbone
├── contrastive_trainer.py  # Lightning module — trains PatchMapBank with contrastive loss
├── sweep.py                # Optuna hyperparameter search
└── scripts/
    ├── train.py            # CLI: train tuned lens
    ├── train_patch_map.py  # CLI: train contrastive patch map
    └── prepare_imagenet.py # Extract ILSVRC2012 tars into ImageFolder layout

src/attribute_lens/
├── evaluate.py             # CLI: run attribution evaluation and insertion/deletion metrics
├── scorer.py               # CLSLensScorer, PatchLensScorer, PatchMapCLSLensScorer
├── metrics.py              # Insertion/deletion curves, AUC
├── postprocess.py          # Neighbor averaging, layer averaging
├── visualize.py            # Heatmap and curve plots
└── config.py               # AttributionConfig dataclass
```

Model structure (timm ViT):
```
patch_embed → _pos_embed → norm_pre → blocks[0..N-1] → norm → fc_norm(cls) → head
```
Forward hooks on `model.blocks[i]` capture `[B, 1+H*W, d_model]`. CLS (position 0) and patches (positions 1:) are split after the hook.

---

## Install

```bash
pip install -e ".[dev]"
```

---

## Data preparation

### 1. Extract ImageNet

```bash
python -m tuned_lens.scripts.prepare_imagenet \
  --train-tar /path/to/ILSVRC2012_img_train.tar \
  --val-tar   /path/to/ILSVRC2012_img_val.tar \
  --output-dir /path/to/imagenet
```

Expected layout:
```
imagenet/
├── train/<synset_id>/*.JPEG    (1,281,167 images)
└── val/<synset_id>/*.JPEG      (50,000 images)
```

### 2. Bounding box annotations (for patch map training)

Download separately from [image-net.org](https://image-net.org/download.php) (ILSVRC 2012 devkit):
- `ILSVRC2012_bbox_train_v2.tar.gz` — train bounding boxes (~544K files, ~42% of train)
- `ILSVRC2012_bbox_val_v3.tgz` — val bounding boxes (all 50K val images)

```
boxes/
├── train/<synset_id>/<synset_id>_<id>.xml
└── val/ILSVRC2012_val_XXXXXXXX.xml
```

---

## Step 1: Train tuned lenses

Trains one lightweight probe per ViT layer (CLS-token mode). The backbone is frozen.

**Training objective:** KL divergence between the probe output and the frozen model's final softmax.

### Lens types

| Type | Description | Config |
|---|---|---|
| `affine` | Single linear layer `d_model → num_classes` | `configs/affine_kld.yaml` |
| `mlp` | Multi-layer MLP with GELU | `configs/mlp_kld.yaml` |
| `affine` + CE loss | Affine with cross-entropy target | `configs/affine_ce.yaml` |
| `affine` + combined | KLD + CE weighted | `configs/affine_combined.yaml` |

### Commands

**CLIP ViT-L/14:**
```bash
python -m tuned_lens.scripts.train \
  --config configs/default.yaml \
  --imagenet-root /path/to/imagenet
```

**ViT-L/16:**
```bash
python -m tuned_lens.scripts.train \
  --config configs/default_vit_l16.yaml \
  --imagenet-root /path/to/imagenet
```

**DeiT3-L/16:**
```bash
python -m tuned_lens.scripts.train \
  --config configs/default_deit3_l16.yaml \
  --imagenet-root /path/to/imagenet
```

**Common overrides** (append to any command above):
```bash
  --lr 5e-4 --batch-size 128 --epochs 20 \
  --target-layers 0 6 12 18 23
```

**Hyperparameter sweep:**
```bash
python -m tuned_lens.scripts.train --config configs/default.yaml --sweep
```

### Outputs

```
outputs/<model_short>/tuned_lens/
├── config.yaml
├── best_lenses/layer_N.pt    # best checkpoint per layer (by val loss)
├── final_lenses/layer_N.pt
└── checkpoints/              # Lightning checkpoints (top-3)
```

Where `<model_short>` is the part of the model name before the first dot, e.g.:
- `vit_large_patch14_clip_224` for CLIP ViT-L/14
- `vit_large_patch16_224` for ViT-L/16
- `deit3_large_patch16_224` for DeiT3-L/16

> The lens configs in `configs/` (e.g. `affine_kld.yaml`, `mlp_kld.yaml`) use the same
> defaults but different loss/architecture combinations. Pass `--config configs/affine_kld.yaml`
> and override `--imagenet-root` and `--output-dir` as needed.

---

## Step 2: Train contrastive patch maps

Trains a per-layer linear map `y = Wx + b` that transforms patch tokens so foreground patches
(inside bounding boxes, overlap ≥ 0.80) move closer to the CLS token while background patches
(overlap ≤ 0.20) move further away.

**Training objective (per layer):**
```
loss = MSE(map(fg_patches), cls) − neg_weight × clamp(MSE(map(bg_patches), cls), max=neg_clip)
```

### Map types

| Type | Parameterization | Config |
|---|---|---|
| `low_rank` | `W = AB`, `A ∈ ℝ^{d×r}`, `B ∈ ℝ^{r×d}`, default rank=128 | `configs/patch_map_lowrank.yaml` |
| `full` | Square matrix `W ∈ ℝ^{d×d}` | `configs/patch_map_full.yaml` |

### Commands

First, edit the paths in the config (or pass them as CLI overrides):

```bash
python -m tuned_lens.scripts.train_patch_map \
  --config configs/patch_map_lowrank.yaml \
  --imagenet-root /path/to/imagenet \
  --bbox-dir-train /path/to/boxes/train \
  --bbox-dir-val   /path/to/boxes/val
```

With model override (for ViT-L/16 or DeiT3-L/16):
```bash
python -m tuned_lens.scripts.train_patch_map \
  --config configs/patch_map_lowrank.yaml \
  --model-name vit_large_patch16_224.augreg_in21k_ft_in1k \
  --imagenet-root /path/to/imagenet \
  --bbox-dir-train /path/to/boxes/train \
  --bbox-dir-val   /path/to/boxes/val \
  --output-dir outputs/vit_large_patch16_224/patch_map_lowrank
```

Smoke test (fast sanity check):
```bash
python -m tuned_lens.scripts.train_patch_map \
  --config configs/patch_map_lowrank.yaml \
  --max-images-per-class 5 --epochs 2 --batch-size 8 \
  --target-layers 12 19
```

### Outputs

```
outputs/<model_short>/patch_map_lowrank/
├── config.yaml
├── best_maps/layer_N.pt     # best checkpoint per layer (by val loss)
├── final_maps/layer_N.pt
└── checkpoints/
```

---

## Step 3: Notebook pipeline

Run these notebooks **before evaluation**. All notebooks have a model selection block at the top — uncomment the model you want to use.

### Required: compute precomputed means

**`notebooks/token_embedding_analysis.ipynb`**

Computes per-layer mean CLS and patch token embeddings over the full val set. Required by the CLS scorer and by notebook 03.

Output: `outputs/<model_short>/precomputed_means.pt`

> Run this once per model before running any evaluation experiment.

### Required (for weighted layer averaging): compute layer weights

**`notebooks/fg_bg_layer_weights.ipynb`**

Measures how well each layer's patch map separates foreground from background patches in cosine-similarity space. Produces a weight per layer.

Output: `tmp/layer_avg_weights_patch_map_full.pt`

> Required only for the `la_weighted` experiments. Uses `patch_map_full` checkpoints.

### Optional analysis notebooks

| Notebook | What it produces |
|---|---|
| `00_extract_embeddings.ipynb` | Saves raw CLS + patch embeddings + bboxes for N val images to `outputs/<model_short>/embeddings/` |
| `01_patch_lens_scoring.ipynb` | Attribution heatmaps via k×k patch lens neighborhoods |
| `02_cls_lens_individual_scoring.ipynb` | Attribution heatmaps via CLS lens on raw patch tokens |
| `03_cls_lens_mean_adjusted_scoring.ipynb` | CLS lens with mean-shift correction (requires precomputed means) |
| `04_patch_map_scoring.ipynb` | Side-by-side: raw patch scores vs. patch-map-transformed scores |
| `val_accuracy_head.ipynb` | Verifies model top-1 / top-5 accuracy on ImageNet val |
| `center_crop_test.ipynb` | Measures accuracy vs. center-crop radius (edge-dependence test) |

---

## Step 4: Evaluation

Runs attribution scoring and insertion/deletion AUC computation for all layers.

### Prerequisites

For each model and experiment, you need:

| Required artifact | Produced by |
|---|---|
| CLS lens checkpoints `outputs/<model_short>/affine_kld/best_lenses/` | Step 1 (tuned lens training) |
| Patch map checkpoints `outputs/<model_short>/patch_map_*/best_maps/` | Step 2 (patch map training) |
| Precomputed means `outputs/<model_short>/precomputed_means.pt` | `token_embedding_analysis.ipynb` |
| Layer weights `tmp/layer_avg_weights_patch_map_full.pt` | `fg_bg_layer_weights.ipynb` (weighted experiments only) |

### The 9 experiment configurations

Experiments vary along two independent axes:

**Axis 1 — Neighbor averaging (`neighbor_avg`)**

| Tag | Setting | Effect |
|---|---|---|
| `no_nb` | disabled | Score each patch independently |
| `nb_emb` | mode=`embedding`, size=3 | Average patch *embeddings* in a 3×3 window before scoring |
| `nb_score` | mode=`score`, size=3 | Average *scores* in a 3×3 window after scoring |

**Axis 2 — Layer averaging (`layer_avg`)**

| Tag | Setting | Effect |
|---|---|---|
| `no_la` | disabled | Use each layer's score map independently |
| `la_uniform` | enabled, `weights_path=""` | Average score maps across layers with equal weights |
| `la_weighted` | enabled, `weights_path=...` | Average score maps weighted by FG/BG discriminability from `fg_bg_layer_weights.ipynb` |

**All 9 combinations:**

| Config file | Neighbor avg | Layer avg |
|---|---|---|
| `exp_no_nb_no_la.yaml` | off | off |
| `exp_no_nb_la_uniform.yaml` | off | uniform |
| `exp_no_nb_la_weighted.yaml` | off | weighted |
| `exp_nb_emb_no_la.yaml` | embedding | off |
| `exp_nb_emb_la_uniform.yaml` | embedding | uniform |
| `exp_nb_emb_la_weighted.yaml` | embedding | weighted |
| `exp_nb_score_no_la.yaml` | score | off |
| `exp_nb_score_la_uniform.yaml` | score | uniform |
| `exp_nb_score_la_weighted.yaml` | score | weighted |

All experiments use `scorer_type: "patch_map_cls"` — the CLS lens applied to patch-map-transformed patch tokens.

### Running a single experiment

Before running, edit the paths in the config file (or override via CLI):
- `lens.cls_lens_dir` — path to trained CLS lens checkpoints
- `lens.patch_map_dir` — path to trained patch map checkpoints
- `lens.means_path` — path to precomputed means `.pt`
- `eval.image_dir` — ImageNet val directory
- `eval.output_dir` — where to save results
- `eval.layer_avg.weights_path` — only needed for `la_weighted` experiments

```bash
python -m attribute_lens.evaluate \
  --config configs/experiments/exp_no_nb_no_la.yaml
```

With CLI overrides:
```bash
python -m attribute_lens.evaluate \
  --config configs/experiments/exp_no_nb_no_la.yaml \
  --image-dir /path/to/imagenet/val \
  --output-dir outputs/experiments/exp_no_nb_no_la \
  --num-images 500
```

### Running all 9 experiments

```bash
for cfg in configs/experiments/exp_*.yaml; do
  name=$(basename "$cfg" .yaml)
  python -m attribute_lens.evaluate \
    --config "$cfg" \
    --output-dir "outputs/experiments/$name"
done
```

### Evaluation output structure

```
outputs/experiments/<exp_name>/
├── summary.json                         # aggregate AUCs across all evaluated images
├── aggregate_patch_map_cls_layer6_curves.png
├── <image_stem>/
│   ├── metrics.json                     # per-image AUCs for every scorer and layer
│   └── patch_map_cls/
│       ├── layer_6_heatmap.png
│       ├── layer_6_curves.png           # insertion + deletion curves
│       ├── layer_6_report.png           # 4-panel combined report
│       └── all_layers_heatmaps.png
...
```

`summary.json` structure:
```json
{
  "images": [
    {
      "image_path": "...",
      "y_hat": 207,
      "scorers": {
        "patch_map_cls": { "12": {"insertion_auc": 0.74, "deletion_auc": 0.28} }
      }
    }
  ],
  "aggregate": {
    "patch_map_cls": {
      "12": {
        "n": 500,
        "insertion_auc_mean": 0.71, "insertion_auc_std": 0.09,
        "deletion_auc_mean":  0.30, "deletion_auc_std":  0.07
      }
    }
  }
}
```

### Metric interpretation

| Metric | Direction | Meaning |
|---|---|---|
| **Insertion AUC** | Higher is better | Attribution correctly identifies the patches that drive the prediction |
| **Deletion AUC** | Lower is better | Removing top-attributed patches quickly destroys prediction confidence |

Random patch ordering gives insertion AUC ≈ deletion AUC ≈ 0.5.

---

## Configuration reference

### Tuned lens (`configs/default.yaml`)

```yaml
model:
  model_name: "vit_large_patch14_clip_224.openai_ft_in1k"  # timm model string
  target_layers: null    # null = all 24 layers; or e.g. [0, 6, 12, 18, 23]

lens:
  lens_type: "affine"    # "affine" | "mlp"
  use_patch_tokens: false  # true = train on k×k patch neighborhoods (not CLS)
  patch_neighbor_size: 3
  patch_border: 2

training:
  lr: 1.0e-3
  loss_type: "kld"       # "kld" | "ce" | "combined"
  max_epochs: 10

output_dir: "outputs/<model_short>/tuned_lens"
```

### Patch map (`configs/patch_map_lowrank.yaml`)

```yaml
patch_map:
  map_type: "low_rank"   # "full" | "low_rank"
  rank: 128
  fg_threshold: 0.80     # patch overlap >= this → foreground
  bg_threshold: 0.20     # patch overlap <= this → background
  neg_weight: 0.5        # repulsion term weight
  neg_clip: 0.3          # clamp on repulsion MSE

training:
  max_epochs: 50
  early_stopping_patience: 10

output_dir: "outputs/<model_short>/patch_map_lowrank"
```

### Evaluation (`configs/attribution_default.yaml`)

```yaml
lens:
  cls_lens_dir: "outputs/.../best_lenses"
  patch_map_dir: "outputs/.../best_maps"
  means_path: "outputs/.../precomputed_means.pt"

eval:
  scorer_type: "patch_map_cls"   # always used in the 9 experiments
  layer_avg:
    enabled: false
    weights_path: ""   # empty = uniform; or path from fg_bg_layer_weights.ipynb
  neighbor_avg:
    enabled: false
    size: 3
    mode: "score"      # "score" | "embedding"
```

---

## Full workflow summary

```
1. Prepare data
   └── prepare_imagenet + download bbox XMLs

2. Train tuned lens (per model)
   └── python -m tuned_lens.scripts.train --config configs/default_<model>.yaml ...

3. Train patch map (per model)
   └── python -m tuned_lens.scripts.train_patch_map --config configs/patch_map_lowrank.yaml ...

4. Notebook pipeline (per model)
   ├── token_embedding_analysis.ipynb  →  outputs/<model_short>/precomputed_means.pt
   └── fg_bg_layer_weights.ipynb       →  tmp/layer_avg_weights_patch_map_full.pt

5. Evaluate (9 experiments per model)
   └── python -m attribute_lens.evaluate --config configs/experiments/exp_<variant>.yaml ...
```
