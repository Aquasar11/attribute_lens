# attribute_lens — Patch Attribution via Tuned Lenses

Generates per-patch attribution heatmaps and computes insertion/deletion AUC
scores for frozen Vision Transformers (ViT), using two types of pre-trained
learnable probes ("lenses"):

| Scorer | Training | Input to lens |
|--------|----------|---------------|
| **CLS** | CLS-token mode | Patch token adjusted toward the CLS distribution |
| **Patch** | Patch-neighborhood mode | 3×3 (k×k) neighborhood of patch tokens |

---

## Prerequisites

1. Trained lens checkpoints from `src/tuned_lens/` training:
   - CLS lenses: `outputs/affine_kld/best_lenses/layer_*.pt`
   - Patch lenses: `outputs/patch_affine_kld/best_lenses/layer_*.pt`
2. Precomputed mean embeddings file for the CLS scorer (see below).
3. Images to evaluate (single file, list, or directory).

### Additional Python dependencies

Install if not already present:

```bash
pip install scikit-learn matplotlib
```

---

## Precomputed Means File

The CLS scorer uses mean embeddings computed over the validation set to shift
patch token representations toward the CLS token distribution.

**Required file structure** (`outputs/precomputed_means.pt`):

```python
{
    "cls_means": {
        layer_idx: Tensor[d_model],    # mean CLS token embedding at that layer
        ...
    },
    "token_means": {
        layer_idx: Tensor[H*W, d_model],  # mean patch-token embedding per position, per layer
        ...
    }
}
```

For ViT-L/14 on 224×224: `d_model=1024`, `H*W=256` (16×16 patch grid).

**Example generation script** (run once on the remote server):

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tuned_lens.config import ModelConfig
from tuned_lens.model import VisionModelWrapper
from tuned_lens.data import create_imagenet_dataloaders

TARGET_LAYERS = list(range(24))  # or a specific subset
IMAGENET_ROOT = "/path/to/imagenet"
OUTPUT_PATH = "outputs/precomputed_means.pt"
DEVICE = "cuda"

model_cfg = ModelConfig(patch_mode=True, target_layers=TARGET_LAYERS)
wrapper = VisionModelWrapper(model_cfg, device=DEVICE)

_, val_loader = create_imagenet_dataloaders(
    imagenet_root=IMAGENET_ROOT,
    batch_size=64,
    num_workers=4,
)

H_p, W_p = wrapper.patch_grid_size
d = wrapper.d_model

# Running sums
cls_sums   = {l: torch.zeros(d)           for l in TARGET_LAYERS}
token_sums = {l: torch.zeros(H_p * W_p, d) for l in TARGET_LAYERS}
counts = {l: 0 for l in TARGET_LAYERS}

with torch.no_grad():
    for images, _ in val_loader:
        images = images.to(DEVICE)
        patch_states, logits = wrapper.extract_patches(images)
        B = images.shape[0]
        for l in TARGET_LAYERS:
            # CLS mean: run in CLS mode or use the last-layer CLS token proxy
            # (Re-run with patch_mode=False to get actual CLS token if needed)
            ps = patch_states[l]  # [B, H_p, W_p, d]
            flat = ps.reshape(B, H_p * W_p, d).cpu()
            token_sums[l] += flat.sum(dim=0)
            counts[l] += B

# Compute means
token_means = {l: token_sums[l] / counts[l] for l in TARGET_LAYERS}

# For cls_means: re-run with patch_mode=False
model_cfg_cls = ModelConfig(patch_mode=False, target_layers=TARGET_LAYERS)
wrapper_cls = VisionModelWrapper(model_cfg_cls, device=DEVICE)
cls_sums2 = {l: torch.zeros(d) for l in TARGET_LAYERS}
counts2 = {l: 0 for l in TARGET_LAYERS}

with torch.no_grad():
    for images, _ in val_loader:
        images = images.to(DEVICE)
        hidden, _ = wrapper_cls.extract(images)
        B = images.shape[0]
        for l in TARGET_LAYERS:
            cls_sums2[l] += hidden[l].cpu().sum(dim=0)
            counts2[l] += B

cls_means = {l: cls_sums2[l] / counts2[l] for l in TARGET_LAYERS}

torch.save({"cls_means": cls_means, "token_means": token_means}, OUTPUT_PATH)
print(f"Saved means to {OUTPUT_PATH}")
```

---

## Configuration

All settings live in a YAML file (see `configs/attribution_default.yaml`).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model.model_name` | str | `vit_large_patch14_clip_224.openai_ft_in1k` | timm model identifier |
| `model.weights_path` | str\|null | `null` | Local safetensors/pt path; null = timm hub |
| `model.target_layers` | list\[int\]\|null | `null` | Restrict to these layers; null = all available |
| `lens.cls_lens_dir` | str | `""` | Dir with `layer_*.pt` CLS lens checkpoints |
| `lens.patch_lens_dir` | str | `""` | Dir with `layer_*.pt` patch lens checkpoints |
| `lens.patch_neighbor_size` | int | `3` | Neighborhood size *k* (must match training) |
| `lens.patch_border` | int | `2` | Border exclusion radius (must match training) |
| `lens.means_path` | str | `""` | Path to precomputed means `.pt` file |
| `eval.image_paths` | list | `[]` | Explicit image file paths |
| `eval.image_dir` | str | `""` | Directory scanned recursively for images |
| `eval.num_images` | int\|null | `null` | Limit total images; null = all |
| `eval.output_dir` | str | `outputs/attribution` | Root output directory |
| `eval.device` | str | `cuda` | `"cuda"` or `"cpu"` |
| `eval.scorer_type` | str | `both` | `"cls"` \| `"patch"` \| `"both"` |
| `eval.blur_kernel_size` | int | `55` | Gaussian blur kernel for baseline |
| `eval.blur_sigma` | float | `10.0` | Gaussian blur sigma |
| `eval.heatmap_colormap` | str | `hot` | Matplotlib colormap for heatmaps |
| `eval.heatmap_alpha` | float | `0.6` | Heatmap overlay opacity |
| `eval.plot_dpi` | int | `150` | DPI for saved figures |
| `seed` | int | `42` | RNG seed for image sampling |

---

## CLI Usage

```bash
# Evaluate using config file
python -m attribute_lens.evaluate --config configs/attribution_default.yaml

# Single image with CLI overrides
python -m attribute_lens.evaluate \
    --config configs/attribution_default.yaml \
    --image /path/to/image.JPEG \
    --output-dir outputs/test_run \
    --scorer-type cls \
    --layer 6 --layer 12 --layer 18

# Evaluate 100 random images from a directory
python -m attribute_lens.evaluate \
    --config configs/attribution_default.yaml \
    --image-dir /path/to/imagenet/val \
    --num-images 100 \
    --scorer-type both
```

---

## Scorer Math

### CLS Lens Scorer

For each patch at position **p** and layer **L**:

```
adjusted[p]  = h[p] - mean_token[L][p] + mean_cls[L]
embedding[p] = cls_lens_L(adjusted[p])      # [d_model] — lens output
score[p]     = softmax(head(embedding[p]))[ŷ]
```

where `h[p]` is the patch token embedding, `mean_token[L][p]` is the mean
embedding of that patch position over the val set, and `mean_cls[L]` is the
mean CLS token embedding. This shifts the patch token into the CLS distribution
before querying the CLS-trained lens. The frozen classification head (`apply_head`)
converts the lens embedding to class logits before softmax.

### Patch Lens Scorer

For each valid center patch **(i, j)** and layer **L**:

```
neighborhood = concat(patch tokens in k×k grid around (i,j))  # [k²·d_model]
embedding    = patch_lens_L(neighborhood)                       # [d_model]
score[i,j]   = softmax(head(embedding))[ŷ]
```

Patches within `patch_border` steps of the image edge receive `NaN` and are
ranked last in insertion/deletion curves.

---

## Output Structure

```
outputs/attribution/
├── summary.json                          # aggregate AUCs (mean, std, median) across all images
├── aggregate_cls_layer6_curves.png       # mean ± std insertion/deletion for CLS scorer, layer 6
├── aggregate_patch_layer6_curves.png     # same for patch scorer
├── <image_stem>/
│   ├── metrics.json                      # per-image AUCs for all scorers and layers
│   ├── cls/
│   │   ├── layer_6_heatmap.png           # score heatmap overlay
│   │   ├── layer_6_curves.png            # insertion + deletion curves
│   │   ├── layer_6_report.png            # 4-panel combined report
│   │   ├── layer_12_*.png
│   │   └── all_layers_heatmaps.png       # grid of all layers at once
│   └── patch/
│       └── layer_*.png
...
```

### `summary.json` structure

```json
{
  "images": [
    {
      "image_path": "...",
      "y_hat": 207,
      "scorers": {
        "cls":   { "6": {"insertion_auc": 0.72, "deletion_auc": 0.31} },
        "patch": { "6": {"insertion_auc": 0.69, "deletion_auc": 0.34} }
      }
    }
  ],
  "aggregate": {
    "cls": {
      "6": {
        "n": 100,
        "insertion_auc_mean": 0.703, "insertion_auc_std": 0.09, "insertion_auc_median": 0.71,
        "deletion_auc_mean":  0.312, "deletion_auc_std":  0.07, "deletion_auc_median":  0.30
      }
    }
  }
}
```

---

## Metric Interpretation

| Metric | Direction | Meaning |
|--------|-----------|---------|
| **Insertion AUC** | Higher is better | Attribution correctly identifies patches that drive the prediction |
| **Deletion AUC** | Lower is better | Removing attributed patches quickly destroys prediction confidence |

Random patch ordering gives insertion AUC ≈ deletion AUC ≈ 0.5.
