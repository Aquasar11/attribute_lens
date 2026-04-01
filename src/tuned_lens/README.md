# attribute_lens

Tuned lens training for vision transformer interpretability. Trains learnable probes on intermediate ViT layers to predict the final classification output, enabling layer-by-layer analysis.

## Server Setup

### 1. Clone and install

```bash
git clone <repo-url> attribute_lens
cd attribute_lens
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare ImageNet

Assuming your tar files are at `/data/imagenet/`:

```bash
python -m tuned_lens.scripts.prepare_imagenet \
  --train-tar /data/imagenet/ILSVRC2012_img_train.tar \
  --val-tar /data/imagenet/ILSVRC2012_img_val.tar \
  --output-dir /data/imagenet/extracted
```

This extracts and organizes images into the `ImageFolder` layout:
```
/data/imagenet/extracted/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
│   └── ... (1000 class dirs)
└── val/
    ├── n01440764/
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   └── ...
    └── ... (1000 class dirs)
```

The script skips already-extracted classes, so it's safe to re-run if interrupted.

**Note:** The val tar extracts flat images. The script downloads a synset mapping to organize them into class subdirectories. If the server has no internet access, you can provide the mapping file manually with `--val-labels /path/to/val_synset_labels.txt` (50000 lines, one synset ID per image in sorted filename order).

### 3. Create your config

Copy and edit the default config:

```bash
cp configs/default.yaml configs/my_run.yaml
```

At minimum, set `data.imagenet_root` to your extracted path:

```yaml
data:
  imagenet_root: "/data/imagenet/extracted"
```

If you have model weights cached locally (e.g. from huggingface), set:

```yaml
model:
  weights_path: "/path/to/model.safetensors"
  pretrained: false
```

## Testing the Pipeline

Run these tests in order to verify each component before launching full training.

### Test 1: Verify imports

```bash
python -c "
from tuned_lens import AffineLens, MLPLens, LensBank, VisionModelWrapper, TunedLensConfig
print('All imports OK')
"
```

### Test 2: Verify config loading

```bash
python -c "
from tuned_lens.config import TunedLensConfig
cfg = TunedLensConfig.from_yaml('configs/my_run.yaml')
print(f'Model: {cfg.model.model_name}')
print(f'Lens: {cfg.lens.lens_type}')
print(f'ImageNet root: {cfg.data.imagenet_root}')
print(f'Loss: {cfg.training.loss_type}')
"
```

### Test 3: Verify model loads and hooks work

```bash
python -c "
import torch
from tuned_lens.config import ModelConfig
from tuned_lens.model import VisionModelWrapper

wrapper = VisionModelWrapper(ModelConfig(target_layers=[0, 12, 23]), device='cuda')
print(f'd_model={wrapper.d_model}, num_classes={wrapper.num_classes}, num_layers={wrapper.num_layers}')
print(f'Target layers: {wrapper.target_layers}')

# Dummy forward pass
x = torch.randn(2, 3, 224, 224, device='cuda')
hidden_states, logits = wrapper.extract(x)
print(f'Logits shape: {logits.shape}')
for layer_idx, h in hidden_states.items():
    print(f'  Layer {layer_idx}: {h.shape}')
wrapper.cleanup()
print('Model test passed')
"
```

### Test 4: Verify dataset loads

```bash
python -c "
from tuned_lens.config import ModelConfig, DataConfig
from tuned_lens.model import VisionModelWrapper
from tuned_lens.data import create_imagenet_dataloaders

wrapper = VisionModelWrapper(ModelConfig(), device='cpu')
train_loader, val_loader = create_imagenet_dataloaders(
    DataConfig(imagenet_root='/data/imagenet/extracted', batch_size=4, num_workers=0),
    train_transform=wrapper.get_train_transform(),
    val_transform=wrapper.get_transform(),
)
images, labels = next(iter(val_loader))
print(f'Batch: images={images.shape}, labels={labels.shape}')
print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')
wrapper.cleanup()
print('Dataset test passed')
"
```

### Test 5: Dry-run training (1 epoch, 2 layers, small batch)

```bash
python -m tuned_lens.scripts.train \
  --config configs/my_run.yaml \
  --target-layers 0 23 \
  --batch-size 8 \
  --epochs 1 \
  --output-dir outputs/dry_run
```

Check TensorBoard logs:

```bash
tensorboard --logdir outputs/dry_run/tensorboard
```

Verify outputs were created:

```bash
ls outputs/dry_run/final_lenses/
# Should show: layer_0.pt  layer_23.pt
```

## Training

### Single run

```bash
python -m tuned_lens.scripts.train --config configs/my_run.yaml
```

### With CLI overrides

```bash
python -m tuned_lens.scripts.train --config configs/my_run.yaml \
  --lr 5e-4 \
  --batch-size 128 \
  --epochs 20 \
  --target-layers 0 3 6 9 12 15 18 21 23 \
  --loss-type combined \
  --output-dir outputs/combined_loss_run
```

### Initialize lenses from the model's classification head

```bash
python -m tuned_lens.scripts.train --config configs/my_run.yaml \
  --init-from-head \
  --output-dir outputs/head_init_run
```

### Patch lens training

Instead of using the CLS token, train lenses on every patch token in the image center. Each patch's input is a k×k neighborhood of surrounding patch embeddings concatenated together.

```bash
python -m tuned_lens.scripts.train --config configs/my_run.yaml \
  --use-patch-tokens \
  --patch-neighbor-size 3 \
  --patch-border 2 \
  --output-dir outputs/patch_run
```

Or set it permanently in your config:

```yaml
lens:
  use_patch_tokens: true
  patch_neighbor_size: 3   # each patch sees a 3×3 = 9 patch neighborhood
  patch_border: 2          # skip the outer 2 patch-rows/cols on each side
```

For ViT-L/14 (16×16 patches), `patch_border=2` means 12×12=144 valid patches are used per image per step.
The constraint `patch_neighbor_size // 2 <= patch_border` should hold — otherwise a warning is shown and edge neighborhoods will include zero-padded regions.

### Hyperparameter sweep

```bash
python -m tuned_lens.scripts.train --config configs/my_run.yaml --sweep
```

This runs Optuna trials searching over lr, batch size, optimizer, weight decay, and temperature. The best config is saved to `outputs/tuned_lens/best_config.yaml`. Then train with it:

```bash
python -m tuned_lens.scripts.train --config outputs/tuned_lens/best_config.yaml
```

### Evaluate lenses and plot per-layer accuracy

After training, run this to measure how well each lens mimics the final model's predictions on the full val set and generate a bar chart:

```bash
python -m tuned_lens.scripts.eval_lens \
  --lens-dir outputs/dry_run/best_lenses \
  --config outputs/dry_run/config.yaml
```

Or without a saved config:

```bash
python -m tuned_lens.scripts.eval_lens \
  --lens-dir outputs/dry_run/best_lenses \
  --model-name vit_large_patch14_clip_224.openai_ft_in1k \
  --imagenet-root /data/imagenet/extracted
```

The plot is saved to `<lens-dir>/accuracy.png` by default (override with `--output`). It shows per-layer accuracy defined as: fraction of val images where `argmax(lens_i(cls_i)) == argmax(model_final_logits)`.

### Compare tuned lens vs logit lens

`eval_lens_comparison.py` compares the **tuned lens** (your trained probe) against the **logit lens** (the frozen model's own classification head applied directly to intermediate CLS tokens — no learned parameters). Two accuracy metrics are plotted side by side for each layer:

1. **vs Ground Truth** — argmax of each lens's output compared to the true ImageNet label
2. **vs Model Prediction (ŷ)** — argmax compared to the final model's prediction

```bash
python -m tuned_lens.scripts.eval_lens_comparison \
  --lens-dir outputs/affine_kld/best_lenses \
  --config outputs/affine_kld/config.yaml
```

Or with explicit model/data args:

```bash
python -m tuned_lens.scripts.eval_lens_comparison \
  --lens-dir outputs/affine_kld/best_lenses \
  --model-name vit_large_patch14_clip_224.openai_ft_in1k \
  --imagenet-root /data/imagenet/extracted
```

The output plot is saved to `<lens-dir>/comparison.png` (override with `--output`). A table with all four accuracy columns per layer is also printed to stdout.

### Monitor training

```bash
tensorboard --logdir outputs/tuned_lens/tensorboard
```

Logged metrics:
- `train/loss_avg` and `val/loss_avg` — average loss across all target layers
- `train/loss_layer_N` and `val/loss_layer_N` — per-layer loss

## Outputs

After training, the output directory contains:

```
outputs/tuned_lens/
├── config.yaml              # Config snapshot for this run
├── checkpoints/             # Top-3 Lightning checkpoints by val loss
├── best_lenses/             # Best lens per layer (saved independently)
│   ├── layer_0.pt
│   ├── layer_12.pt
│   └── ...
├── final_lenses/            # Lenses at end of training
│   ├── layer_0.pt
│   └── ...
└── tensorboard/             # TensorBoard logs
```

Each `.pt` file contains the lens `state_dict` and metadata (layer index, val loss, epoch, model name).

## Configuration Reference

All parameters in `configs/default.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `model` | `model_name` | `vit_large_patch14_clip_224.openai_ft_in1k` | timm model identifier |
| | `pretrained` | `true` | Download pretrained weights from timm |
| | `weights_path` | `null` | Local `.safetensors` or `.pt` path (overrides pretrained download) |
| | `target_layers` | `null` | Layer indices to probe; `null` = all layers |
| | `freeze_model` | `true` | Freeze backbone (always true for tuned lens) |
| | `patch_mode` | `false` | Auto-set to `true` when `use_patch_tokens=true` |
| `lens` | `lens_type` | `affine` | `affine` (single linear) or `mlp` |
| | `bias` | `true` | Include bias in linear layers |
| | `mlp_hidden_dim` | `null` | MLP hidden dim; `null` defaults to `d_model` |
| | `mlp_num_layers` | `2` | Number of MLP layers |
| | `init_from_head` | `false` | Initialize lens from model's classification head |
| | `init_from_pretrained` | `null` | Path to saved lens weights for initialization |
| | `dropout` | `0.0` | Dropout rate (MLP only) |
| | `use_patch_tokens` | `false` | Train on patch token neighborhoods instead of CLS token |
| | `patch_neighbor_size` | `3` | k for k×k neighborhood; lens input dim = k²×d_model |
| | `patch_border` | `2` | Exclude patches within this many steps of the image edge |
| `data` | `imagenet_root` | `""` | Path to extracted ImageNet |
| | `batch_size` | `64` | Training batch size |
| | `num_workers` | `4` | DataLoader workers |
| | `max_images_per_class` | `null` | Max training images per class (`null` = all ~1300); val set is always unaffected |
| `training` | `lr` | `1e-3` | Learning rate |
| | `weight_decay` | `0.0` | Weight decay |
| | `optimizer` | `adam` | `adam`, `adamw`, `sgd`, `rmsprop`, `nadam` |
| | `scheduler` | `cosine` | `cosine`, `cosine_warmup`, `step`, `exponential`, `linear`, `plateau`, `none` |
| | `warmup_steps` | `100` | LR warmup steps |
| | `max_epochs` | `10` | Maximum training epochs |
| | `gradient_accumulation_steps` | `1` | Gradient accumulation batches |
| | `val_check_interval` | `1.0` | Validation frequency: float = fraction of epoch, int = every N steps |
| | `loss_type` | `kld` | `kld`, `ce`, or `combined` |
| | `ce_weight` | `0.1` | CE weight when `loss_type=combined` |
| | `temperature` | `1.0` | Softmax temperature for KLD |
| | `grad_clip_norm` | `1.0` | Gradient clipping norm |
| `sweep` | `n_trials` | `50` | Number of Optuna trials |
| | `max_epochs_per_trial` | `3` | Epochs per sweep trial |
| | `optimizer_choices` | all 5 | Optimizers Optuna can pick from |
| | `scheduler_choices` | all 7 | Schedulers Optuna can pick from |
