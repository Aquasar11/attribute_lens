"""CLI entry point for contrastive patch map training.

Usage:
    python -m tuned_lens.scripts.train_patch_map --config configs/patch_map.yaml

Common overrides:
    --imagenet-root /path/to/imagenet
    --bbox-dir-train /path/to/boxes/train
    --bbox-dir-val   /path/to/boxes/val
    --target-layers 12 19
    --map-type low_rank --rank 128
    --epochs 20 --batch-size 32 --lr 1e-3
    --max-images-per-class 50   # quick smoke test
"""

from __future__ import annotations

import argparse
import sys

import pytorch_lightning as pl

from tuned_lens.bbox_data import create_bbox_dataloaders
from tuned_lens.config import PatchMapFullConfig
from tuned_lens.contrastive_trainer import PatchMapLightningModule
from tuned_lens.model import VisionModelWrapper


class _EpochLogger(pl.Callback):
    """Prints newline-terminated progress lines — works correctly when piped."""

    LOG_EVERY_N_STEPS = 500

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.LOG_EVERY_N_STEPS != 0:
            return
        m = trainer.callback_metrics
        step_loss = m.get("train/loss_avg_step", float("nan"))
        print(
            f"[Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}  "
            f"step {batch_idx + 1}/{trainer.num_training_batches}] "
            f"train_loss={step_loss:.4f}",
            flush=True,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        train_loss = m.get("train/loss_avg_epoch", float("nan"))
        print(
            f"[Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}] "
            f"train_loss_epoch={train_loss:.4f}",
            flush=True,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        val_w = m.get("val/layer_weight_avg", float("nan"))
        print(
            f"[Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}] "
            f"val_layer_weight_avg={val_w:.4f}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train contrastive patch map")
    parser.add_argument("--config", type=str, default="configs/patch_map.yaml",
                        help="Path to YAML config file")

    # Data paths
    parser.add_argument("--imagenet-root", type=str, default=None)
    parser.add_argument("--bbox-dir-train", type=str, default=None)
    parser.add_argument("--bbox-dir-val", type=str, default=None)
    parser.add_argument("--max-images-per-class", type=int, default=None)

    # Model
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--target-layers", type=int, nargs="+", default=None)

    # Map architecture
    parser.add_argument("--map-type", type=str, choices=["full", "low_rank"], default=None)
    parser.add_argument("--rank", type=int, default=None)

    # Contrastive loss
    parser.add_argument("--fg-threshold", type=float, default=None)
    parser.add_argument("--bg-threshold", type=float, default=None)
    parser.add_argument("--neg-weight", type=float, default=None)
    parser.add_argument("--neg-clip", type=float, default=None)

    # Training
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Early stopping patience (epochs). Omit or set to 0 to disable.")

    return parser.parse_args()


def apply_overrides(config: PatchMapFullConfig, args: argparse.Namespace) -> None:
    if args.imagenet_root is not None:
        config.data.imagenet_root = args.imagenet_root
    if args.bbox_dir_train is not None:
        config.data.bbox_dir_train = args.bbox_dir_train
    if args.bbox_dir_val is not None:
        config.data.bbox_dir_val = args.bbox_dir_val
    if args.max_images_per_class is not None:
        config.data.max_images_per_class = args.max_images_per_class
    if args.model_name is not None:
        config.model.model_name = args.model_name
    if args.target_layers is not None:
        config.model.target_layers = args.target_layers
    if args.map_type is not None:
        config.patch_map.map_type = args.map_type
    if args.rank is not None:
        config.patch_map.rank = args.rank
    if args.fg_threshold is not None:
        config.patch_map.fg_threshold = args.fg_threshold
    if args.bg_threshold is not None:
        config.patch_map.bg_threshold = args.bg_threshold
    if args.neg_weight is not None:
        config.patch_map.neg_weight = args.neg_weight
    if args.neg_clip is not None:
        config.patch_map.neg_clip = args.neg_clip
    if args.lr is not None:
        config.training.lr = args.lr
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.epochs is not None:
        config.training.max_epochs = args.epochs
    if args.optimizer is not None:
        config.training.optimizer = args.optimizer
    if args.scheduler is not None:
        config.training.scheduler = args.scheduler
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed
    if args.early_stopping_patience is not None:
        config.training.early_stopping_patience = args.early_stopping_patience or None


def _build_callbacks(config: PatchMapFullConfig) -> list:
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=f"{config.output_dir}/checkpoints",
            monitor="val/layer_weight_avg",
            mode="max",
            save_top_k=3,
            filename="epoch{epoch:02d}-val_w{val/layer_weight_avg:.4f}",
            auto_insert_metric_name=False,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        _EpochLogger(),
    ]
    if config.training.early_stopping_patience is not None:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val/layer_weight_avg",
                patience=config.training.early_stopping_patience,
                mode="max",
            )
        )
    return callbacks


def main() -> None:
    import torch
    torch.set_float32_matmul_precision("high")

    args = parse_args()
    config = PatchMapFullConfig.from_yaml(args.config)
    apply_overrides(config, args)

    pl.seed_everything(config.seed, workers=True)

    # Build transforms from a throw-away model wrapper
    tmp_wrapper = VisionModelWrapper(config.model, device="cpu")
    train_transform = tmp_wrapper.get_train_transform()
    val_transform = tmp_wrapper.get_transform()
    tmp_wrapper.cleanup()
    del tmp_wrapper

    train_loader, val_loader = create_bbox_dataloaders(
        config.data,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    module = PatchMapLightningModule(config)
    config.to_yaml(f"{config.output_dir}/config.yaml")

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        gradient_clip_val=config.training.grad_clip_norm,
        val_check_interval=config.training.val_check_interval,
        callbacks=_build_callbacks(config),
        logger=pl.loggers.TensorBoardLogger(
            save_dir=config.output_dir,
            name="tensorboard",
        ),
        precision="16-mixed",
        enable_progress_bar=sys.stdout.isatty(),
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    module.map_bank.save_all(
        f"{config.output_dir}/final_maps",
        metadata={
            "model_name": config.model.model_name,
            "map_type": config.patch_map.map_type,
            "epochs_trained": trainer.current_epoch,
        },
    )
    print(f"\nTraining complete. Outputs saved to {config.output_dir}/")


if __name__ == "__main__":
    main()
