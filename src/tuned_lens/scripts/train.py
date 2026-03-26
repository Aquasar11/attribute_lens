"""CLI entry point for tuned lens training."""

from __future__ import annotations

import argparse

import pytorch_lightning as pl

from tuned_lens.config import TunedLensConfig
from tuned_lens.data import create_imagenet_dataloaders
from tuned_lens.model import VisionModelWrapper
from tuned_lens.sweep import run_sweep
from tuned_lens.trainer import TunedLensLightningModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tuned lens for vision models")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--sweep", action="store_true",
                        help="Run Optuna hyperparameter sweep instead of single training")

    # CLI overrides for common params
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--imagenet-root", type=str, default=None)
    parser.add_argument("--target-layers", type=int, nargs="+", default=None)
    parser.add_argument("--loss-type", type=str, choices=["kld", "ce", "combined"], default=None)
    parser.add_argument("--lens-type", type=str, choices=["affine", "mlp"], default=None)
    parser.add_argument("--init-from-head", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def apply_overrides(config: TunedLensConfig, args: argparse.Namespace) -> None:
    """Apply CLI argument overrides to the config."""
    if args.lr is not None:
        config.training.lr = args.lr
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.epochs is not None:
        config.training.max_epochs = args.epochs
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.model_name is not None:
        config.model.model_name = args.model_name
    if args.imagenet_root is not None:
        config.data.imagenet_root = args.imagenet_root
    if args.target_layers is not None:
        config.model.target_layers = args.target_layers
    if args.loss_type is not None:
        config.training.loss_type = args.loss_type
    if args.lens_type is not None:
        config.lens.lens_type = args.lens_type
    if args.init_from_head is True:
        config.lens.init_from_head = True
    if args.seed is not None:
        config.seed = args.seed


def main() -> None:
    args = parse_args()
    config = TunedLensConfig.from_yaml(args.config)
    apply_overrides(config, args)

    pl.seed_everything(config.seed, workers=True)

    if args.sweep:
        study = run_sweep(config)
        # Save best config
        best_config = TunedLensConfig.from_yaml(args.config)
        apply_overrides(best_config, args)
        for k, v in study.best_trial.params.items():
            if k == "lr":
                best_config.training.lr = v
            elif k == "batch_size":
                best_config.data.batch_size = v
            elif k == "optimizer":
                best_config.training.optimizer = v
            elif k == "weight_decay":
                best_config.training.weight_decay = v
            elif k == "temperature":
                best_config.training.temperature = v
        best_config.to_yaml(f"{config.output_dir}/best_config.yaml")
        print(f"Best config saved to {config.output_dir}/best_config.yaml")
        return

    # Create a temporary wrapper just for transforms
    tmp_wrapper = VisionModelWrapper(config.model, device="cpu")
    train_transform = tmp_wrapper.get_train_transform()
    val_transform = tmp_wrapper.get_transform()
    tmp_wrapper.cleanup()
    del tmp_wrapper

    train_loader, val_loader = create_imagenet_dataloaders(
        config.data,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    module = TunedLensLightningModule(config)

    # Save config alongside outputs
    config.to_yaml(f"{config.output_dir}/config.yaml")

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accumulate_grad_batches=config.training.gradient_accumulation_steps,
        gradient_clip_val=config.training.grad_clip_norm,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=f"{config.output_dir}/checkpoints",
                monitor="val/loss_avg",
                mode="min",
                save_top_k=3,
                filename="epoch{epoch:02d}-val_loss{val/loss_avg:.4f}",
                auto_insert_metric_name=False,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.EarlyStopping(
                monitor="val/loss_avg",
                patience=5,
                mode="min",
            ),
        ],
        logger=pl.loggers.TensorBoardLogger(
            save_dir=config.output_dir,
            name="tensorboard",
        ),
        precision="16-mixed",
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save final lenses
    module.lens_bank.save_all(
        f"{config.output_dir}/final_lenses",
        metadata={
            "model_name": config.model.model_name,
            "loss_type": config.training.loss_type,
            "epochs_trained": trainer.current_epoch,
        },
    )
    print(f"\nTraining complete. Outputs saved to {config.output_dir}/")


if __name__ == "__main__":
    main()
