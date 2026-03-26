"""Optuna hyperparameter sweep for tuned lens training."""

from __future__ import annotations

import copy
import os

import optuna
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback

from .config import TunedLensConfig
from .data import create_imagenet_dataloaders
from .model import VisionModelWrapper
from .trainer import TunedLensLightningModule


def create_objective(
    base_config: TunedLensConfig,
    train_loader: pl.LightningDataModule | None = None,
    val_loader: pl.LightningDataModule | None = None,
):
    """Returns an Optuna objective function.

    If dataloaders are not provided, they are created from the config.
    Passing them avoids re-loading the dataset for each trial.
    """

    # Pre-build dataloaders so they are shared across trials
    if train_loader is None or val_loader is None:
        wrapper = VisionModelWrapper(base_config.model, device="cpu")
        train_loader, val_loader = create_imagenet_dataloaders(
            base_config.data,
            train_transform=wrapper.get_train_transform(),
            val_transform=wrapper.get_transform(),
        )
        wrapper.cleanup()

    def objective(trial: optuna.Trial) -> float:
        config = copy.deepcopy(base_config)
        sweep = config.sweep

        # Sample hyperparameters
        config.training.lr = trial.suggest_float(
            "lr", sweep.lr_range[0], sweep.lr_range[1], log=True
        )
        config.data.batch_size = trial.suggest_categorical(
            "batch_size", sweep.batch_size_choices
        )
        config.training.optimizer = trial.suggest_categorical(
            "optimizer", sweep.optimizer_choices
        )
        config.training.weight_decay = trial.suggest_float(
            "weight_decay", sweep.weight_decay_range[0], sweep.weight_decay_range[1], log=True
        )
        config.training.temperature = trial.suggest_float(
            "temperature", sweep.temperature_range[0], sweep.temperature_range[1]
        )

        # Use reduced epochs for sweep
        config.training.max_epochs = sweep.max_epochs_per_trial

        # Per-trial output dir
        config.output_dir = os.path.join(base_config.output_dir, "sweep", f"trial_{trial.number}")

        module = TunedLensLightningModule(config)

        trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            accumulate_grad_batches=config.training.gradient_accumulation_steps,
            gradient_clip_val=config.training.grad_clip_norm,
            callbacks=[
                PyTorchLightningPruningCallback(trial, monitor="val/loss_avg"),
            ],
            enable_checkpointing=False,
            logger=pl.loggers.TensorBoardLogger(
                save_dir=config.output_dir, name="tensorboard"
            ),
            enable_progress_bar=False,
        )

        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        return trainer.callback_metrics["val/loss_avg"].item()

    return objective


def run_sweep(config: TunedLensConfig) -> optuna.Study:
    """Run a full Optuna hyperparameter sweep and return the study."""
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(create_objective(config), n_trials=config.sweep.n_trials)

    print(f"\nBest trial #{study.best_trial.number}:")
    print(f"  Value (val/loss_avg): {study.best_value:.6f}")
    print(f"  Params: {study.best_trial.params}")

    return study
