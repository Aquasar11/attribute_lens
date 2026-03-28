"""PyTorch Lightning module for tuned lens training."""

from __future__ import annotations

import os
from typing import Any

import pytorch_lightning as pl
import torch

from .config import TunedLensConfig
from .lens import LensBank
from .loss import get_loss_fn
from .model import VisionModelWrapper


class TunedLensLightningModule(pl.LightningModule):
    """Trains a bank of lenses (one per target layer) to predict the final
    layer's output distribution from intermediate CLS token hidden states.

    The vision backbone is frozen and kept as a plain attribute (not an nn.Module child)
    so Lightning only optimizes the lens parameters.
    """

    def __init__(self, config: TunedLensConfig) -> None:
        super().__init__()
        self.save_hyperparameters({"config": config.to_dict()})
        self.config = config

        # Backbone — plain attribute, NOT nn.Module child
        self.model_wrapper = VisionModelWrapper(config.model, device="cpu")

        # Trainable lens bank
        head_weight, head_bias = (None, None)
        if config.lens.init_from_head:
            head_weight, head_bias = self.model_wrapper.get_head_parameters()

        self.lens_bank = LensBank.create(
            config=config.lens,
            target_layers=self.model_wrapper.target_layers,
            d_model=self.model_wrapper.d_model,
            num_classes=self.model_wrapper.num_classes,
            head_weight=head_weight,
            head_bias=head_bias,
        )

        self.loss_fn = get_loss_fn(config.training)
        self.best_val_loss: dict[int, float] = {
            i: float("inf") for i in self.model_wrapper.target_layers
        }

    def setup(self, stage: str | None = None) -> None:
        """Move the frozen backbone to the correct device."""
        self.model_wrapper.to(self.device)

    def _compute_loss(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        images, gt_labels = batch

        # Extract hidden states from frozen backbone (no gradients)
        hidden_states, target_logits = self.model_wrapper.extract(images)

        total_loss = torch.tensor(0.0, device=self.device)
        target_layers = self.model_wrapper.target_layers

        for layer_idx in target_layers:
            cls_token = hidden_states[layer_idx].to(self.device)
            lens_logits = self.lens_bank(layer_idx, cls_token)

            # gt_labels is only used when loss_type in (ce, combined) and ce_target="gt"
            layer_loss = self.loss_fn(lens_logits, target_logits.detach(), gt_labels)

            self.log(f"{stage}/loss_layer_{layer_idx}", layer_loss, on_step=False, on_epoch=True)
            total_loss = total_loss + layer_loss

        avg_loss = total_loss / len(target_layers)
        self.log(f"{stage}/loss_avg", avg_loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        return avg_loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._compute_loss(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._compute_loss(batch, "val")

    def on_validation_epoch_end(self) -> None:
        """Save best lens per layer based on validation loss."""
        for layer_idx in self.model_wrapper.target_layers:
            key = f"val/loss_layer_{layer_idx}"
            current = self.trainer.callback_metrics.get(key)
            if current is not None and current.item() < self.best_val_loss[layer_idx]:
                self.best_val_loss[layer_idx] = current.item()
                save_dir = os.path.join(self.config.output_dir, "best_lenses")
                self.lens_bank.lenses[str(layer_idx)].save(
                    os.path.join(save_dir, f"layer_{layer_idx}.pt"),
                    metadata={
                        "layer_idx": layer_idx,
                        "val_loss": current.item(),
                        "epoch": self.current_epoch,
                        "model_name": self.config.model.model_name,
                    },
                )

    def configure_optimizers(self) -> dict[str, Any] | torch.optim.Optimizer:
        cfg = self.config.training
        params = self.lens_bank.parameters()

        if cfg.optimizer == "adamw":
            opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "sgd":
            opt = torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
        elif cfg.optimizer == "rmsprop":
            opt = torch.optim.RMSprop(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "nadam":
            opt = torch.optim.NAdam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:  # adam
            opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        if cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
        elif cfg.scheduler == "cosine_warmup":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=max(1, cfg.max_epochs // 3))
        elif cfg.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)
        elif cfg.scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
        elif cfg.scheduler == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=cfg.max_epochs)
        elif cfg.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss_avg",
                    "interval": "epoch",
                },
            }
        else:  # none
            return opt

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
