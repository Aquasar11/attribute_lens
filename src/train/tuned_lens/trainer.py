"""PyTorch Lightning module for tuned lens training."""

from __future__ import annotations

import os
import warnings
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .config import TunedLensConfig
from .lens import LensBank
from .loss import get_loss_fn
from .model import VisionModelWrapper


class TunedLensLightningModule(pl.LightningModule):
    """Trains a bank of lenses (one per target layer) to predict the final
    layer's output distribution from intermediate hidden states.

    Supports two modes controlled by ``config.lens.use_patch_tokens``:

    - **CLS mode** (default): one forward pass through the lens per image, using
      the CLS token hidden state at each target layer.
    - **Patch mode**: for each image, all valid center patches are passed through
      the lens (one (k×k neighborhood) → logits per patch), and the loss is averaged
      across all valid patches.  Valid patches are those at least ``patch_border``
      steps away from the image edges.

    The vision backbone is frozen and kept as a plain attribute (not an nn.Module
    child) so Lightning only optimizes the lens parameters.
    """

    def __init__(self, config: TunedLensConfig) -> None:
        super().__init__()
        self.save_hyperparameters({"config": config.to_dict()})
        self.config = config

        # Auto-enable patch_mode on the model config when using patch tokens
        if config.lens.use_patch_tokens:
            config.model.patch_mode = True
            k = config.lens.patch_neighbor_size
            b = config.lens.patch_border
            if k // 2 > b:
                warnings.warn(
                    f"patch_neighbor_size // 2 ({k // 2}) > patch_border ({b}): "
                    "edge neighborhoods will include zero-padded regions."
                )

        # Backbone — plain attribute, NOT nn.Module child
        self.model_wrapper = VisionModelWrapper(config.model, device="cpu")

        # Trainable lens bank
        k = config.lens.patch_neighbor_size if config.lens.use_patch_tokens else 1
        self.lens_bank = LensBank.create(
            config=config.lens,
            target_layers=self.model_wrapper.target_layers,
            d_model=self.model_wrapper.d_model,
            num_classes=self.model_wrapper.num_classes,
            patch_neighbor_size=k,
        )

        self.loss_fn = get_loss_fn(config.training)
        self.best_val_loss: dict[int, float] = {
            i: float("inf") for i in self.model_wrapper.target_layers
        }

    def setup(self, stage: str | None = None) -> None:
        """Move the frozen backbone to the correct device."""
        self.model_wrapper.to(self.device)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_loss(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        images, gt_labels = batch

        if self.config.lens.use_patch_tokens:
            return self._compute_loss_patch(images, gt_labels, stage)
        else:
            return self._compute_loss_cls(images, gt_labels, stage)

    def _compute_loss_cls(
        self,
        images: torch.Tensor,
        gt_labels: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        """CLS-token loss with the tuned-lens formulation.

        The lens maps each intermediate CLS embedding to the d_model embedding
        space (predicting the final-layer CLS token), then the frozen classification
        head is applied to obtain logits.  This is compared against the model's
        actual output logits via KLD.
        """
        hidden_states, target_logits = self.model_wrapper.extract(images)

        total_loss = torch.tensor(0.0, device=self.device)
        target_layers = self.model_wrapper.target_layers

        for layer_idx in target_layers:
            cls_token = hidden_states[layer_idx].to(self.device)
            lens_embedding = self.lens_bank(layer_idx, cls_token)           # [B, d_model]
            lens_logits = self.model_wrapper.apply_head(lens_embedding)    # [B, num_classes]

            layer_loss = self.loss_fn(lens_logits, target_logits.detach(), gt_labels)

            self.log(f"{stage}/loss_layer_{layer_idx}", layer_loss, on_step=False, on_epoch=True)
            total_loss = total_loss + layer_loss

        avg_loss = total_loss / len(target_layers)
        self.log(f"{stage}/loss_avg", avg_loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        return avg_loss

    def _compute_loss_patch(
        self,
        images: torch.Tensor,
        gt_labels: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        """Patch-neighborhood loss.

        For each valid patch (i, j) in the center region, the k×k neighborhood of
        patch tokens is flattened and passed through the lens.  Loss is averaged
        over all valid patches and all target layers.
        """
        patch_states, target_logits = self.model_wrapper.extract_patches(images)

        B = images.shape[0]
        k = self.config.lens.patch_neighbor_size
        half_k = k // 2
        border = self.config.lens.patch_border
        d_model = self.model_wrapper.d_model
        target_logits = target_logits.detach()

        total_loss = torch.tensor(0.0, device=self.device)
        target_layers = self.model_wrapper.target_layers

        for layer_idx in target_layers:
            patches = patch_states[layer_idx].to(self.device)  # [B, H, W, d_model]
            H, W = patches.shape[1], patches.shape[2]

            # Zero-pad so every valid center patch has a full k×k neighborhood
            # patches: [B, H, W, d_model] → permute → [B, d_model, H, W] for F.pad
            padded = F.pad(
                patches.permute(0, 3, 1, 2),  # [B, d_model, H, W]
                (half_k, half_k, half_k, half_k),
            ).permute(0, 2, 3, 1)  # [B, H+2*half_k, W+2*half_k, d_model]

            # Build neighborhood tensors for all valid patches
            neighborhoods = []
            for i in range(border, H - border):
                for j in range(border, W - border):
                    pi, pj = i + half_k, j + half_k  # coords in padded grid
                    nb = padded[:, pi - half_k: pi + half_k + 1,
                                   pj - half_k: pj + half_k + 1, :]  # [B, k, k, d_model]
                    neighborhoods.append(nb.reshape(B, k * k * d_model))

            num_valid = len(neighborhoods)
            if num_valid == 0:
                continue

            # [num_valid*B, k*k*d_model]
            nb_tensor = torch.stack(neighborhoods, dim=0).view(num_valid * B, k * k * d_model)

            lens_embedding = self.lens_bank(layer_idx, nb_tensor)               # [num_valid*B, d_model]
            lens_logits = self.model_wrapper.apply_head(lens_embedding)        # [num_valid*B, num_classes]

            # Replicate targets and labels for each patch
            target_rep = target_logits.unsqueeze(0).expand(num_valid, -1, -1).reshape(num_valid * B, -1)
            gt_rep = gt_labels.unsqueeze(0).expand(num_valid, -1).reshape(num_valid * B)

            layer_loss = self.loss_fn(lens_logits, target_rep, gt_rep) / num_valid

            self.log(f"{stage}/loss_layer_{layer_idx}", layer_loss, on_step=False, on_epoch=True)
            total_loss = total_loss + layer_loss

        avg_loss = total_loss / len(target_layers)
        self.log(f"{stage}/loss_avg", avg_loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        return avg_loss

    # ------------------------------------------------------------------

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
