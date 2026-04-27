"""PyTorch Lightning module for contrastive patch map training.

For each training batch the frozen backbone extracts CLS tokens and patch tokens
at every target layer.  Bounding box annotations supply weak FG/BG labels for
the patch grid.  The PatchMapBank is trained with a contrastive MSE objective:

    loss = MSE(y_fg, cls) - neg_weight * clamp(MSE(y_bg, cls), max=neg_clip)

where y = map(patch_token) is the transformed patch embedding.
"""

from __future__ import annotations

import os
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .bbox_data import classify_patches
from .config import PatchMapFullConfig
from .model import VisionModelWrapper
from .patch_map import PatchMapBank


def contrastive_patch_loss(
    y_pred: torch.Tensor,
    cls_target: torch.Tensor,
    fg_mask: torch.Tensor,
    bg_mask: torch.Tensor,
    neg_weight: float,
    neg_clip: float,
) -> torch.Tensor:
    """Contrastive MSE loss for patch map training.

    Pulls FG patch predictions toward CLS and repels BG predictions from CLS.

    Args:
        y_pred:     [B, N, d] — transformed patch embeddings (N = H*W).
        cls_target: [B, d]    — CLS token at this layer.
        fg_mask:    [B, N] bool — True for foreground patches.
        bg_mask:    [B, N] bool — True for background patches.
        neg_weight: Scaling factor for the repulsion term.
        neg_clip:   Upper bound on the negative MSE before weighting,
                    preventing unbounded negative signal.

    Returns:
        Scalar loss = pos_mse - neg_weight * clamp(neg_mse, max=neg_clip).
    """
    cls_expanded = cls_target.unsqueeze(1).expand_as(y_pred)  # [B, N, d]

    fg_preds = y_pred[fg_mask]          # [N_fg, d]
    fg_targets = cls_expanded[fg_mask]
    pos_loss = F.mse_loss(fg_preds, fg_targets)

    bg_preds = y_pred[bg_mask]          # [N_bg, d]
    bg_targets = cls_expanded[bg_mask]
    neg_loss = F.mse_loss(bg_preds, bg_targets)

    return pos_loss - neg_weight * torch.clamp(neg_loss, max=neg_clip)


class PatchMapLightningModule(pl.LightningModule):
    """Trains a PatchMapBank contrastively with FG/BG bbox supervision.

    The frozen ViT backbone is stored as a plain Python attribute (not an
    nn.Module child) so Lightning only optimizes the PatchMapBank parameters.
    """

    def __init__(self, config: PatchMapFullConfig) -> None:
        super().__init__()
        self.save_hyperparameters({"config": config.to_dict()})
        self.config = config

        # Backbone — plain attribute, NOT nn.Module child (Lightning won't
        # checkpoint or optimize it).
        self.model_wrapper = VisionModelWrapper(config.model, device="cpu")

        self.map_bank = PatchMapBank.create(
            config=config.patch_map,
            target_layers=self.model_wrapper.target_layers,
            d_model=self.model_wrapper.d_model,
        )

        # Best-map tracking uses layer weight (maximize), not val loss (minimize)
        self.best_val_weight: dict[int, float] = {
            i: -1.0 for i in self.model_wrapper.target_layers
        }
        # Per-epoch cosine-similarity accumulators, reset each validation epoch
        self._val_fg_sims: dict[int, list] = {i: [] for i in self.model_wrapper.target_layers}
        self._val_bg_sims: dict[int, list] = {i: [] for i in self.model_wrapper.target_layers}

    def setup(self, stage: str | None = None) -> None:
        """Move the frozen backbone to the training device."""
        self.model_wrapper.to(self.device)

    def on_validation_epoch_start(self) -> None:
        for layer_idx in self.model_wrapper.target_layers:
            self._val_fg_sims[layer_idx] = []
            self._val_bg_sims[layer_idx] = []

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, list[list[dict]]],
        stage: str,
    ) -> torch.Tensor | None:
        images, _labels, bboxes_list = batch

        cls_dict, patch_dict, _logits = self.model_wrapper.extract_cls_and_patches(images)

        B = images.shape[0]
        H, W = self.model_wrapper.patch_grid_size
        patch_size = 224 // H  # e.g. 14 for ViT-L/14 on 224×224

        cfg = self.config.patch_map

        # Classify patches for every image in the batch
        fg_list = []
        bg_list = []
        for bboxes_224 in bboxes_list:
            fg, bg = classify_patches(
                bboxes_224,
                grid_size=H,
                patch_size=patch_size,
                fg_threshold=cfg.fg_threshold,
                bg_threshold=cfg.bg_threshold,
            )
            fg_list.append(torch.from_numpy(fg.reshape(-1)))
            bg_list.append(torch.from_numpy(bg.reshape(-1)))

        fg_mask = torch.stack(fg_list, dim=0).to(self.device)  # [B, H*W]
        bg_mask = torch.stack(bg_list, dim=0).to(self.device)

        has_fg = fg_mask.any().item()
        has_bg = bg_mask.any().item()

        total_loss = torch.tensor(0.0, device=self.device)
        n_layers = 0
        logged_counts = False

        for layer_idx in self.model_wrapper.target_layers:
            patches = patch_dict[layer_idx].to(self.device)  # [B, H, W, d]
            cls = cls_dict[layer_idx].to(self.device)         # [B, d]

            # Flatten spatial dims: [B, H*W, d]
            y = self.map_bank(layer_idx, patches.reshape(B, H * W, -1))

            if not (has_fg and has_bg):
                # Contrastive loss requires both classes; skip this batch
                continue

            layer_loss = contrastive_patch_loss(
                y, cls, fg_mask, bg_mask, cfg.neg_weight, cfg.neg_clip
            )

            self.log(
                f"{stage}/loss_layer_{layer_idx}", layer_loss,
                on_step=False, on_epoch=True,
            )

            if stage == "train" and not logged_counts:
                self.log("train/fg_patches", float(fg_mask.sum()), on_step=True, on_epoch=False)
                self.log("train/bg_patches", float(bg_mask.sum()), on_step=True, on_epoch=False)
                logged_counts = True

            # Accumulate cosine similarities for val-epoch layer weight computation
            if stage == "val":
                with torch.no_grad():
                    y_norm   = F.normalize(y.float(), p=2, dim=-1)          # [B, H*W, d]
                    cls_norm = F.normalize(cls.float(), p=2, dim=-1)        # [B, d]
                    sims     = (y_norm * cls_norm.unsqueeze(1)).sum(dim=-1) # [B, H*W]
                self._val_fg_sims[layer_idx].extend(sims[fg_mask].cpu().tolist())
                self._val_bg_sims[layer_idx].extend(sims[bg_mask].cpu().tolist())

            total_loss = total_loss + layer_loss
            n_layers += 1

        if n_layers == 0:
            return None

        avg_loss = total_loss / n_layers
        self.log(
            f"{stage}/loss_avg", avg_loss,
            prog_bar=True, on_step=(stage == "train"), on_epoch=True,
        )
        return avg_loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor | None:
        return self._compute_loss(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._compute_loss(batch, "val")

    def on_validation_epoch_end(self) -> None:
        """Compute per-layer FG/BG discriminability weights and save best maps."""
        _EPS = 1e-6
        layer_weights = []

        for layer_idx in self.model_wrapper.target_layers:
            fg_vals = self._val_fg_sims[layer_idx]
            bg_vals = self._val_bg_sims[layer_idx]
            if not fg_vals or not bg_vals:
                continue

            fg = torch.tensor(fg_vals)
            bg = torch.tensor(bg_vals)
            w = float(max(fg.mean() - bg.mean(), 0.0) / (fg.std() + bg.std() + _EPS))
            layer_weights.append(w)

            self.log(f"val/layer_weight_{layer_idx}", w, on_epoch=True)

            if w > self.best_val_weight[layer_idx]:
                self.best_val_weight[layer_idx] = w
                save_dir = os.path.join(self.config.output_dir, "best_maps")
                self.map_bank.maps[str(layer_idx)].save(
                    os.path.join(save_dir, f"layer_{layer_idx}.pt"),
                    metadata={
                        "layer_idx": layer_idx,
                        "val_layer_weight": w,
                        "epoch": self.current_epoch,
                        "model_name": self.config.model.model_name,
                        "map_type": self.config.patch_map.map_type,
                    },
                )

        if layer_weights:
            avg_w = sum(layer_weights) / len(layer_weights)
            self.log("val/layer_weight_avg", avg_w, prog_bar=True, on_epoch=True)

    # ------------------------------------------------------------------
    # Optimizer / scheduler — mirrors trainer.py patterns
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any] | torch.optim.Optimizer:
        cfg = self.config.training
        params = self.map_bank.parameters()

        if cfg.optimizer == "adamw":
            opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "sgd":
            opt = torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
        elif cfg.optimizer == "rmsprop":
            opt = torch.optim.RMSprop(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:  # adam
            opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        if cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        elif cfg.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        elif cfg.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="max", factor=0.5, patience=2
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/layer_weight_avg",
                    "interval": "epoch",
                },
            }
        else:  # none
            return opt
