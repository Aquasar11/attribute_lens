"""Vision model wrapper with hook-based hidden state extraction."""

from __future__ import annotations

from typing import Any

import timm
import timm.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.hooks import RemovableHandle
from torchvision import transforms as T

from .config import ModelConfig


class VisionModelWrapper:
    """Wraps a timm ViT model to extract CLS token hidden states at each layer.

    The model is always frozen and in eval mode. Forward hooks on
    ``model.blocks[i]`` capture the CLS token (position 0) after each target block.
    """

    def __init__(self, config: ModelConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = device
        self._hooks: list[RemovableHandle] = []
        self._hidden_states: dict[int, torch.Tensor] = {}
        self._full_sequence: bool = False
        self._full_states: dict[int, torch.Tensor] = {}

        self._load_model()

        # Infer model properties
        if self._custom_head is not None:
            self.d_model: int = self.model.num_features
            self.num_classes: int = self._custom_head.out_features
        else:
            self.d_model: int = self.model.head.in_features
            self.num_classes: int = self.model.head.out_features
        self.num_layers: int = len(self.model.blocks)

        # Resolve target layers
        if config.target_layers is not None:
            self.target_layers = list(config.target_layers)
        else:
            self.target_layers = list(range(self.num_layers))

        self._register_hooks()

    def _load_model(self) -> None:
        """Create and freeze the timm model."""
        self._cls_patch_avg_head: bool = False
        self._custom_head: nn.Linear | None = None

        if self.config.weights_path:
            # Load from local safetensors / checkpoint
            self.model = timm.create_model(
                self.config.model_name, pretrained=False
            )
            if self.config.weights_path.endswith(".safetensors"):
                state_dict = load_file(self.config.weights_path)
            else:
                state_dict = torch.load(
                    self.config.weights_path, map_location="cpu", weights_only=True
                )
            self.model.load_state_dict(state_dict, strict=False)
        else:
            if self.config.head_weights_path:
                _sd = torch.load(
                    self.config.head_weights_path, map_location="cpu", weights_only=True
                )
                head_in = _sd["weight"].shape[1]
                head_out = _sd["weight"].shape[0]

                # Use global_pool='' so forward() returns the full token sequence
                # [B, 1+HW, d], enabling both CLS-only and CLS+patch_avg heads.
                self.model = timm.create_model(
                    self.config.model_name,
                    pretrained=self.config.pretrained,
                    num_classes=0,
                    global_pool="",
                )
                backbone_dim = self.model.num_features
                if head_in not in (backbone_dim, 2 * backbone_dim):
                    raise ValueError(
                        f"Head weight input dim {head_in} is incompatible with backbone dim "
                        f"{backbone_dim}. Expected {backbone_dim} (CLS-only) or "
                        f"{2 * backbone_dim} (CLS+patch_avg, e.g. HuggingFace DINOv2)."
                    )
                self._cls_patch_avg_head = head_in == 2 * backbone_dim
                self._custom_head = nn.Linear(head_in, head_out)
                self._custom_head.weight.data.copy_(_sd["weight"])
                self._custom_head.bias.data.copy_(_sd["bias"])
            else:
                self.model = timm.create_model(
                    self.config.model_name,
                    pretrained=self.config.pretrained,
                )

        self.model.eval()
        if self.config.freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False
            if self._custom_head is not None:
                for param in self._custom_head.parameters():
                    param.requires_grad = False
        self.model.to(self.device)
        if self._custom_head is not None:
            self._custom_head.to(self.device)

    def _register_hooks(self) -> None:
        """Register forward hooks to capture CLS tokens at target layers."""
        self._remove_hooks()
        for layer_idx in self.target_layers:
            handle = self.model.blocks[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(handle)

    def _make_hook(self, layer_idx: int) -> Any:
        def hook_fn(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            # output shape: [B, seq_len, d_model]
            if self.config.patch_mode:
                # Capture all patch tokens (exclude CLS at position 0)
                self._hidden_states[layer_idx] = output[:, 1:, :].detach()
            else:
                # Capture CLS token only
                self._hidden_states[layer_idx] = output[:, 0, :].detach()
        return hook_fn

    def _make_full_seq_hook(self, layer_idx: int) -> Any:
        def hook_fn(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self._full_states[layer_idx] = output.detach()  # [B, 1+H*W, d]
        return hook_fn

    def enable_full_sequence_mode(self) -> None:
        """Switch to persistent full-sequence hooks (CLS + all patches in one pass).

        After this call, ``extract_cls_and_patches`` no longer re-registers hooks
        on every invocation — it simply runs the forward pass and slices the
        already-captured full-sequence tensors.  This eliminates the per-batch
        hook teardown / re-install overhead.
        """
        self._full_sequence = True
        self._full_states = {}
        self._remove_hooks()
        for layer_idx in self.target_layers:
            handle = self.model.blocks[layer_idx].register_forward_hook(
                self._make_full_seq_hook(layer_idx)
            )
            self._hooks.append(handle)

    def apply_head(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """Map a CLS-token embedding [B, d_model] → [B, num_classes].

        For standard models: delegates to model.head (nn.Linear).
        For CLS+patch_avg heads (e.g. HuggingFace DINOv2): uses only the
        CLS-column slice of the weight matrix so the input dimension matches.
        """
        if self._custom_head is not None:
            if self._cls_patch_avg_head:
                W = self._custom_head.weight[:, : self.d_model]
                return F.linear(cls_embedding, W, self._custom_head.bias)
            return self._custom_head(cls_embedding)
        return self.model.head(cls_embedding)

    def _run_model(self, images: torch.Tensor) -> torch.Tensor:
        """Run model forward and return classification logits."""
        if self._custom_head is not None:
            seq = self.model(images)  # [B, 1+HW, d] — global_pool='' returns full sequence
            cls = seq[:, 0]
            if self._cls_patch_avg_head:
                patch_avg = seq[:, 1:].mean(dim=1)
                features = torch.cat([cls, patch_avg], dim=-1)
            else:
                features = cls
            return self._custom_head(features)
        return self.model(images)

    def _remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def to(self, device: str | torch.device) -> VisionModelWrapper:
        self.device = str(device)
        self.model.to(device)
        if self._custom_head is not None:
            self._custom_head.to(device)
        return self

    @property
    def patch_grid_size(self) -> tuple[int, int]:
        """Return (H, W) patch grid dimensions, e.g. (16, 16) for ViT-L/14 on 224×224."""
        return self.model.patch_embed.grid_size

    @torch.no_grad()
    def extract_patches(self, images: torch.Tensor) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Run forward pass in patch_mode, returning per-layer patch tensors.

        Must be called when the wrapper was created with ``config.patch_mode=True``.

        Returns:
            patch_states: {layer_idx: [B, H, W, d_model]} reshaped patch tokens.
            target_logits: Final model output [B, num_classes].
        """
        self._hidden_states.clear()
        logits = self._run_model(images)
        H, W = self.patch_grid_size
        patches = {
            k: v.reshape(v.shape[0], H, W, v.shape[-1])
            for k, v in self._hidden_states.items()
        }
        return patches, logits

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Run forward pass, return hidden states and target logits.

        Args:
            images: Input images [B, C, H, W].

        Returns:
            hidden_states: {layer_idx: cls_token [B, d_model]} for each target layer.
            target_logits: Final model output [B, num_classes].
        """
        self._hidden_states.clear()
        logits = self._run_model(images)
        return dict(self._hidden_states), logits

    @torch.no_grad()
    def extract_cls_and_patches(
        self, images: torch.Tensor
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], torch.Tensor]:
        """Extract CLS tokens and patch grids in a single forward pass.

        If ``enable_full_sequence_mode()`` has been called, uses the persistent
        hooks already installed — no per-call hook teardown/reinstall overhead.
        Otherwise, temporarily installs full-sequence hooks and restores original
        hooks afterwards (one-off / notebook use).

        Args:
            images: Input images [B, C, H, W].

        Returns:
            cls_dict:      {layer_idx: [B, d_model]}
            patch_dict:    {layer_idx: [B, H, W, d_model]}
            target_logits: Final model output [B, num_classes].
        """
        H, W = self.patch_grid_size

        if self._full_sequence:
            # Fast path: persistent hooks already capture full sequences.
            self._full_states.clear()
            logits = self._run_model(images)
            cls_dict = {k: v[:, 0, :] for k, v in self._full_states.items()}
            patch_dict = {
                k: v[:, 1:, :].reshape(v.shape[0], H, W, v.shape[-1])
                for k, v in self._full_states.items()
            }
            self._full_states.clear()
            return cls_dict, patch_dict, logits

        # Slow path: install temp hooks, run, restore original hooks.
        self._remove_hooks()
        full_states: dict[int, torch.Tensor] = {}

        def make_full_hook(layer_idx: int) -> Any:
            def hook_fn(module: nn.Module, input: Any, output: torch.Tensor) -> None:
                full_states[layer_idx] = output.detach()  # [B, 1+H*W, d]
            return hook_fn

        temp_hooks: list[RemovableHandle] = []
        for layer_idx in self.target_layers:
            h = self.model.blocks[layer_idx].register_forward_hook(make_full_hook(layer_idx))
            temp_hooks.append(h)

        logits = self._run_model(images)

        for h in temp_hooks:
            h.remove()
        self._register_hooks()

        cls_dict = {k: v[:, 0, :] for k, v in full_states.items()}
        patch_dict = {
            k: v[:, 1:, :].reshape(v.shape[0], H, W, v.shape[-1])
            for k, v in full_states.items()
        }
        return cls_dict, patch_dict, logits

    def get_head_parameters(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (weight, bias) of the pretrained classification head."""
        if self._custom_head is not None:
            weight = self._custom_head.weight.data.clone()
            bias = self._custom_head.bias.data.clone() if self._custom_head.bias is not None else None
        else:
            weight = self.model.head.weight.data.clone()
            bias = self.model.head.bias.data.clone() if self.model.head.bias is not None else None
        return weight, bias

    def get_transform(self) -> T.Compose:
        """Return the validation/inference image transform for this model."""
        data_config = timm.data.resolve_model_data_config(self.model)
        return timm.data.create_transform(**data_config, is_training=False)

    def get_train_transform(self) -> T.Compose:
        """Return the training image transform with augmentation."""
        data_config = timm.data.resolve_model_data_config(self.model)
        return timm.data.create_transform(**data_config, is_training=True)

    def cleanup(self) -> None:
        """Remove hooks and free resources."""
        self._remove_hooks()
        self._hidden_states.clear()

    def __del__(self) -> None:
        self.cleanup()
