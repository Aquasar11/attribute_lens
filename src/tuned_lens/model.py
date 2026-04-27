"""Vision model wrapper with hook-based hidden state extraction."""

from __future__ import annotations

from typing import Any

import timm
import timm.data
import torch
import torch.nn as nn
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
            create_kwargs: dict[str, Any] = {}
            if self.config.head_weights_path:
                create_kwargs["num_classes"] = 1000  # adds nn.Linear(d_model, 1000) as model.head

            self.model = timm.create_model(
                self.config.model_name,
                pretrained=self.config.pretrained,
                **create_kwargs,
            )

            if self.config.head_weights_path:
                _sd = torch.load(
                    self.config.head_weights_path, map_location="cpu", weights_only=True
                )
                self.model.head.weight.data.copy_(_sd["weight"])
                self.model.head.bias.data.copy_(_sd["bias"])

        self.model.eval()
        if self.config.freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.to(self.device)

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

    def _remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def to(self, device: str | torch.device) -> VisionModelWrapper:
        self.device = str(device)
        self.model.to(device)
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
        logits = self.model(images)
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
        logits = self.model(images)
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
            logits = self.model(images)
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

        logits = self.model(images)

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
