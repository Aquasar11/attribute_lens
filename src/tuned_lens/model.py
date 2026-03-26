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
            self.model = timm.create_model(
                self.config.model_name, pretrained=self.config.pretrained
            )

        self.model.eval()
        if self.config.freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

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
            # output shape: [B, seq_len, d_model] — capture CLS at position 0
            self._hidden_states[layer_idx] = output[:, 0, :].detach()
        return hook_fn

    def _remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def to(self, device: str | torch.device) -> VisionModelWrapper:
        self.device = str(device)
        self.model.to(device)
        return self

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
