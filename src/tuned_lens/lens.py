"""Lens architectures: learnable probes that map hidden states to label logits."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from .config import LensConfig


class BaseLens(nn.Module, ABC):
    """Abstract base class for all lens types.

    A lens maps a hidden state vector [B, d_model] to logits [B, num_classes].
    """

    @abstractmethod
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        ...

    def save(self, path: str, metadata: dict[str, Any] | None = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {"state_dict": self.state_dict()}
        if metadata:
            payload["metadata"] = metadata
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> BaseLens:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(**kwargs)
        instance.load_state_dict(payload["state_dict"])
        return instance


class AffineLens(BaseLens):
    """Single affine transformation: d_model -> num_classes."""

    def __init__(self, d_model: int, num_classes: int, bias: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.linear = nn.Linear(d_model, num_classes, bias=bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_state)


class MLPLens(BaseLens):
    """Multi-layer MLP lens with configurable depth.

    Architecture: Linear -> GELU -> Dropout -> [Linear -> GELU -> Dropout] * (n-2) -> Linear
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        layers: list[nn.Module] = []
        in_dim = d_model
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_state)


class LensBank(nn.Module):
    """Container holding one lens per target layer. This is what gets trained."""

    def __init__(self, lenses: dict[int, BaseLens]) -> None:
        super().__init__()
        self.lenses = nn.ModuleDict({str(k): v for k, v in lenses.items()})

    @property
    def layer_indices(self) -> list[int]:
        return sorted(int(k) for k in self.lenses.keys())

    def forward(self, layer_idx: int, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.lenses[str(layer_idx)](hidden_state)

    @staticmethod
    def create(
        config: LensConfig,
        target_layers: list[int],
        d_model: int,
        num_classes: int,
        head_weight: torch.Tensor | None = None,
        head_bias: torch.Tensor | None = None,
    ) -> LensBank:
        """Factory method to create a LensBank.

        Args:
            config: Lens configuration.
            target_layers: List of layer indices to create lenses for.
            d_model: Hidden state dimension of the model.
            num_classes: Number of output classes.
            head_weight: Optional pretrained head weight for initialization.
            head_bias: Optional pretrained head bias for initialization.
        """
        lenses: dict[int, BaseLens] = {}

        for layer_idx in target_layers:
            if config.lens_type == "mlp":
                hidden_dim = config.mlp_hidden_dim or d_model
                lens = MLPLens(
                    d_model=d_model,
                    num_classes=num_classes,
                    hidden_dim=hidden_dim,
                    num_layers=config.mlp_num_layers,
                    dropout=config.dropout,
                )
            else:  # affine
                lens = AffineLens(
                    d_model=d_model,
                    num_classes=num_classes,
                    bias=config.bias,
                )

            # Initialize from pretrained head weights if requested
            if config.init_from_head and head_weight is not None:
                _init_from_head(lens, head_weight, head_bias)

            # Initialize from a saved lens file if requested
            if config.init_from_pretrained is not None:
                _init_from_pretrained(lens, config.init_from_pretrained, layer_idx)

            lenses[layer_idx] = lens

        return LensBank(lenses)

    def save_all(self, output_dir: str, metadata: dict[str, Any] | None = None) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for layer_idx_str, lens in self.lenses.items():
            layer_meta = dict(metadata or {})
            layer_meta["layer_idx"] = int(layer_idx_str)
            lens.save(os.path.join(output_dir, f"layer_{layer_idx_str}.pt"), metadata=layer_meta)

    def load_all(self, output_dir: str) -> None:
        for layer_idx_str in self.lenses:
            path = os.path.join(output_dir, f"layer_{layer_idx_str}.pt")
            if os.path.exists(path):
                payload = torch.load(path, map_location="cpu", weights_only=False)
                self.lenses[layer_idx_str].load_state_dict(payload["state_dict"])


def _init_from_head(
    lens: BaseLens,
    head_weight: torch.Tensor,
    head_bias: torch.Tensor | None,
) -> None:
    """Copy pretrained classification head weights into a lens."""
    if isinstance(lens, AffineLens):
        lens.linear.weight.data.copy_(head_weight)
        if head_bias is not None and lens.linear.bias is not None:
            lens.linear.bias.data.copy_(head_bias)
    elif isinstance(lens, MLPLens):
        # For MLP, initialize only the last linear layer from the head
        last_linear = None
        for module in reversed(list(lens.net.modules())):
            if isinstance(module, nn.Linear):
                last_linear = module
                break
        if last_linear is not None and last_linear.out_features == head_weight.shape[0]:
            last_linear.weight.data.copy_(head_weight)
            if head_bias is not None and last_linear.bias is not None:
                last_linear.bias.data.copy_(head_bias)


def _init_from_pretrained(lens: BaseLens, pretrained_path: str, layer_idx: int) -> None:
    """Load lens weights from a saved checkpoint.

    Looks for either a single file or a per-layer file (layer_{idx}.pt).
    """
    if os.path.isdir(pretrained_path):
        path = os.path.join(pretrained_path, f"layer_{layer_idx}.pt")
    else:
        path = pretrained_path

    if os.path.exists(path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        lens.load_state_dict(payload["state_dict"])
