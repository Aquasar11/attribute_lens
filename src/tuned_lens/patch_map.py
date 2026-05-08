"""Patch map: per-layer learnable linear map for patch embedding transformation.

Transforms patch token embeddings x -> y = Wx + b so that foreground patches
are pulled toward the CLS token and background patches are pushed away.

Two variants:
  - FullPatchMap: square W in R^{d x d}
  - LowRankPatchMap: W = A @ B with rank <= r (bottleneck factorization)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from .config import PatchMapConfig


class BasePatchMap(nn.Module, ABC):
    """Abstract base: maps patch embeddings [*, d_model] -> [*, d_model]."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def save(self, path: str, metadata: dict[str, Any] | None = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload: dict[str, Any] = {"state_dict": self.state_dict()}
        if metadata:
            payload["metadata"] = metadata
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> BasePatchMap:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(**kwargs)
        instance.load_state_dict(payload["state_dict"])
        return instance


class FullPatchMap(BasePatchMap):
    """Full square affine map: y = W x + b, W in R^{d x d}.

    Initialized to the identity so training starts from a neutral point.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(d_model, d_model, bias=True)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LowRankPatchMap(BasePatchMap):
    """Low-rank affine map: y = up(down(x)) + b, where W = up.weight @ down.weight.

    The factorization W = A B with A in R^{d x r}, B in R^{r x d} limits the
    effective rank to r, acting as a bottleneck.  Initialized with orthogonal
    matrices so the initial map is near-identity for the top-r components.
    """

    def __init__(self, d_model: int, rank: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_model))
        nn.init.orthogonal_(self.down.weight)
        nn.init.orthogonal_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x)) + self.bias


class PatchMapBank(nn.Module):
    """Container holding one patch map per target layer.

    Analogous to LensBank — trained jointly, saved/loaded per layer.
    """

    def __init__(self, maps: dict[int, BasePatchMap]) -> None:
        super().__init__()
        self.maps = nn.ModuleDict({str(k): v for k, v in maps.items()})

    @property
    def layer_indices(self) -> list[int]:
        return sorted(int(k) for k in self.maps.keys())

    def forward(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Apply the map for *layer_idx* to *x*.

        *x* can be any shape ending in d_model, e.g. [B, d] or [B, N, d].
        """
        return self.maps[str(layer_idx)](x)

    @staticmethod
    def create(
        config: PatchMapConfig,
        target_layers: list[int],
        d_model: int,
    ) -> PatchMapBank:
        """Factory: one map per target layer, type controlled by config."""
        maps: dict[int, BasePatchMap] = {}
        for layer_idx in target_layers:
            if config.map_type == "low_rank":
                maps[layer_idx] = LowRankPatchMap(d_model, config.rank)
            else:  # full
                maps[layer_idx] = FullPatchMap(d_model)
        return PatchMapBank(maps)

    def save_all(self, output_dir: str, metadata: dict[str, Any] | None = None) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for layer_idx_str, m in self.maps.items():
            layer_meta = dict(metadata or {})
            layer_meta["layer_idx"] = int(layer_idx_str)
            m.save(os.path.join(output_dir, f"layer_{layer_idx_str}.pt"), metadata=layer_meta)

    def load_all(self, output_dir: str) -> None:
        for layer_idx_str in self.maps:
            path = os.path.join(output_dir, f"layer_{layer_idx_str}.pt")
            if os.path.exists(path):
                payload = torch.load(path, map_location="cpu", weights_only=False)
                self.maps[layer_idx_str].load_state_dict(payload["state_dict"])
