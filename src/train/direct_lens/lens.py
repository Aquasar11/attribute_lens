"""Lens modules for direct_lens.

Two types:
  EmbeddingLens  d_model → d_model   (apply frozen classification head after to get logits)
  DirectLens     d_model → num_classes (outputs logits directly, warm-started from head weights)

Both are a single linear layer (W, b) — the goal is minimal computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class EmbeddingLens(nn.Module):
    """Linear probe mapping patch embeddings back into the model's embedding space.

    Use apply_head(lens(x)) to get logits.
    Initialized to identity so training starts from the model's own representations.
    """

    def __init__(self, d_model: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        nn.init.eye_(self.proj.weight)
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, d_model] → [B, d_model]
        return self.proj(x)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "type": "embedding"}, path)

    @classmethod
    def load(cls, path: str | Path, d_model: int, bias: bool = True) -> EmbeddingLens:
        data = torch.load(path, map_location="cpu", weights_only=True)
        lens = cls(d_model, bias)
        lens.load_state_dict(data["state_dict"])
        return lens


class DirectLens(nn.Module):
    """Linear probe mapping patch embeddings directly to class logits.

    Outputs logits directly — no separate classification head needed.
    Warm-started from the model's own head weights when provided.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        bias: bool = True,
        head_weight: Tensor | None = None,
        head_bias: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, num_classes, bias=bias)
        if head_weight is not None:
            self.proj.weight.data.copy_(head_weight)
            if bias and head_bias is not None:
                self.proj.bias.data.copy_(head_bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, d_model] → [B, num_classes]
        return self.proj(x)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "type": "direct"}, path)

    @classmethod
    def load(cls, path: str | Path, d_model: int, num_classes: int, bias: bool = True) -> DirectLens:
        data = torch.load(path, map_location="cpu", weights_only=True)
        lens = cls(d_model, num_classes, bias)
        lens.load_state_dict(data["state_dict"])
        return lens
