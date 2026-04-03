"""Patch scorers using trained CLS and patch lenses.

Two scorer types:
- CLSLensScorer: adjusts each patch token to the CLS token distribution,
  then scores via a lens trained on CLS tokens.
- PatchLensScorer: scores each patch center via a lens trained on k×k
  patch-token neighborhoods.
"""

from __future__ import annotations

import os
from glob import glob

import torch
import torch.nn.functional as F

# Import lens classes from the sibling tuned_lens package
from tuned_lens.lens import AffineLens, MLPLens, BaseLens


# ---------------------------------------------------------------------------
# Checkpoint auto-detection
# ---------------------------------------------------------------------------

def load_lens_checkpoint(path: str, device: str = "cpu") -> BaseLens:
    """Load a lens from a .pt checkpoint, auto-detecting AffineLens vs MLPLens.

    Inspects state_dict keys:
    - ``linear.weight`` → AffineLens(d_model, num_classes)
    - ``net.0.weight``  → MLPLens(d_model, num_classes, hidden_dim, num_layers)
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sd = payload["state_dict"]

    if "linear.weight" in sd:
        # AffineLens: linear.weight shape [num_classes, d_model]
        w = sd["linear.weight"]
        num_classes, d_model = w.shape
        bias = "linear.bias" in sd and sd["linear.bias"] is not None
        lens: BaseLens = AffineLens(d_model=d_model, num_classes=num_classes, bias=bias)
    else:
        # MLPLens: infer architecture from sequential layer keys
        # net.0.weight → first linear [hidden_dim, d_model]
        # last net.N.weight → last linear [num_classes, hidden_dim or d_model]
        linear_keys = sorted(
            k for k in sd if k.endswith(".weight") and k.startswith("net.")
        )
        first_w = sd[linear_keys[0]]   # [hidden_dim, d_model]
        last_w = sd[linear_keys[-1]]   # [num_classes, hidden_dim]
        d_model = first_w.shape[1]
        hidden_dim = first_w.shape[0]
        num_classes = last_w.shape[0]
        # Count linear layers to determine num_layers
        num_linear = len(linear_keys)
        num_layers = num_linear  # one linear per "layer" in MLP sense
        dropout = 0.0  # not stored; reconstruct with 0 dropout
        lens = MLPLens(
            d_model=d_model,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    lens.load_state_dict(sd)
    lens.to(device)
    lens.eval()
    return lens


def discover_lens_files(lens_dir: str) -> dict[int, str]:
    """Find all layer_*.pt files in a directory.

    Returns {layer_idx: path} sorted by layer index.
    """
    pattern = os.path.join(lens_dir, "layer_*.pt")
    paths = glob(pattern)
    result: dict[int, str] = {}
    for p in paths:
        name = os.path.basename(p)          # "layer_3.pt"
        idx_str = name[len("layer_"):-len(".pt")]
        try:
            result[int(idx_str)] = p
        except ValueError:
            pass
    return dict(sorted(result.items()))


# ---------------------------------------------------------------------------
# CLS Lens Scorer
# ---------------------------------------------------------------------------

class CLSLensScorer:
    """Scores patch tokens by shifting them toward the CLS token distribution.

    For each patch position p at layer L::

        adjusted[p] = h[p] - mean_token[L][p] + mean_cls[L]
        score[p]    = softmax(cls_lens_L(adjusted[p]))[y_hat]

    Args:
        cls_lens_dir: Directory with ``layer_{idx}.pt`` CLS lens checkpoints.
        means_path: Path to a ``.pt`` file with keys:
            - ``cls_means``: ``dict[int, Tensor[d_model]]`` — mean CLS embedding per layer.
            - ``token_means``: ``dict[int, Tensor[H*W, d_model]]`` — mean patch token per position per layer.
        target_layers: Layer indices to score. Must be a subset of available lens files.
        device: Torch device string.
    """

    def __init__(
        self,
        cls_lens_dir: str,
        means_path: str,
        target_layers: list[int],
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.target_layers = target_layers

        # Load lenses
        available = discover_lens_files(cls_lens_dir)
        self.lenses: dict[int, BaseLens] = {}
        for idx in target_layers:
            if idx not in available:
                raise FileNotFoundError(
                    f"CLS lens checkpoint not found for layer {idx} in {cls_lens_dir}"
                )
            self.lenses[idx] = load_lens_checkpoint(available[idx], device=device)

        # Load precomputed means
        means = torch.load(means_path, map_location="cpu", weights_only=False)
        self.cls_means: dict[int, torch.Tensor] = {
            int(k): v.to(device) for k, v in means["cls_means"].items()
        }
        self.token_means: dict[int, torch.Tensor] = {
            int(k): v.to(device) for k, v in means["token_means"].items()
        }

        # Validate means are available for all requested layers
        for idx in target_layers:
            if idx not in self.cls_means:
                raise KeyError(f"cls_means missing for layer {idx} in {means_path}")
            if idx not in self.token_means:
                raise KeyError(f"token_means missing for layer {idx} in {means_path}")

    @torch.no_grad()
    def score_all_layers(
        self,
        patch_states: dict[int, torch.Tensor],
        y_hat: int,
    ) -> dict[int, torch.Tensor]:
        """Compute patch score maps for all target layers.

        Args:
            patch_states: ``{layer_idx: Tensor[1, H, W, d_model]}`` from
                ``VisionModelWrapper.extract_patches()``.
            y_hat: Predicted class index (int) whose probability is the score.

        Returns:
            ``{layer_idx: Tensor[H, W]}`` float scores in [0, 1].
        """
        results: dict[int, torch.Tensor] = {}
        for idx in self.target_layers:
            patches = patch_states[idx].to(self.device)  # [1, H, W, d_model]
            _, H, W, d = patches.shape
            flat = patches[0].reshape(H * W, d)           # [H*W, d_model]

            mean_tok = self.token_means[idx]               # [H*W, d_model]
            mean_cls = self.cls_means[idx]                 # [d_model]

            adjusted = flat - mean_tok + mean_cls.unsqueeze(0)  # [H*W, d_model]

            logits = self.lenses[idx](adjusted)            # [H*W, num_classes]
            scores = F.softmax(logits, dim=-1)[:, y_hat]  # [H*W]
            results[idx] = scores.reshape(H, W).cpu()
        return results


# ---------------------------------------------------------------------------
# Patch Lens Scorer
# ---------------------------------------------------------------------------

class PatchLensScorer:
    """Scores patch tokens using a lens trained on k×k patch neighborhoods.

    For each valid center patch (i, j) at layer L::

        neighborhood = concat(k×k patch tokens around (i, j))  # [k*k*d_model]
        score[i, j] = softmax(patch_lens_L(neighborhood))[y_hat]

    Border patches (within ``patch_border`` steps of the image edge) receive
    ``torch.nan`` and are placed last in any ranking.

    Args:
        patch_lens_dir: Directory with ``layer_{idx}.pt`` patch lens checkpoints.
        patch_neighbor_size: Neighborhood side length ``k`` (must be odd, ≥ 1).
        patch_border: Number of border rows/columns to exclude.
        target_layers: Layer indices to score.
        device: Torch device string.
    """

    def __init__(
        self,
        patch_lens_dir: str,
        patch_neighbor_size: int,
        patch_border: int,
        target_layers: list[int],
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.target_layers = target_layers
        self.k = patch_neighbor_size
        self.half_k = patch_neighbor_size // 2
        self.border = patch_border

        # Load lenses
        available = discover_lens_files(patch_lens_dir)
        self.lenses: dict[int, BaseLens] = {}
        for idx in target_layers:
            if idx not in available:
                raise FileNotFoundError(
                    f"Patch lens checkpoint not found for layer {idx} in {patch_lens_dir}"
                )
            self.lenses[idx] = load_lens_checkpoint(available[idx], device=device)

    @torch.no_grad()
    def score_all_layers(
        self,
        patch_states: dict[int, torch.Tensor],
        y_hat: int,
    ) -> dict[int, torch.Tensor]:
        """Compute patch score maps for all target layers.

        Args:
            patch_states: ``{layer_idx: Tensor[1, H, W, d_model]}`` from
                ``VisionModelWrapper.extract_patches()``.
            y_hat: Predicted class index whose probability is the score.

        Returns:
            ``{layer_idx: Tensor[H, W]}`` float scores; border positions are NaN.
        """
        results: dict[int, torch.Tensor] = {}
        k = self.k
        half_k = self.half_k
        border = self.border

        for idx in self.target_layers:
            patches = patch_states[idx].to(self.device)  # [1, H, W, d_model]
            _, H, W, d = patches.shape

            # Zero-pad so every valid center has a full k×k neighborhood
            # [1, d, H, W] → pad → [1, d, H+2*half_k, W+2*half_k] → [1, H', W', d]
            padded = F.pad(
                patches.permute(0, 3, 1, 2),
                (half_k, half_k, half_k, half_k),
            ).permute(0, 2, 3, 1)  # [1, H+2*half_k, W+2*half_k, d]

            # Collect all valid neighborhoods in one pass
            valid_positions: list[tuple[int, int]] = []
            neighborhoods: list[torch.Tensor] = []

            for i in range(border, H - border):
                for j in range(border, W - border):
                    pi = i + half_k  # position in padded grid
                    pj = j + half_k
                    nb = padded[
                        0,
                        pi - half_k: pi + half_k + 1,
                        pj - half_k: pj + half_k + 1,
                        :,
                    ]  # [k, k, d]
                    neighborhoods.append(nb.reshape(k * k * d))
                    valid_positions.append((i, j))

            # Initialize output with NaN (border = NaN)
            score_map = torch.full((H, W), float("nan"), device=self.device)

            if neighborhoods:
                nb_tensor = torch.stack(neighborhoods, dim=0)  # [num_valid, k*k*d]
                logits = self.lenses[idx](nb_tensor)           # [num_valid, num_classes]
                scores = F.softmax(logits, dim=-1)[:, y_hat]  # [num_valid]

                for (i, j), s in zip(valid_positions, scores.tolist()):
                    score_map[i, j] = s

            results[idx] = score_map.cpu()
        return results
