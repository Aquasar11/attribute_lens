"""
unified_lens.lens
=================
Shared building blocks for the unified-lens trainers (v1 proxy, v2 faithfulness).

A unified lens is a single ``1024x1024`` affine ``Wx + b`` per ViT block that maps a
*patch* token into classifier-ready space, so the per-patch target-class score

    a_i(c) = log_softmax( head( fc_norm( norm( W h_l[i] + b ) ) ) )[c]

is the attribution value of patch i -- exactly what the faithfulness evaluation consumes.
The ViT backbone is always frozen; only the per-layer affine is trained.
"""

import os

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import timm


# CLIP normalization -- the eval-correct stats for vit_large_patch14_clip_224.
# Both trainers use these so there is no train/eval normalization shift.
CLIP_MEAN = [0.4815, 0.4578, 0.4082]
CLIP_STD = [0.2686, 0.2613, 0.2758]

# Insertion-surrogate blur radius (matches the eval's GaussianBlur(radius=10)).
UBLUR_RADIUS = 10


class AffineLens(nn.Module):
    """Single ``dim x dim`` linear mapping a patch token to classifier-ready space."""

    def __init__(self, dim=1024, init_identity=True):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        if init_identity:
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(dim))
                self.proj.bias.zero_()

    def forward(self, x):
        return self.proj(x)


class MultiLayerHook:
    """Captures patch-only activations from a list of ViT block indices."""

    def __init__(self, model, layer_indices):
        self.cache = {}
        self.handles = []
        for li in layer_indices:
            self.handles.append(
                model.blocks[li].register_forward_hook(self._make_hook(li))
            )

    def _make_hook(self, li):
        def fn(_mod, _inp, out):
            # ViT block output is (B, 1+N, D); drop CLS.
            self.cache[li] = out[:, 1:, :]
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


def build_head_fn(vit):
    """Frozen classifier head matching the eval's ``make_logits_fn('unified')``:

        logits = head( fc_norm( norm( x ) ) )

    ``fc_norm`` is ``Identity`` for the CLIP ViT-L, so this also matches the
    ``head(norm(x))`` path; applying it explicitly keeps train and eval identical.
    """
    norm = vit.norm
    fc_norm = getattr(vit, "fc_norm", nn.Identity())
    head = vit.head
    for m in (norm, fc_norm, head):
        for p in m.parameters():
            p.requires_grad_(False)

    def apply(x):
        return head(fc_norm(norm(x)))
    return apply


def make_blurred_baseline(imgs, sigma=UBLUR_RADIUS):
    """GaussianBlur of a normalized image batch (insertion-surrogate baseline).

    Blur is a weighted average and Normalize is affine, so blurring the normalized
    tensor equals normalizing the blurred RGB image (what the eval does). Computed
    under ``no_grad`` -- a fixed baseline, not learnable.
    """
    k = int(6 * sigma) | 1  # odd kernel ~ +/-3 sigma
    with torch.no_grad():
        return TF.gaussian_blur(imgs, kernel_size=[k, k], sigma=[float(sigma), float(sigma)])


def load_vit(model_name, weights_path=None, device="cpu"):
    """Load a frozen timm ViT.

    If ``weights_path`` exists, weights are loaded from that local checkpoint;
    otherwise the model is created with ``pretrained=True`` (timm download).
    """
    if weights_path and os.path.exists(weights_path):
        vit = timm.create_model(model_name, pretrained=False)
        vit.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        vit = timm.create_model(model_name, pretrained=True)
    vit = vit.to(device)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    return vit
