"""Download and save the DINOv2-L/14 ImageNet-1K classifier head weights.

Fetches the pretrained linear head from ``facebook/dinov2-large-imagenet1k-1-layer``
on HuggingFace and saves ``{'weight': Tensor[1000, 1024], 'bias': Tensor[1000]}``
to a local .pt file that can be referenced via ``model.head_weights_path`` in any
training or evaluation config.

Usage::

    pip install transformers
    python -m tuned_lens.scripts.download_dinov2_head --output dinov2_large_imagenet1k_head.pt
"""

import argparse

import torch
from transformers import Dinov2ForImageClassification

parser = argparse.ArgumentParser(description="Download DINOv2-L/14 ImageNet-1K head weights")
parser.add_argument(
    "--output",
    default="dinov2_large_imagenet1k_head.pt",
    help="Output path for the saved .pt file (default: dinov2_large_imagenet1k_head.pt)",
)
args = parser.parse_args()

print("Downloading facebook/dinov2-large-imagenet1k-1-layer …")
hf = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-large-imagenet1k-1-layer")
torch.save(
    {
        "weight": hf.classifier.weight.data.clone(),
        "bias": hf.classifier.bias.data.clone(),
    },
    args.output,
)
print(f"Saved → {args.output}  (weight: {list(hf.classifier.weight.shape)}, bias: {list(hf.classifier.bias.shape)})")
