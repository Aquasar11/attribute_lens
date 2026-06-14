"""
eval.y_hat_creator
=================
Generate the per-image ``y_hat`` baseline CSV consumed by the faithfulness eval.

Each row is (image_id, class_id) where class_id is the frozen model's argmax prediction
on the CLIP-preprocessed image; the faithfulness eval ranks patches by the per-patch
log-prob of this class_id.

Run:
    python -m eval.y_hat_creator \
        --image-dir data/first_100/images \
        --output-csv data/first_100/y_hat_baseline.csv
"""

import argparse
import os

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from . import common

DEFAULT_WEIGHTS = "./model/pretrained_models/vit_large_patch14_clip_224.openai_ft_in1k.pt"


def create_yhat_csv(image_dir, output_csv, weights, model_name, device):
    clip_model = common.load_clip_model(model_name, weights, device)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpeg")]
    results = []
    print(f"Generating y_hat predictions for {len(image_files)} images...")
    for img_name in tqdm(image_files):
        pil_img = Image.open(os.path.join(image_dir, img_name)).convert("RGB")
        img_tensor = common.transform_image(pil_img, device)
        with torch.no_grad():
            y_hat = clip_model(img_tensor).argmax(dim=-1).item()
        results.append({"image_id": img_name, "class_id": y_hat})

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved baseline predictions to {output_csv}")


def main():
    p = argparse.ArgumentParser(description="Create the y_hat baseline CSV for faithfulness eval.")
    p.add_argument("--image-dir", required=True, help="dir of .JPEG images")
    p.add_argument("--output-csv", required=True, help="output CSV path")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS,
                   help="local timm checkpoint; downloads pretrained if missing")
    p.add_argument("--model-name", default=common.DEFAULT_MODEL_NAME)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    create_yhat_csv(args.image_dir, args.output_csv, args.weights, args.model_name, device)


if __name__ == "__main__":
    main()
