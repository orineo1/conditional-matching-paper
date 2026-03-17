"""Re-run gender evaluation on existing experiment runs.

Scans a base directory for run subdirectories, evaluates gender balance
on eval_photos_guided/ and eval_photos_unguided/ using MTCNN + FairFace.

Usage:
    python scripts/reeval_gender.py --base_dir autoresearch_output
    python scripts/reeval_gender.py --base_dir autoresearch_output --run run2_zeta5
"""

import argparse
import os
import sys

import torch
from PIL import Image

# Ensure scripts/ is on path for fairface import
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from fairface.gender_classifier import evaluate_gender_balance

DEFAULT_WEIGHTS = "/sci/labs/orzuk/shaulytolk/models/fairface/res34_fair_align_multi_7_20190809.pt"


def evaluate_dir(image_dir, device, weights_path):
    """Load PNGs from directory, classify with MTCNN + FairFace."""
    files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png") and f[:3].isdigit()])
    if not files:
        print(f"  No images found in {image_dir}")
        return

    pil_images = [Image.open(os.path.join(image_dir, f)).convert("RGB") for f in files]
    print(f"  Loaded {len(pil_images)} images from {image_dir}")

    result = evaluate_gender_balance(
        pil_images, device, save_dir=image_dir, weights_path=weights_path)

    print(f"  → {result['n_male']}M/{result['n_female']}F, L={result['gender_L']}")


def main():
    parser = argparse.ArgumentParser(description="Re-run gender eval on experiment runs")
    parser.add_argument("--base_dir", type=str, default="autoresearch_output",
                        help="Directory containing run subdirectories")
    parser.add_argument("--run", type=str, default=None,
                        help="Specific run to evaluate (default: all runs)")
    parser.add_argument("--weights_path", type=str, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.run:
        run_dirs = [args.run]
    else:
        run_dirs = sorted([d for d in os.listdir(args.base_dir)
                          if os.path.isdir(os.path.join(args.base_dir, d))])

    if not run_dirs:
        print(f"No run directories found in {args.base_dir}")
        return

    for run_dir in run_dirs:
        run_path = os.path.join(args.base_dir, run_dir)
        print(f"\n=== {run_dir} ===")
        for subdir in ["eval_photos_guided", "eval_photos_unguided"]:
            photo_dir = os.path.join(run_path, subdir)
            if os.path.isdir(photo_dir):
                evaluate_dir(photo_dir, device, args.weights_path)


if __name__ == "__main__":
    main()
