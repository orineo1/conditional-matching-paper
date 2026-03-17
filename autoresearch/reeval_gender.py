"""Re-run gender evaluation on existing runs with updated output:
  - male/ and female/ sorted image directories
  - gender_results.json with per-image confidence
  - gender_confidence_boxplot.png
"""
import os
import sys

import torch
from PIL import Image

# FairFace classifier — scripts/ is one level up from autoresearch/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scripts_dir = os.path.join(project_root, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from fairface.gender_classifier import evaluate_gender_balance

FAIRFACE_WEIGHTS = "/sci/labs/orzuk/shaulytolk/models/fairface/res34_fair_align_multi_7_20190809.pt"


def evaluate_dir(image_dir, device):
    """Load PNGs from directory, classify with MTCNN + FairFace, save sorted dirs + JSON + boxplot."""
    # Load only numbered PNGs (skip any previous male/female subdirs)
    files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png") and f[:3].isdigit()])
    if not files:
        print(f"  No images found in {image_dir}")
        return

    pil_images = [Image.open(os.path.join(image_dir, f)).convert("RGB") for f in files]
    print(f"  Loaded {len(pil_images)} images from {image_dir}")

    result = evaluate_gender_balance(
        pil_images, device, save_dir=image_dir, weights_path=FAIRFACE_WEIGHTS)

    print(f"  → {result['n_male']}M/{result['n_female']}F, L={result['gender_L']}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Find all run directories
    base_dir = os.environ.get("AR_BASE_DIR", "autoresearch_output")
    run_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

    if not run_dirs:
        print(f"No run directories found in {base_dir}")
        return

    for run_dir in run_dirs:
        run_path = os.path.join(base_dir, run_dir)
        print(f"\n=== {run_dir} ===")
        for subdir in ["eval_photos_guided", "eval_photos_unguided"]:
            photo_dir = os.path.join(run_path, subdir)
            if os.path.isdir(photo_dir):
                evaluate_dir(photo_dir, device)


if __name__ == "__main__":
    main()
