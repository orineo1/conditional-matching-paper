"""Evaluate gender balance of generated images using FairFace classifier.

Thin CLI wrapper around fairface.gender_classifier.evaluate_gender_balance().
Finds photo_*.png in --image_dir, classifies with MTCNN + FairFace,
saves sorted dirs + boxplot + JSON, optionally logs to WandB.

Usage:
    python scripts/evaluate_gender_balance.py \
        --image_dir outputs/my_run/ \
        --run_name my_eval \
        --wandb_project gender-classifier \
        --wandb_entity conditional-matching
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from fairface.gender_classifier import evaluate_gender_balance


def main():
    parser = argparse.ArgumentParser(description="Evaluate gender balance of generated images")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with photo_*.png files")
    parser.add_argument("--target_ratio", type=float, nargs=2, default=[50, 50],
                        help="Target gender ratio [male, female]")
    parser.add_argument("--run_name", type=str, required=True, help="Name for WandB run and output")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--weights_path", type=str, default=None, help="Path to FairFace weights")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Find individual sprinter photos
    image_dir = Path(args.image_dir)
    photo_paths = sorted(image_dir.rglob("photo_*.png"))
    if not photo_paths:
        print("No photo_*.png found. Exiting.")
        return

    pil_images = [Image.open(p).convert("RGB") for p in photo_paths]
    print(f"Found {len(pil_images)} sprinter photos in {image_dir}", flush=True)

    # Run unified evaluation (MTCNN crop + classify + sort + boxplot + JSON)
    result = evaluate_gender_balance(
        pil_images, device,
        save_dir=str(image_dir),
        target_ratio=tuple(args.target_ratio),
        weights_path=args.weights_path,
    )

    print(f"Results: {result['n_male']}M / {result['n_female']}F, L={result['gender_L']}", flush=True)

    # Optional WandB logging
    if args.wandb_project:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            job_type="gender_evaluation",
            config={"image_dir": str(image_dir), "n_images": result["n_male"] + result["n_female"]},
        )
        wandb.log({
            "gender_L": result["gender_L"],
            "n_male": result["n_male"],
            "n_female": result["n_female"],
            "gender_conf_mean": result["gender_conf_mean"],
            "conf_loss": 1.0 - result["gender_conf_mean"],
        })
        wandb.finish()
        print("WandB logging complete.", flush=True)


if __name__ == "__main__":
    main()
