"""Evaluate gender balance of generated images using FairFace classifier.

Classifies sprinter outputs by gender, computes balance metrics, logs to WandB.

Usage:
    python scripts/evaluate_gender_balance.py \
        --image_dir outputs/sprinter/run_001/ \
        --run_name run_001 \
        --wandb_project conditional-matching
"""

import argparse
import json
import math
from pathlib import Path

import torch
import wandb

from fairface.gender_classifier import classify_gender, load_model


def compute_metrics(results, target_ratio):
    """Compute gender balance metrics from classification results.

    Returns dict with L (balance loss), counts, and confidence stats.
    """
    valid = [r for r in results if r["predicted_gender"] != "Error"]
    n = len(valid)
    if n == 0:
        return {"n_images": 0, "n_male": 0, "n_female": 0, "L": 0.0,
                "conf_male_mean": 0.0, "conf_female_mean": 0.0, "conf_overall_mean": 0.0}

    n_male = sum(1 for r in valid if r["predicted_gender"] == "Male")
    n_female = n - n_male

    # Target counts from ratio
    ratio_sum = target_ratio[0] + target_ratio[1]
    t_male = n * target_ratio[0] / ratio_sum
    t_female = n * target_ratio[1] / ratio_sum

    # Gender Balance Loss: L2 distance between observed and target counts
    L = math.sqrt((n_male - t_male) ** 2 + (n_female - t_female) ** 2)

    # Confidence stats
    male_confs = [r["confidence"] for r in valid if r["predicted_gender"] == "Male"]
    female_confs = [r["confidence"] for r in valid if r["predicted_gender"] == "Female"]
    all_confs = [r["confidence"] for r in valid]

    return {
        "n_images": n,
        "n_male": n_male,
        "n_female": n_female,
        "L": round(L, 4),
        "conf_male_mean": round(sum(male_confs) / len(male_confs), 4) if male_confs else 0.0,
        "conf_female_mean": round(sum(female_confs) / len(female_confs), 4) if female_confs else 0.0,
        "conf_overall_mean": round(sum(all_confs) / len(all_confs), 4) if all_confs else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate gender balance of generated images")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of generated images")
    parser.add_argument("--target_ratio", type=float, nargs=2, default=[50, 50],
                        help="Target gender ratio [male, female]")
    parser.add_argument("--run_name", type=str, required=True, help="Name for WandB run and output")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path for per-image results JSON (default: results/<run_name>_gender_eval.json)")
    parser.add_argument("--wandb_project", type=str, default="conditional-matching")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--weights_path", type=str, default=None, help="Path to FairFace weights")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Find images
    image_dir = Path(args.image_dir)
    image_paths = sorted(
        [str(p) for p in image_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    print(f"Found {len(image_paths)} images in {image_dir}", flush=True)

    if not image_paths:
        print("No images found. Exiting.")
        return

    # Load model and classify
    print(f"Loading FairFace model on {device}...", flush=True)
    model = load_model(args.weights_path, device)

    print("Classifying gender...", flush=True)
    results = classify_gender(image_paths, model=model, device=device)

    # Compute metrics
    metrics = compute_metrics(results, args.target_ratio)
    print(f"Results: {metrics['n_male']} male, {metrics['n_female']} female, L={metrics['L']}", flush=True)

    # Save JSON
    output_json = args.output_json or f"results/{args.run_name}_gender_eval.json"
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "run_name": args.run_name,
        **metrics,
        "per_image": results,
    }
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved per-image results to {output_json}", flush=True)

    # Log to WandB
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        job_type="gender_evaluation",
        config={
            "image_dir": str(image_dir),
            "n_images": metrics["n_images"],
            "target_ratio": args.target_ratio,
        },
    )
    wandb.log({
        "gender_L": metrics["L"],
        "n_male": metrics["n_male"],
        "n_female": metrics["n_female"],
        "conf_male_mean": metrics["conf_male_mean"],
        "conf_female_mean": metrics["conf_female_mean"],
        "conf_overall_mean": metrics["conf_overall_mean"],
        "conf_loss_overall": 1.0 - metrics["conf_overall_mean"],
        "conf_loss_male": 1.0 - metrics["conf_male_mean"],
        "conf_loss_female": 1.0 - metrics["conf_female_mean"],
    })
    wandb.finish()
    print("WandB logging complete.", flush=True)


if __name__ == "__main__":
    main()
