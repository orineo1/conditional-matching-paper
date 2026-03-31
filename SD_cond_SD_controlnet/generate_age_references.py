"""
Generate age-varied reference face images using SDXL Base 1.0.

Samples ages from a clipped Gaussian (mean=50, std=15, clipped to [20, 80]) and generates
one portrait per age. Saves images and the sampled ages JSON for reproducibility.

Usage:
    python generate_age_references.py --gender woman --n_samples 100 --output_dir reference_images
    python generate_age_references.py --gender man   --n_samples 100 --output_dir reference_images
"""

import argparse
import json
import os

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gender", type=str, required=True, choices=["man", "woman"])
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--output_dir", type=str, default="reference_images")
    p.add_argument("--age_mean", type=float, default=50.0)
    p.add_argument("--age_std", type=float, default=15.0)
    p.add_argument("--age_min", type=int, default=20)
    p.add_argument("--age_max", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_id", type=str, default="stabilityai/sdxl-turbo")
    p.add_argument("--n_steps", type=int, default=4)
    p.add_argument("--guidance_scale", type=float, default=0.0)
    return p.parse_args()


def sample_ages(n, mean, std, age_min, age_max, rng):
    """Sample n ages from a Gaussian clipped to [age_min, age_max]."""
    ages = []
    while len(ages) < n:
        batch = rng.normal(mean, std, size=n * 4)
        batch = batch[(batch >= age_min) & (batch <= age_max)]
        ages.extend(batch.tolist())
    ages = ages[:n]
    return [int(round(a)) for a in ages]


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    rng = np.random.default_rng(args.seed)
    ages = sample_ages(args.n_samples, args.age_mean, args.age_std,
                       args.age_min, args.age_max, rng)

    save_dir = os.path.join(args.output_dir, f"age_{args.gender}")
    os.makedirs(save_dir, exist_ok=True)

    # Save ages for reproducibility
    ages_path = os.path.join(save_dir, "ages.json")
    with open(ages_path, "w") as f:
        json.dump({"ages": ages, "seed": args.seed, "gender": args.gender,
                   "age_mean": args.age_mean, "age_std": args.age_std,
                   "age_min": args.age_min, "age_max": args.age_max}, f, indent=2)
    print(f"Ages saved to {ages_path}", flush=True)
    print(f"Age stats: min={min(ages)}, max={max(ages)}, "
          f"mean={np.mean(ages):.1f}, std={np.std(ages):.1f}", flush=True)

    print(f"Loading {args.model_id}...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    print("Model loaded.", flush=True)

    for i, age in enumerate(ages):
        prompt = f"a superrealistic professional photograph of a {age}-year-old {args.gender}"
        neg_prompt = "cartoon, illustration, painting, drawing, blurry, low quality, deformed"

        # Per-image seed for reproducibility
        generator = torch.Generator(device=device).manual_seed(args.seed * 10000 + i)

        image = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=args.n_steps,
            guidance_scale=args.guidance_scale,
            height=512,
            width=512,
            generator=generator,
        ).images[0]

        filename = f"age_{age:03d}_idx_{i:04d}.png"
        save_path = os.path.join(save_dir, filename)
        image.save(save_path)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{args.n_samples}] age={age}  saved {filename}", flush=True)

    print(f"\nDone. {args.n_samples} images saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
