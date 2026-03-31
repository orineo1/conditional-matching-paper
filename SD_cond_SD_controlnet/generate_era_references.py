"""
Generate era-varied reference portraits using SDXL Turbo.

Samples years from a Gaussian centered on the Enlightenment era (~1700 AD),
clipped to [min_year, max_year]. Generates one portrait per sampled year
using a prompt that describes the person's historical context.

Usage:
    python generate_era_references.py --n_samples 100 --output_dir reference_images
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
    p.add_argument("--gender", type=str, default=None, choices=["man", "woman"],
                   help="If set, generate portraits of this gender; subdir becomes era_man/era_woman")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--output_dir", type=str, default="reference_images")
    p.add_argument("--year_mean", type=float, default=1700.0,
                   help="Center of Gaussian (default: Enlightenment ~1700 AD)")
    p.add_argument("--year_std", type=float, default=4000.0,
                   help="Std of Gaussian in years")
    p.add_argument("--year_min", type=int, default=-10000,
                   help="Earliest year (negative = BC)")
    p.add_argument("--year_max", type=int, default=2200,
                   help="Latest year")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_id", type=str, default="stabilityai/sdxl-turbo")
    p.add_argument("--n_steps", type=int, default=4)
    p.add_argument("--guidance_scale", type=float, default=0.0)
    return p.parse_args()


def year_to_prompt(year, gender=None):
    """Convert a year integer to a natural language prompt."""
    if year <= -5000:
        era = f"prehistoric times, around {abs(year):,} BC"
    elif year < 0:
        era = f"ancient times, around {abs(year):,} BC"
    elif year < 500:
        era = f"the classical era, around {year} AD"
    elif year < 1000:
        era = f"the early medieval era, around {year} AD"
    elif year < 1400:
        era = f"the medieval era, around {year} AD"
    elif year < 1600:
        era = f"the Renaissance, around {year} AD"
    elif year < 1800:
        era = f"the Enlightenment era, around {year} AD"
    elif year < 1900:
        era = f"the industrial era, around {year} AD"
    elif year < 2000:
        era = f"the modern era, around {year} AD"
    else:
        era = f"the future, around the year {year}"
    subject = f"a {gender}" if gender else "a person"
    return f"a superrealistic portrait of {subject} from {era}"


def year_to_filename_str(year):
    """Format year for filename: -3000 → 'n03000', 1700 → 'p01700'."""
    sign = "n" if year < 0 else "p"
    return f"{sign}{abs(year):05d}"


def sample_years(n, mean, std, year_min, year_max, rng):
    years = []
    while len(years) < n:
        batch = rng.normal(mean, std, size=n * 4)
        batch = batch[(batch >= year_min) & (batch <= year_max)]
        years.extend(batch.tolist())
    return [int(round(y)) for y in years[:n]]


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    rng = np.random.default_rng(args.seed)
    years = sample_years(args.n_samples, args.year_mean, args.year_std,
                         args.year_min, args.year_max, rng)

    subdir = f"era_{args.gender}" if args.gender else "era"
    save_dir = os.path.join(args.output_dir, subdir)
    os.makedirs(save_dir, exist_ok=True)

    # Save years for reproducibility
    years_path = os.path.join(save_dir, "years.json")
    with open(years_path, "w") as f:
        json.dump({
            "years": years, "seed": args.seed,
            "year_mean": args.year_mean, "year_std": args.year_std,
            "year_min": args.year_min, "year_max": args.year_max,
        }, f, indent=2)

    print(f"Year stats: min={min(years):,}, max={max(years):,}, "
          f"mean={np.mean(years):.0f}, std={np.std(years):.0f}", flush=True)

    print(f"Loading {args.model_id}...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    print("Model loaded.", flush=True)

    neg_prompt = "cartoon, blurry, low quality, deformed, modern photo filter"

    for i, year in enumerate(years):
        prompt = year_to_prompt(year, gender=args.gender)
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

        year_str = year_to_filename_str(year)
        filename = f"year_{year_str}_idx_{i:04d}.png"
        image.save(os.path.join(save_dir, filename))

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{args.n_samples}] year={year:,}  prompt='{prompt}'", flush=True)

    print(f"\nDone. {args.n_samples} images saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
