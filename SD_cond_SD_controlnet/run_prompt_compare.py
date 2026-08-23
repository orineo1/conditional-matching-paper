"""Generate images from a scribble with two different prompts and compare."""

import argparse
import os
import torch
from PIL import Image

from models import load_models
from generation import generate_and_store_cs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scribble_path", type=str,
                   default="scripts/assets/zeta5_final_guided.png")
    p.add_argument("--prompt_a", type=str,
                   default="a superrealistic professional photograph of")
    p.add_argument("--prompt_b", type=str,
                   default="in a world without gender norms, a superrealistic professional photograph of")
    p.add_argument("--n_images", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--controlnet_scale", type=float, default=0.5)
    p.add_argument("--output_dir", type=str, default="output/prompt_compare")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load scribble
    scribble_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.scribble_path) \
        if not os.path.isabs(args.scribble_path) else args.scribble_path
    scribble = Image.open(scribble_path).convert("RGB")
    print(f"Loaded scribble: {scribble_path} ({scribble.size})", flush=True)

    # Load models (we only need sprinter)
    _, sprinter = load_models(device)

    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
      for label, prompt in [("prompt_a", args.prompt_a), ("prompt_b", args.prompt_b)]:
        print(f"\n=== Generating {args.n_images} images for {label} ===", flush=True)
        print(f"  Prompt: \"{prompt}\"", flush=True)

        torch.manual_seed(args.seed)  # same seed for both

        images, _ = generate_and_store_cs(
            sprinter, prompt, scribble, args.n_images,
            batch_size=args.batch_size, cn_scale=args.controlnet_scale,
        )

        save_dir = os.path.join(args.output_dir, label)
        os.makedirs(save_dir, exist_ok=True)
        for i, img in enumerate(images):
            img.save(os.path.join(save_dir, f"photo_{i:04d}.png"))
        print(f"  Saved {len(images)} images to {save_dir}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
