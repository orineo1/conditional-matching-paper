"""
run_gender.py — MLGD-F experiment for distributional gender/attribute interpolation.

Target distribution is defined via --groups, e.g.:
    --groups "Woman:a portrait of a woman, studio lighting:50"
             "Man:a portrait of a man, studio lighting:50"

Each entry is  <name>:<prompt>:<percentage>  where percentages must sum to 100.
The total number of target images is controlled by --n_targets (default 100).
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
import wandb
from sklearn.decomposition import PCA

# Make sure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.clip_utils import encode_images_clip, load_clip_model
from src.generation import generate_and_store_cs
from src.image_utils import extract_scribble_hed
from src.models import load_models
from src.visualization import plot_row
from dps_loop import (
    pil_images_to_tensor,
    run_mlgdf_loop,
    save_image_list_npy,
)

_COLORS = [
    "crimson", "dodgerblue", "limegreen", "orange", "mediumorchid",
    "gold", "deepskyblue", "hotpink", "slategray", "peru",
]
_MARKERS = ["o", "x", "^", "s", "D", "P", "v", "<", ">", "h"]


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MLGD-F gender/attribute interpolation experiment"
    )
    p.add_argument("--output_dir", type=str, default="output/run_gender")
    p.add_argument("--wandb_project", type=str, default="mlgdf-gender")
    p.add_argument("--wandb_entity", type=str, default=None,
                   help="wandb team/entity (leave unset to use personal account)")

    # Target distribution
    p.add_argument(
        "--groups", type=str, nargs="+", required=True,
        metavar="NAME:PROMPT:PCT",
        help=(
            "Target groups as 'name:prompt:percentage' triples. "
            "Percentages must sum to 100. "
            "Example: --groups 'Woman:a portrait of a woman:50' "
            "'Man:a portrait of a man:50'"
        ),
    )
    p.add_argument("--n_targets", type=int, default=100,
                   help="Total number of target images (split by percentage)")

    # Scheduler / loop
    p.add_argument("--n_steps", type=int, default=30)
    p.add_argument("--start_step", type=int, default=15,
                   help="SDEdit start step — MLGD-F runs from here to n_steps")
    p.add_argument("--seed", type=int, default=None)

    # Guidance
    p.add_argument("--base_zeta", type=float, default=5.0)
    p.add_argument("--guidance_scale", type=float, default=0.0)
    p.add_argument("--controlnet_scale", type=float, default=0.5)
    p.add_argument("--loss_fn", type=str, default="mmd", choices=["mmd", "swd"])
    p.add_argument("--bandwidth_scale", type=float, default=1.0)
    p.add_argument("--loss_scale", type=float, default=1.0)
    p.add_argument("--kernel_alpha", type=float, default=1.0)

    # Variations / eval
    p.add_argument("--num_variations", type=int, default=6)
    p.add_argument("--n_eval", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=0)

    # Prompts
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--sprinter_variation_prompt", type=str,
                   default="a superrealistic professional photograph of")
    p.add_argument("--sprinter_eval_prompt", type=str,
                   default="a superrealistic professional photograph of")

    # Models
    p.add_argument("--controlnet_model_id", type=str,
                   default="xinsir/controlnet-scribble-sdxl-1.0")
    p.add_argument("--sprinter_model_id", type=str,
                   default="stabilityai/sdxl-turbo")
    p.add_argument("--architect_model_id", type=str,
                   default="stabilityai/sdxl-turbo")
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--architect_unet_path", type=str, default=None)

    # Optional: provide a scribble directly instead of generating one
    p.add_argument(
        "--scribble_image", type=str, default=None,
        help=(
            "Path to an existing scribble image (.png/.jpg). "
            "When provided, skips unconditioned target generation and HED extraction. "
            "Target images are generated directly conditioned on this scribble."
        ),
    )

    return p.parse_args()


def parse_groups(group_specs, n_targets):
    """
    Parse --groups entries and compute per-group sample counts.

    Returns list of (name, prompt, n_samples, color, marker).
    Raises ValueError if percentages don't sum to 100.
    """
    groups = []
    for idx, spec in enumerate(group_specs):
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --groups entry '{spec}'. "
                "Expected format: 'name:prompt:percentage'"
            )
        name, prompt, pct_str = parts
        pct = float(pct_str)
        groups.append((name.strip(), prompt.strip(), pct,
                       _COLORS[idx % len(_COLORS)],
                       _MARKERS[idx % len(_MARKERS)]))

    total_pct = sum(g[2] for g in groups)
    if abs(total_pct - 100.0) > 0.5:
        raise ValueError(
            f"Group percentages must sum to 100, got {total_pct:.1f}"
        )

    # Convert percentages to sample counts (round, ensure sum == n_targets)
    counts = [round(g[2] / 100.0 * n_targets) for g in groups]
    diff = n_targets - sum(counts)
    if diff != 0:
        counts[-1] += diff  # adjust last group to fix rounding

    return [
        (name, prompt, counts[i], color, marker)
        for i, (name, prompt, _, color, marker) in enumerate(groups)
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    steps_dir = os.path.join(args.output_dir, "steps")
    os.makedirs(steps_dir, exist_ok=True)

    # Parse group specs
    groups = parse_groups(args.groups, args.n_targets)
    group_names = [g[0] for g in groups]
    group_colors = [g[3] for g in groups]
    group_markers = [g[4] for g in groups]
    n_groups = len(groups)

    print(f"Target groups:", flush=True)
    for name, prompt, n, _, _ in groups:
        print(f"  {name} ({n} images): {prompt}", flush=True)

    # Load models
    print("Loading models...", flush=True)
    architect, sprinter = load_models(
        device,
        architect_lora_path=args.lora_path,
        architect_unet_path=args.architect_unet_path,
        controlnet_model_id=args.controlnet_model_id,
        sprinter_model_id=args.sprinter_model_id,
        architect_model_id=args.architect_model_id,
    )
    clip_model, clip_processor = load_clip_model(device)
    print("Models loaded.", flush=True)

    # ── Scribble: load from file or extract from generated targets ────────────
    if args.scribble_image is not None:
        # Path provided — load directly, skip unconditioned generation
        from PIL import Image as _PIL
        if not os.path.exists(args.scribble_image):
            raise FileNotFoundError(f"--scribble_image not found: {args.scribble_image}")
        scribble_pil = _PIL.open(args.scribble_image).convert("RGB")
        source_image = scribble_pil  # use scribble itself as the "source" for logging
        print(f"Using provided scribble: {args.scribble_image}  "
              f"size={scribble_pil.size}", flush=True)

        # Generate targets conditioned on the provided scribble
        print("Generating target images conditioned on provided scribble...", flush=True)
        target_images_per_group = {}
        with torch.no_grad():
            for name, prompt, n, _, _ in groups:
                imgs, _ = generate_and_store_cs(
                    sprinter, prompt, scribble_pil,
                    n, batch_size=2, cn_scale=args.controlnet_scale,
                )
                target_images_per_group[name] = imgs
                plot_row(imgs, f"Target: {name}",
                         save_path=os.path.join(args.output_dir,
                                                f"target_samples_{name}.png"))
                print(f"  {name}: {len(imgs)} images", flush=True)

    else:
        # No scribble provided — generate unconditioned targets, extract HED scribble,
        # then re-generate targets conditioned on the scribble.
        print("Generating target images (unconditioned, for scribble extraction)...",
              flush=True)
        target_images_per_group = {}
        with torch.no_grad():
            for name, prompt, n, _, _ in groups:
                imgs, _ = generate_and_store_cs(
                    sprinter, prompt,
                    None,  # blank white placeholder — see generate_and_store_cs
                    n, batch_size=2, cn_scale=args.controlnet_scale,
                )
                target_images_per_group[name] = imgs
                plot_row(imgs, f"Target: {name}",
                         save_path=os.path.join(args.output_dir,
                                                f"target_samples_{name}_uncond.png"))
                print(f"  {name}: {len(imgs)} images", flush=True)

        # Extract HED scribble from first group (use index 2 if available, else 0)
        print("Extracting HED scribble...", flush=True)
        first_group_imgs = target_images_per_group[group_names[0]]
        source_image = first_group_imgs[min(2, len(first_group_imgs) - 1)]
        scribble_pil = extract_scribble_hed(source_image)

        # Re-generate targets conditioned on the actual scribble
        print("Re-generating targets with scribble conditioning...", flush=True)
        with torch.no_grad():
            for name, prompt, n, _, _ in groups:
                imgs, _ = generate_and_store_cs(
                    sprinter, prompt, scribble_pil,
                    n, batch_size=2, cn_scale=args.controlnet_scale,
                )
                target_images_per_group[name] = imgs
                plot_row(imgs, f"Target: {name}",
                         save_path=os.path.join(args.output_dir,
                                                f"target_samples_{name}.png"))
                print(f"  {name}: {len(imgs)} images", flush=True)

    source_image.save(os.path.join(args.output_dir, "source_portrait.png"))
    scribble_pil.save(os.path.join(args.output_dir, "scribble.png"))
    print(f"Scribble ready  size={scribble_pil.size}", flush=True)

    # Encode targets to CLIP
    print("Encoding targets to CLIP...", flush=True)
    clip_model.to(device)
    clip_embs_per_group = {}
    with torch.no_grad():
        for name, imgs in target_images_per_group.items():
            clip_embs_per_group[name] = encode_images_clip(
                pil_images_to_tensor(imgs, device), clip_model, clip_processor
            )
    clip_model.to("cpu")

    all_clip_embeddings = torch.cat(
        [clip_embs_per_group[name] for name, _, _, _, _ in groups], dim=0
    )
    N_total = all_clip_embeddings.shape[0]
    print(f"Target CLIP embeddings: {all_clip_embeddings.shape}", flush=True)

    # PCA fitted on first and last group (two extremes)
    anchor_a = clip_embs_per_group[group_names[0]].cpu().numpy()
    anchor_b = clip_embs_per_group[group_names[-1]].cpu().numpy()
    pca_fixed = PCA(n_components=2)
    pca_fixed.fit(np.vstack([anchor_a, anchor_b]))

    # PCA plot of target distribution
    def plot_target_pca(save_path):
        fig, ax = plt.subplots(figsize=(9, 7))
        for (name, _, _, color, marker) in groups:
            embs = clip_embs_per_group[name].cpu().numpy()
            coords = pca_fixed.transform(embs)
            ax.scatter(coords[:, 0], coords[:, 1],
                       c=color, label=name, alpha=0.6, marker=marker, s=50)
            cx, cy = coords.mean(0)
            ax.scatter(cx, cy, c=color, marker="*", s=200,
                       edgecolors="black", linewidths=0.6, zorder=5)
            ax.annotate(name, (cx, cy), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=color, fontweight="bold")
        ax.set_xlabel(f"PC1 ({pca_fixed.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca_fixed.explained_variance_ratio_[1]:.1%})")
        ax.set_title("Target CLIP PCA (fitted on extreme groups)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    pca_path = os.path.join(args.output_dir, "target_clip_pca.png")
    plot_target_pca(pca_path)

    target_clip_np = all_clip_embeddings.cpu().numpy()
    softmax_prompt_a = groups[0][1]   # first group prompt
    softmax_prompt_b = groups[-1][1]  # last group prompt

    # wandb init
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "experiment": "gender",
            "groups": {
                name: {"prompt": prompt, "n_samples": n, "pct": g[2]}
                for g, (name, prompt, n, _, _) in
                zip(groups, [(g[0], g[1], g[2], None, None) for g in groups])
            },
            "n_targets": N_total,
            "n_steps": args.n_steps,
            "start_step": args.start_step,
            "num_variations": args.num_variations,
            "base_zeta": args.base_zeta,
            "guidance_scale": args.guidance_scale,
            "controlnet_scale": args.controlnet_scale,
            "loss_fn": args.loss_fn,
            "loss_scale": args.loss_scale,
            "bandwidth_scale": args.bandwidth_scale,
            "kernel_alpha": args.kernel_alpha,
            "n_eval": args.n_eval,
            "sprinter_variation_prompt": args.sprinter_variation_prompt,
            "sprinter_eval_prompt": args.sprinter_eval_prompt,
            "architect_model": args.architect_model_id,
            "sprinter_model": args.sprinter_model_id,
            "edge_method": "hed_scribble",
            "seed": args.seed,
        },
    )
    print(f"wandb run: {run.name}", flush=True)

    wandb.log({
        "scribble": wandb.Image(scribble_pil),
        "source_portrait": wandb.Image(source_image),
        "target_clip_pca": wandb.Image(pca_path),
        **{
            f"target_samples/{name}": [wandb.Image(p) for p in imgs]
            for name, imgs in target_images_per_group.items()
        },
    })

    # Extra npy to save
    extra_npy = {}
    for name, imgs in target_images_per_group.items():
        safe = name.lower().replace(" ", "_")
        extra_npy[f"targets_{safe}"] = (imgs, f"targets_{safe}.npy")
    extra_npy["scribble"] = ([scribble_pil], "scribble.npy")
    extra_npy["source_portrait"] = ([source_image], "source_portrait.npy")

    # Run MLGD-F loop
    metrics = run_mlgdf_loop(
        architect=architect,
        sprinter=sprinter,
        clip_model=clip_model,
        clip_processor=clip_processor,
        scribble_pil=scribble_pil,
        all_clip_embeddings=all_clip_embeddings,
        pca_fixed=pca_fixed,
        target_clip_np=target_clip_np,
        group_names_list=group_names,
        group_colors=group_colors,
        group_markers=group_markers,
        n_groups=n_groups,
        args=args,
        output_dir=args.output_dir,
        steps_dir=steps_dir,
        device=device,
        softmax_prompt_a=softmax_prompt_a,
        softmax_prompt_b=softmax_prompt_b,
        extra_npy_saves=extra_npy,
    )

    print(f"\nDone. MMD delta = {metrics['mmd_delta']:.6f}", flush=True)


if __name__ == "__main__":
    main()
