"""
run_age.py — MLGD-F experiment for age distribution interpolation.

Target distribution spans a range of ages via --age_min, --age_max, --age_step.
Each age generates --n_per_age portrait images; total is auto-scaled to ~100 if
n_per_age is 0.
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


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MLGD-F age interpolation experiment"
    )
    p.add_argument("--output_dir", type=str, default="output/run_age")
    p.add_argument("--wandb_project", type=str, default="mlgdf-age")
    p.add_argument("--wandb_entity", type=str, default=None,
                   help="wandb team/entity (leave unset to use personal account)")

    # Age distribution
    p.add_argument("--age_min", type=int, default=10)
    p.add_argument("--age_max", type=int, default=80,
                   help="Exclusive upper bound")
    p.add_argument("--age_step", type=int, default=1)
    p.add_argument("--n_per_age", type=int, default=0,
                   help="Images per age. 0 = auto-scale to ~100 total")
    p.add_argument("--age_gender", type=str, default="man",
                   help="Gender word used in age prompts (man / woman / person)")

    # Scheduler / loop
    p.add_argument("--n_steps", type=int, default=30)
    p.add_argument("--start_step", type=int, default=15)
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

    return p.parse_args()


def age_prompt(age, gender):
    return (
        f"a superrealistic portrait photograph of a {age}-year-old {gender}, "
        "studio lighting, sharp focus, photographic"
    )


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

    AGES = list(range(args.age_min, args.age_max, args.age_step))
    N_PER_AGE = (
        args.n_per_age if args.n_per_age > 0
        else max(1, round(100 / len(AGES)))
    )
    N_TOTAL = len(AGES) * N_PER_AGE

    print(f"Age range: {args.age_min}–{args.age_max-1}  "
          f"step={args.age_step}  "
          f"{len(AGES)} ages × {N_PER_AGE} = {N_TOTAL} target images", flush=True)

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

    # Generate age target images (without scribble conditioning first,
    # then re-generate once scribble is extracted)
    print("Generating initial age targets for scribble extraction...", flush=True)
    age_images = {}
    with torch.no_grad():
        for age in AGES:
            prompt = age_prompt(age, args.age_gender)
            imgs, _ = generate_and_store_cs(
                sprinter, prompt,
                None,  # no conditioning yet
                N_PER_AGE, batch_size=2, cn_scale=args.controlnet_scale,
            )
            age_images[age] = imgs

    # Extract HED scribble from a young-adult image
    ref_age = AGES[min(2, len(AGES) - 1)]
    source_image = age_images[ref_age][0]
    print(f"Extracting HED scribble from age-{ref_age} image...", flush=True)
    scribble_pil = extract_scribble_hed(source_image)
    source_image.save(os.path.join(args.output_dir, "source_portrait.png"))
    scribble_pil.save(os.path.join(args.output_dir, "scribble.png"))

    # Re-generate targets conditioned on the scribble
    print("Re-generating targets with scribble conditioning...", flush=True)
    with torch.no_grad():
        for age in AGES:
            prompt = age_prompt(age, args.age_gender)
            imgs, _ = generate_and_store_cs(
                sprinter, prompt, scribble_pil,
                N_PER_AGE, batch_size=2, cn_scale=args.controlnet_scale,
            )
            age_images[age] = imgs
            print(f"  Age {age:3d}: {len(imgs)} images", flush=True)

    all_imgs_flat = [img for age in AGES for img in age_images[age]]

    # Save sample rows for inspection
    plot_row(all_imgs_flat[:10], f"Age samples ({AGES[0]}–{AGES[9]})",
             save_path=os.path.join(args.output_dir, "target_samples_young.png"))
    plot_row(all_imgs_flat[-10:], f"Age samples ({AGES[-10]}–{AGES[-1]})",
             save_path=os.path.join(args.output_dir, "target_samples_old.png"))

    # Encode to CLIP
    print("Encoding age targets to CLIP...", flush=True)
    clip_model.to(device)
    age_clip_embs = {}
    with torch.no_grad():
        for age in AGES:
            age_clip_embs[age] = encode_images_clip(
                pil_images_to_tensor(age_images[age], device),
                clip_model, clip_processor,
            )
    clip_model.to("cpu")

    all_clip_embeddings = torch.cat(
        [age_clip_embs[age] for age in AGES], dim=0
    )
    N_total = all_clip_embeddings.shape[0]
    print(f"Target CLIP embeddings: {all_clip_embeddings.shape}", flush=True)

    # PCA fitted on youngest vs oldest bracket
    n_anchor = min(10, len(AGES) // 4)
    young_np = torch.cat(
        [age_clip_embs[a] for a in AGES[:n_anchor]], dim=0
    ).cpu().numpy()
    old_np = torch.cat(
        [age_clip_embs[a] for a in AGES[-n_anchor:]], dim=0
    ).cpu().numpy()
    pca_fixed = PCA(n_components=2)
    pca_fixed.fit(np.vstack([young_np, old_np]))

    age_labels_arr = np.array([a for a in AGES for _ in range(N_PER_AGE)])
    target_clip_np = all_clip_embeddings.cpu().numpy()

    # Continuous color map for age
    group_colors = [
        plt.cm.plasma((a - AGES[0]) / (AGES[-1] - AGES[0]))
        for a in AGES
    ]
    group_names_list = [str(a) for a in AGES]
    group_markers = ["o"] * len(AGES)
    n_groups = len(AGES)

    # PCA plot
    def plot_target_pca(save_path):
        coords_all = pca_fixed.transform(target_clip_np)
        fig, ax = plt.subplots(figsize=(10, 7))
        sc = ax.scatter(
            coords_all[:, 0], coords_all[:, 1],
            c=age_labels_arr, cmap="plasma",
            s=60, alpha=0.8, edgecolors="white", linewidths=0.4,
        )
        plt.colorbar(sc, ax=ax, label="Age")
        for age in AGES:
            if age % 10 == 0:
                c = pca_fixed.transform(
                    age_clip_embs[age].cpu().numpy()
                ).mean(0)
                ax.annotate(
                    str(age), c, fontsize=8, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points",
                    color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.4),
                )
        ax.set_xlabel(f"PC1 ({pca_fixed.explained_variance_ratio_[0]:.1%}) — Age axis")
        ax.set_ylabel(f"PC2 ({pca_fixed.explained_variance_ratio_[1]:.1%})")
        ax.set_title("Target CLIP PCA — Age distribution")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    pca_path = os.path.join(args.output_dir, "target_clip_pca.png")
    plot_target_pca(pca_path)

    softmax_prompt_a = age_prompt(AGES[-1], args.age_gender)  # oldest
    softmax_prompt_b = age_prompt(AGES[0], args.age_gender)   # youngest

    # wandb init
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "experiment": "age",
            "age_min": args.age_min,
            "age_max": args.age_max,
            "age_step": args.age_step,
            "age_gender": args.age_gender,
            "n_per_age": N_PER_AGE,
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

    # Log sample images for a few decades
    age_image_log = {}
    for age in AGES[::10]:
        age_image_log[f"target_samples/age_{age}"] = [
            wandb.Image(p) for p in age_images[age]
        ]
    wandb.log({
        "scribble": wandb.Image(scribble_pil),
        "source_portrait": wandb.Image(source_image),
        "target_clip_pca": wandb.Image(pca_path),
        **age_image_log,
    })

    # Extra npy
    extra_npy = {
        "targets_all_ages": (all_imgs_flat, "targets_all_ages.npy"),
        "targets_young": (age_images[AGES[0]], "targets_young.npy"),
        "targets_old": (age_images[AGES[-1]], "targets_old.npy"),
        "scribble": ([scribble_pil], "scribble.npy"),
        "source_portrait": ([source_image], "source_portrait.npy"),
    }

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
        group_names_list=group_names_list,
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
