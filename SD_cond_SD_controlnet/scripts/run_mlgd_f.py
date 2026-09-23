"""
run_mlgd_f.py — MLGD-F pipeline entry point.

Cluster-ready script version of main.ipynb (scribble_cond_loss branch).

Pipeline:
    1. Load Architect + Sprinter + CLIP.
    2. Generate target distribution (man/woman portraits) and encode to CLIP.
    3. Extract HED scribble from one portrait (SDEdit-style init).
    4. Noise the scribble latent to start_step and run MLGD-F guidance.
       Pass --run_unguided to also run the regular (unguided) path in
       parallel for comparison at each step (an extra full UNet forward pass
       every step; off by default).
    5. Evaluate final MMD/SWD (both paths if --run_unguided, else MLGD-F
       only) and log to wandb -- only the source scribble, source portrait,
       final MLGD-F scribble, and the per-step visualization are uploaded as
       images; everything else image-shaped is saved locally only.

Usage:
    python scripts/run_mlgd_f.py --output_dir output/run_001 --seed 1
    python scripts/run_mlgd_f.py --output_dir output/run_002 --seed 1 --run_unguided
"""

import argparse
import copy
import gc
import json
import os
import sys
import time
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA

# Make src/ importable when called from the repo root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.join(os.path.dirname(_SCRIPT_DIR), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from analysis import compare_scribbles_heatmap
from clip_utils import encode_images_clip, load_clip_model
from generation import (
    compute_pred_x0_direct,
    denoise_step,
    generate_and_store_cs,
    predict_noise_cfg,
    run_dps_step_clip,
)
from image_utils import build_base_image, latent_to_pil, sobel_proxy
from metrics import compute_mmd, compute_swd, evaluate_distribution_mmd, mmd_report
from models import DEFAULT_CONTROLNET_MODEL_ID, load_models, setup_gradient_checkpointing
from visualization import plot_row, visualize_step

LOSS_FNS = {"mmd": compute_mmd, "swd": compute_swd}


# ---------------------------------------------------------------------------
# Particle filter (P-MLGD-F) helper
# ---------------------------------------------------------------------------

def systematic_resample(weights, generator=None):
    """
    Standard systematic (a.k.a. stratified-systematic) resampling: N=len(weights)
    equally-spaced strata with one shared random offset, giving lower-variance
    index draws than plain multinomial resampling while remaining unbiased.

    Args:
        weights:   [N] tensor of normalized (sum to 1) particle weights.
        generator: optional torch.Generator for reproducibility.

    Returns:
        [N] LongTensor of ancestor indices (with replacement; a high-weight
        index typically appears multiple times, a low-weight one may not appear).
    """
    n = weights.shape[0]
    u0 = torch.rand(1, generator=generator).item() / n
    positions = u0 + torch.arange(n, dtype=torch.float64) / n
    cumsum = torch.cumsum(weights.double(), dim=0)
    cumsum[-1] = 1.0  # guard against floating-point drift leaving position 1 unmatched
    idx = torch.searchsorted(cumsum, positions)
    return idx.clamp(max=n - 1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MLGD-F Pipeline")

    p.add_argument("--output_dir",          type=str, default="output/mlgd_f_run")
    p.add_argument("--lora_path",           type=str, default=None)
    p.add_argument("--architect_unet_path", type=str, default=None)
    p.add_argument("--wandb_project",       type=str, default="MLGDF-EXP")
    p.add_argument("--wandb_entity",        type=str, default="")

    # Scheduler / loop
    p.add_argument("--n_steps",    type=int, default=30)
    p.add_argument("--start_step", type=int, default=15,
                   help="SDEdit start step — MLGD-F runs from here to n_steps")

    # Guidance
    p.add_argument("--base_zeta",        type=float, default=1.0)
    p.add_argument("--guidance_scale",   type=float, default=0.0,
                   help="CFG scale for architect (0.0 = unconditional)")
    p.add_argument("--controlnet_scale", type=float, default=0.5)
    p.add_argument("--loss_fn",          type=str,   default="mmd",
                   choices=["mmd", "swd"])
    p.add_argument("--bandwidth_scale",  type=float, default=1.0,
                   help="Scale factor for MMD bandwidth (< 1 = sharper kernel)")
    p.add_argument("--loss_scale",       type=float, default=1.0,
                   help="Multiply loss before grad to amplify weak gradients")
    p.add_argument("--kernel_alpha",     type=float, default=1.0,
                   help="Generalised RBF exponent (>1 = sharper falloff)")

    # Adam on the DPS correction (alternative to the zeta_i-scaled gradient step)
    p.add_argument("--use_adam", action="store_true",
                   help="Use a persistent Adam optimizer on the correction instead "
                        "of -zeta_i * grad. Adam keeps running first/second-moment "
                        "estimates of the gradient across the whole guidance "
                        "trajectory (a smoothed, per-coordinate adaptive step size); "
                        "zeta_i already does a cruder scalar version of the same "
                        "job (rescale by 1/loss), so zeta_i is ignored entirely "
                        "when this is set -- combining both would double-normalize "
                        "the step and make the effective step size impossible to "
                        "reason about.")
    p.add_argument("--adam_lr", type=float, default=0.01,
                   help="Adam learning rate (only used with --use_adam; this is "
                        "the step-size knob in Adam mode, playing the role "
                        "--base_zeta plays in zeta_i mode)")
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_eps",   type=float, default=1e-8)

    # Particle filter (P-MLGD-F): replaces the "R independent restarts, keep the
    # best by final loss" protocol with a proper sequential Monte Carlo scheme --
    # num_particles trajectories run in lockstep, reweighted each step toward the
    # tilted target Q(x) ∝ P(x) exp(-beta * loss(x)), and resampled (systematic)
    # whenever the effective sample size drops below half the particle count.
    p.add_argument("--particle_filter", action="store_true",
                   help="Run P-MLGD-F: num_particles trajectories in lockstep with "
                        "SMC-style reweighting/resampling, instead of a single "
                        "trajectory. Not compatible with --run_unguided.")
    p.add_argument("--num_particles", type=int, default=10,
                   help="Number of particles (only used with --particle_filter)")
    p.add_argument("--beta_min", type=float, default=0.0,
                   help="Annealed tilting strength at the first guidance step "
                        "(only used with --particle_filter)")
    p.add_argument("--beta_max", type=float, default=50.0,
                   help="Annealed tilting strength at the last guidance step -- "
                        "beta_t is linearly interpolated between beta_min and "
                        "beta_max over the guidance trajectory (only used with "
                        "--particle_filter). Needs empirical tuning: too small "
                        "barely reweights particles, too large collapses onto one "
                        "particle immediately.")
    p.add_argument("--resample_scheme", type=str, default="systematic",
                   choices=["systematic"],
                   help="Resampling scheme used when ESS < num_particles/2 "
                        "(only used with --particle_filter)")

    # Variations / eval
    p.add_argument("--num_variations", type=int, default=6)
    p.add_argument("--eval_interval",  type=int, default=0,
                   help="Evaluate intermediate MMD every N steps (0 = auto ~5 checkpoints)")

    # Backprop subsampling: generate num_variations samples for the loss, but only
    # backprop through a subset — cuts backward-pass cost without shrinking the loss's
    # sample size. None = differentiate through every fresh variation (original behavior).
    p.add_argument("--backsel_k", type=int, default=None,
                   help="Of the freshly-generated variations, how many to backprop "
                        "through each step (None = all)")
    p.add_argument("--backsel_rule", type=str, default="uniform",
                   choices=["uniform", "witness"],
                   help="How to choose the backsel_k differentiated variations: "
                        "'uniform' (default, no extra cost) or 'witness' "
                        "(MMD witness-function importance sampling — scores all "
                        "fresh variations cheaply first, then backprops through "
                        "the highest-|score| subset for a lower-variance gradient)")
    p.add_argument("--witness_floor", type=float, default=0.3,
                   help="Uniform-mixing floor for witness sampling probabilities "
                        "(0 = pure importance sampling, 1 = uniform). Defensive "
                        "mixture p_i = floor/n + (1-floor)*|w_i|/sum(|w|); "
                        "recommended 0.3-0.5 to bound the worst-case weight and "
                        "avoid collapsing onto the same outliers every step")
    p.add_argument("--witness_temperature", type=float, default=1.0,
                   help="Reshapes |score|^(1/T) before the witness_floor blend. T=1 "
                        "(default) is plain |w|; T>1 flattens the selection distribution "
                        "toward uniform; T<1 sharpens it toward the top-|score| rows.")
    p.add_argument("--witness_replacement", action="store_true",
                   help="Sample the backsel_k witness-selected indices WITH "
                        "replacement (a repeated index counts multiple times in "
                        "the differentiable batch). Default: without replacement "
                        "(recommended unless backsel_k is tiny relative to "
                        "num_variations) -- removes the 'same outlier picked "
                        "repeatedly' failure mode entirely.")
    p.add_argument("--uniform_normalize_by_nsel", action="store_true",
                   help="With --backsel_rule uniform, rescale the gradient by "
                        "num_variations/backsel_k (Horvitz-Thompson-style), so its "
                        "magnitude is comparable across different backsel_k choices "
                        "instead of scaling ~linearly with backsel_k/num_variations. "
                        "No-op for backsel_rule='witness' or backsel_k=None. "
                        "Default: off (original, unnormalized behavior).")

    # Prompts
    p.add_argument("--prompt",          type=str, default="")
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--sprinter_variation_prompt", type=str,
                   default="a superrealistic professional photograph of")
    p.add_argument("--sprinter_eval_prompt", type=str,
                   default="a superrealistic professional photograph of")
    p.add_argument(
        "--target_prompts", type=str, nargs="+", default=None,
        metavar="NAME:PROMPT:N",
        help=(
            "Target prompt specs as 'name:prompt:n' triples. "
            "Example: --target_prompts 'Man:a portrait of a man:10' 'Woman:a portrait of a woman:10'. "
            "Defaults to a balanced man/woman split (10+10) when not provided."
        ),
    )

    # Models
    p.add_argument("--controlnet_model_id", type=str,
                   default=DEFAULT_CONTROLNET_MODEL_ID,
                   help="Must match the ControlNet eval uses (models.load_models()'s "
                        "own default, e.g. eval_scribbl_interpolation*.ipynb / "
                        "eval_all_experiments.ipynb calling load_models(device) with "
                        "no override) -- both default to the same "
                        "models.DEFAULT_CONTROLNET_MODEL_ID, so they can't silently "
                        "drift apart unless this flag is explicitly overridden.")
    p.add_argument("--sprinter_model_id",   type=str,
                   default="stabilityai/sdxl-turbo")
    p.add_argument("--architect_model_id",  type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0")

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run_unguided", action="store_true",
                   help="Also run the regular (unguided) path in parallel for "
                        "comparison -- an extra full UNet forward pass every step, "
                        "plus the final regular MMD eval and heatmap. Off by "
                        "default (MLGD-F path only, no comparison baseline).")

    # Mode
    p.add_argument("--mode", type=str, default="gender", choices=["gender", "age"],
                   help="Target distribution mode: 'gender' (prompt-based) or 'age' (age sweep)")

    # Age mode args (only used when --mode age)
    p.add_argument("--age_min",    type=int,   default=10,
                   help="Minimum age (inclusive) for age mode")
    p.add_argument("--age_max",    type=int,   default=80,
                   help="Maximum age (exclusive) for age mode")
    p.add_argument("--age_step",   type=int,   default=1,
                   help="Step between ages (1=every year, 5=every 5 years)")
    p.add_argument("--n_per_age",  type=int,   default=0,
                   help="Images per age value. 0 = auto (~100 total)")
    p.add_argument("--age_gender", type=str,   default="man",
                   help="Gender word used in age prompt (man/woman)")
    p.add_argument("--age_source", type=str,   default="mid", choices=["min", "mid", "max"],
                   help="Which age's portrait to extract the init HED scribble from")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pil_images_to_tensor(pil_list, device):
    tensors = [TF.to_tensor(img).unsqueeze(0) for img in pil_list]
    return torch.cat(tensors, dim=0).to(device)


def save_image_list_npy(pil_list, path):
    """Save a list of PIL images as [N, H, W, 3] uint8 numpy array."""
    arr = np.stack([np.array(img) for img in pil_list], axis=0)
    np.save(path, arr)


def extract_scribble_hed(pil_image):
    """Extract a HED scribble from a PIL portrait image."""
    from controlnet_aux import HEDdetector
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
    return hed(pil_image, scribble=True)


def compute_clip_softmax(pil_list, clip_model, clip_processor,
                         man_prompt, woman_prompt, device):
    """
    CLIP softmax probability over [man_prompt, woman_prompt] for each image.

    Returns:
        (results, image_features_np)
        results: list of dicts {"p_male", "p_female", "label"}
    """
    import torch.nn.functional as F

    text_inputs = clip_processor(
        text=[man_prompt, woman_prompt], return_tensors="pt", padding=True,
    ).to(device)

    clip_model.to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )
        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    print(f"  [softmax] text_features shape: {text_features.shape}  "
          f"dtype: {text_features.dtype}", flush=True)

    all_image_features = []
    for start in range(0, len(pil_list), 8):
        batch = pil_list[start:start + 8]
        tensors = torch.cat(
            [TF.to_tensor(img).unsqueeze(0) for img in batch], dim=0
        ).to(device)
        with torch.no_grad():
            all_image_features.append(
                encode_images_clip(tensors, clip_model, clip_processor)
            )
    image_features = torch.cat(all_image_features, dim=0)

    probs = F.softmax((image_features @ text_features.T) * 100.0, dim=-1).cpu().numpy()
    results = [
        {"p_male": float(pm), "p_female": float(pf),
         "label": "male" if pm > 0.5 else "female"}
        for pm, pf in probs
    ]

    clip_model.to("cpu")
    return results, image_features.cpu().numpy()


# ---------------------------------------------------------------------------
# Target builders — return the same interface regardless of mode
# ---------------------------------------------------------------------------

def build_targets_gender(args, sprinter, clip_model, clip_processor, device,
                         sobel_cond_pil):
    """
    Build target distribution from prompt groups (gender / custom).

    Returns
    -------
    target_images_per_group : dict[name -> list[PIL]]
    clip_embs_per_group     : dict[name -> Tensor [n, 768]]
    all_clip_embeddings     : Tensor [N, 768]
    group_names             : list[str]
    group_sizes             : list[int]
    group_colors            : list[str]
    group_markers           : list[str]
    pca_fixed               : fitted PCA(n_components=2)
    source_image            : PIL  (used for HED extraction)
    scribble_pil            : PIL  (HED scribble)
    N_total                 : int
    target_groups           : list of (name, prompt, n, color, marker) tuples
    """
    _DEFAULT_COLORS  = ["royalblue", "crimson", "limegreen", "orange",
                        "mediumpurple", "gold", "deepskyblue", "hotpink"]
    _DEFAULT_MARKERS = ["o", "x", "^", "s", "D", "P", "v", "<"]

    if args.target_prompts:
        target_groups = []
        for idx, spec in enumerate(args.target_prompts):
            parts = spec.split(":", 2)
            if len(parts) != 3:
                raise ValueError(f"--target_prompts entry '{spec}' must be 'name:prompt:n'")
            name, prompt_text, n_str = parts
            target_groups.append((
                name.strip(), prompt_text.strip(), int(n_str),
                _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)],
                _DEFAULT_MARKERS[idx % len(_DEFAULT_MARKERS)],
            ))
    else:
        target_groups = [
            ("Man",   "a superrealistic portrait photograph of a man, studio lighting",   10,
             "royalblue", "o"),
            ("Woman", "a superrealistic portrait photograph of a woman, studio lighting", 10,
             "crimson",   "x"),
        ]

    group_names   = [g[0] for g in target_groups]
    group_colors  = [g[3] for g in target_groups]
    group_markers = [g[4] for g in target_groups]
    N_total       = sum(g[2] for g in target_groups)
    print(f"Target groups: {[(g[0], g[2]) for g in target_groups]}", flush=True)

    # Generate a few male portraits on the oval scribble for HED extraction
    # Always use a male prompt for the scribble regardless of target groups
    _male_prompt = "a superrealistic portrait photograph of a man, studio lighting"
    print("Generating male portraits for HED scribble extraction...", flush=True)
    with torch.no_grad():
        _scribble_init_imgs, _ = generate_and_store_cs(
            sprinter, _male_prompt,
            sobel_cond_pil, 3, batch_size=2, cn_scale=args.controlnet_scale,
        )

    # Extract HED scribble from a male portrait
    print("Extracting HED scribble...", flush=True)
    source_image = _scribble_init_imgs[2]
    scribble_pil = extract_scribble_hed(source_image)

    # Regenerate targets conditioned on HED scribble
    print(f"Regenerating {N_total} targets conditioned on HED scribble...", flush=True)
    target_images_per_group = {}
    with torch.no_grad():
        for name, prompt_text, n, _, _ in target_groups:
            imgs, _ = generate_and_store_cs(
                sprinter, prompt_text,
                scribble_pil, n, batch_size=2, cn_scale=args.controlnet_scale,
            )
            target_images_per_group[name] = imgs
            plot_row(imgs, f"Target: {name}",
                     save_path=os.path.join(args.output_dir, f"target_samples_{name}.png"))

    # Encode to CLIP
    print("Encoding targets to CLIP...", flush=True)
    clip_model.to(device)
    clip_embs_per_group = {}
    with torch.no_grad():
        for name, imgs in target_images_per_group.items():
            clip_embs_per_group[name] = encode_images_clip(
                pil_images_to_tensor(imgs, device), clip_model, clip_processor)
    clip_model.to("cpu")
    all_clip_embeddings = torch.cat(list(clip_embs_per_group.values()), dim=0)
    group_sizes = [len(imgs) for imgs in target_images_per_group.values()]
    print(f"Target CLIP embeddings: {all_clip_embeddings.shape}", flush=True)

    # PCA — fit on first and last group only (the two extremes)
    # so the principal axis aligns with the target distribution's main direction
    anchor_embs = np.vstack([
        clip_embs_per_group[group_names[0]].cpu().numpy(),
        clip_embs_per_group[group_names[-1]].cpu().numpy(),
    ])
    pca_fixed = PCA(n_components=2)
    pca_fixed.fit(anchor_embs)
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, _, _, color, marker), embs in zip(target_groups, clip_embs_per_group.values()):
        coords = pca_fixed.transform(embs.cpu().numpy())
        ax.scatter(coords[:, 0], coords[:, 1], c=color, label=name,
                   marker=marker, alpha=0.7, s=50)
    ax.set_title(f"PCA of Target CLIP Embeddings\n(fitted on '{group_names[0]}' vs '{group_names[-1]}')")
    ax.legend(); ax.grid(True, alpha=0.3)
    pca_path = os.path.join(args.output_dir, "target_clip_pca.png")
    fig.savefig(pca_path, dpi=100, bbox_inches="tight"); plt.close(fig)

    return (target_images_per_group, clip_embs_per_group, all_clip_embeddings,
            group_names, group_sizes, group_colors, group_markers,
            pca_fixed, source_image, scribble_pil, N_total, target_groups, pca_path)


def build_targets_age(args, sprinter, clip_model, clip_processor, device,
                      sobel_cond_pil):
    """
    Build target distribution as an age sweep.

    Each age value becomes its own "group". The scribble is taken directly
    from sobel_cond_pil (oval) — no HED extraction needed since there is
    no single portrait to extract from.

    Returns the same interface as build_targets_gender.
    """
    ages      = list(range(args.age_min, args.age_max, args.age_step))
    n_per_age = args.n_per_age if args.n_per_age > 0 else max(1, round(100 / len(ages)))
    N_total   = len(ages) * n_per_age

    # Color = plasma colormap along the age axis
    age_colors  = [plt.cm.plasma((a - ages[0]) / max(ages[-1] - ages[0], 1))
                   for a in ages]
    age_markers = ["o"] * len(ages)

    target_groups = [
        (str(age),
         f"a superrealistic portrait photograph of a {age}-year-old {args.age_gender}, "
         "studio lighting, sharp focus, photographic",
         n_per_age,
         age_colors[i],
         age_markers[i])
        for i, age in enumerate(ages)
    ]

    group_names   = [g[0] for g in target_groups]
    group_colors  = [g[3] for g in target_groups]
    group_markers = [g[4] for g in target_groups]

    print(f"Age mode: {len(ages)} ages × {n_per_age} = {N_total} target images", flush=True)

    # Generate targets on oval scribble (no HED step for age mode)
    print("Generating age targets...", flush=True)
    target_images_per_group = {}
    with torch.no_grad():
        for name, prompt_text, n, _, _ in target_groups:
            imgs, _ = generate_and_store_cs(
                sprinter, prompt_text,
                sobel_cond_pil, n, batch_size=2, cn_scale=args.controlnet_scale,
            )
            target_images_per_group[name] = imgs

    # Log a few sample ages
    for age in ages[::max(1, len(ages)//5)]:
        plot_row(target_images_per_group[str(age)], f"Age {age}",
                 save_path=os.path.join(args.output_dir, f"target_samples_age{age}.png"))

    # Extract HED scribble from a portrait at the requested age (min/mid/max
    # of the sweep) -- default "mid" matches prior runs.
    source_age = {"min": ages[0], "mid": ages[len(ages) // 2], "max": ages[-1]}[args.age_source]
    print(f"Extracting HED scribble from age-{source_age} portrait ({args.age_source})...", flush=True)
    source_image = target_images_per_group[str(source_age)][0]
    scribble_pil = extract_scribble_hed(source_image)

    # Encode to CLIP
    print("Encoding age targets to CLIP...", flush=True)
    clip_model.to(device)
    clip_embs_per_group = {}
    with torch.no_grad():
        for name, imgs in target_images_per_group.items():
            clip_embs_per_group[name] = encode_images_clip(
                pil_images_to_tensor(imgs, device), clip_model, clip_processor)
    clip_model.to("cpu")
    all_clip_embeddings = torch.cat(list(clip_embs_per_group.values()), dim=0)
    group_sizes = [len(imgs) for imgs in target_images_per_group.values()]
    print(f"Age target CLIP embeddings: {all_clip_embeddings.shape}", flush=True)

    # PCA — fit on youngest vs oldest bracket, colour by age
    n_anchor = min(n_per_age * max(1, len(ages)//10), len(ages) * n_per_age // 2)
    pca_fixed = PCA(n_components=2)
    pca_fixed.fit(all_clip_embeddings.cpu().numpy())

    age_vals = np.array([int(n) for n in group_names for _ in range(n_per_age)])
    all_coords = pca_fixed.transform(all_clip_embeddings.cpu().numpy())
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(all_coords[:, 0], all_coords[:, 1],
                    c=age_vals, cmap="plasma", s=40, alpha=0.7,
                    edgecolors="white", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="Age")
    # Annotate every 10th age
    offset = 0
    for age in ages:
        c = all_coords[offset:offset + n_per_age].mean(axis=0)
        if age % 10 == 0:
            ax.annotate(str(age), c, fontsize=8, ha="center",
                        xytext=(0, 4), textcoords="offset points")
        offset += n_per_age
    ax.set_title("PCA of Age Target CLIP Embeddings"); ax.grid(True, alpha=0.3)
    pca_path = os.path.join(args.output_dir, "target_clip_pca.png")
    fig.savefig(pca_path, dpi=100, bbox_inches="tight"); plt.close(fig)

    return (target_images_per_group, clip_embs_per_group, all_clip_embeddings,
            group_names, group_sizes, group_colors, group_markers,
            pca_fixed, source_image, scribble_pil, N_total, target_groups, pca_path)


def sample_target_embeddings(all_clip_embeddings, n, generator=None):
    """Draw exactly n rows from all_clip_embeddings, for a fixed-sample-size final
    MMD comparison independent of however many target images the run happened to
    build. Without replacement when there are >= n distinct target embeddings
    already available; with replacement (bootstrap) otherwise."""
    total = all_clip_embeddings.shape[0]
    if total >= n:
        idx = torch.randperm(total, generator=generator)[:n]
    else:
        idx = torch.randint(0, total, (n,), generator=generator)
    return all_clip_embeddings[idx.to(all_clip_embeddings.device)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    if args.particle_filter and args.run_unguided:
        raise ValueError("--particle_filter and --run_unguided are not supported "
                          "together: the regular path is a single unguided "
                          "trajectory, which doesn't have a natural particle-filter "
                          "counterpart here.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    steps_dir = os.path.join(args.output_dir, "steps")
    os.makedirs(steps_dir, exist_ok=True)

    n_eval = 10  # number of Sprinter photos per MMD evaluation

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    import wandb

    # ── 1. Load models ─────────────────────────────────────────────────────
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

    # ── 2. Base oval image + Sobel (for initial target generation) ──────────
    base_image_pil, base_tensor = build_base_image(device)
    with torch.no_grad():
        sobel_cond_tensor = sobel_proxy(base_tensor, device)
        sobel_cond_pil    = T.ToPILImage()(sobel_cond_tensor.squeeze(0).cpu())

    # ── 3–6. Build target distribution (mode-dependent) ───────────────────────
    print(f"Mode: {args.mode}", flush=True)
    if args.mode == "gender":
        (target_images_per_group, clip_embs_per_group, all_clip_embeddings,
         group_names, group_sizes, group_colors, group_markers,
         pca_fixed, source_image, scribble_pil, N_total, target_groups, pca_path) =             build_targets_gender(args, sprinter, clip_model, clip_processor,
                                 device, sobel_cond_pil)
    else:  # age
        (target_images_per_group, clip_embs_per_group, all_clip_embeddings,
         group_names, group_sizes, group_colors, group_markers,
         pca_fixed, source_image, scribble_pil, N_total, target_groups, pca_path) =             build_targets_age(args, sprinter, clip_model, clip_processor,
                              device, sobel_cond_pil)

    source_image.save(os.path.join(args.output_dir, "source_portrait.png"))
    scribble_pil.save(os.path.join(args.output_dir, "scribble.png"))

    # ── 7. wandb init ───────────────────────────────────────────────────────
    eval_interval = (args.eval_interval if args.eval_interval > 0
                     else max(1, (args.n_steps - args.start_step) // 5))

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,  # None = use whoever is logged in
        config={
            "seed":                         args.seed,
            "run_unguided":                 args.run_unguided,
            "prompt":                       args.prompt,
            "negative_prompt":              args.negative_prompt,
            "n_targets":                    N_total,
            "target_groups":                {name: {"prompt": pt, "n": n} for name, pt, n, _, _ in target_groups},
            "n_steps":                      args.n_steps,
            "start_step":                   args.start_step,
            "strength":                     1 - args.start_step / args.n_steps,
            "steps_run":                    args.n_steps - args.start_step,
            "scheduler_type":               type(architect.scheduler).__name__,
            "num_variations":               args.num_variations,
            "backsel_k":                    args.backsel_k,
            "backsel_rule":                 args.backsel_rule,
            "uniform_normalize_by_nsel":    args.uniform_normalize_by_nsel,
            "witness_floor":                args.witness_floor,
            "witness_temperature":          args.witness_temperature,
            "witness_replacement":          args.witness_replacement,
            "base_zeta":                    args.base_zeta,
            "guidance_scale":               args.guidance_scale,
            "controlnet_scale":             args.controlnet_scale,
            "edge_method":                  "hed_scribble",
            "n_eval":                       n_eval,
            "eval_interval":                eval_interval,
            "lora_path":                    args.lora_path,
            "architect_unet_path":          args.architect_unet_path,
            "architect_model":              args.architect_model_id,
            "sprinter_model":               args.sprinter_model_id,
            "sprinter_variation_prompt":    args.sprinter_variation_prompt,
            "sprinter_eval_prompt":         args.sprinter_eval_prompt,
            "loss_fn":                      args.loss_fn,
            "loss_scale":                   args.loss_scale,
            "bandwidth_scale":              args.bandwidth_scale,
            "kernel_alpha":                 args.kernel_alpha,
            "mode":                         args.mode,
            "use_adam":                     args.use_adam,
            "adam_lr":                      args.adam_lr,
            "adam_beta1":                   args.adam_beta1,
            "adam_beta2":                   args.adam_beta2,
            "adam_eps":                     args.adam_eps,
            "particle_filter":              args.particle_filter,
            "num_particles":                args.num_particles if args.particle_filter else None,
            "beta_min":                     args.beta_min if args.particle_filter else None,
            "beta_max":                     args.beta_max if args.particle_filter else None,
            "resample_scheme":              args.resample_scheme if args.particle_filter else None,
        },
    )
    print(f"✅ wandb run: {run.name}", flush=True)

    # Only these two images go to wandb here -- target_clip_pca and the per-group
    # target_samples galleries are saved locally (npy + PNGs below) but not
    # uploaded, per the "only source scribble / source portrait / final MLGD-F
    # scribble / per-step plot" wandb policy for this script.
    wandb.log({
        "scribble":        wandb.Image(scribble_pil),
        "source_portrait": wandb.Image(source_image),
    })
    print("✅ Input images logged to wandb.", flush=True)

    # ── 8. Prepare MLGD-F loop ──────────────────────────────────────────────
    height, width   = 512, 512
    n_steps         = args.n_steps
    start_step      = args.start_step
    prompt          = args.prompt
    negative_prompt = args.negative_prompt

    sprinter.vae.to(dtype=torch.float32)
    setup_gradient_checkpointing(architect, sprinter)

    with torch.no_grad():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = architect.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            device=device, do_classifier_free_guidance=True, num_images_per_prompt=1,
        )

    architect.scheduler.set_timesteps(n_steps, device=device)
    timesteps         = architect.scheduler.timesteps
    scheduler_regular = copy.deepcopy(architect.scheduler) if args.run_unguided else None

    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]], dtype=prompt_embeds.dtype, device=device
    )
    added_cond_kwargs = {
        "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0),
        "time_ids":    add_time_ids.repeat(2, 1),
    }
    cfg_encoder_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # SDEdit-style init: encode scribble -> latent, noise to start_step
    with torch.no_grad():
        scribble_tensor = TF.to_tensor(scribble_pil).unsqueeze(0).to(device).to(torch.float32)
        scribble_tensor = (scribble_tensor * 2.0) - 1.0
        scribble_latent = architect.vae.encode(scribble_tensor).latent_dist.mean
        scribble_latent = scribble_latent * architect.vae.config.scaling_factor

    t_start        = timesteps[start_step]
    alphas_cumprod = architect.scheduler.alphas_cumprod.to(device)
    alpha          = alphas_cumprod[t_start.long()].to(torch.float32)
    noise          = torch.randn_like(scribble_latent)
    latents        = ((alpha ** 0.5) * scribble_latent + ((1 - alpha) ** 0.5) * noise).to(torch.float16)
    latents_regular = latents.detach().clone()

    if args.particle_filter:
        # Independently-noised particles from the same SDEdit scribble latent --
        # the initial (unweighted) particle cloud. scribble_latent is [1,C,H,W]
        # and broadcasts against the [B,C,H,W] particle noise below.
        particle_noise = torch.randn(
            args.num_particles, *scribble_latent.shape[1:],
            device=device, dtype=scribble_latent.dtype,
        )
        latents = (
            (alpha ** 0.5) * scribble_latent + ((1 - alpha) ** 0.5) * particle_noise
        ).to(torch.float16)

    timesteps_to_run = timesteps[start_step:]
    print(f"✅ Ready. Starting from step {start_step}/{n_steps}  (t={t_start.item():.0f})", flush=True)
    print(f"   Running {len(timesteps_to_run)} MLGD-F steps...", flush=True)

    step_gradients = []
    step_vis_data  = []
    backsel_generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None

    # Persistent Adam moment buffers across the whole guidance trajectory (only
    # used when --use_adam). Accumulated in float32 regardless of latents' own
    # dtype (float16 here) -- fp16 accumulation of squared gradients in adam_v can
    # underflow/overflow and destabilize the sqrt(v)+eps denominator, so moments
    # are always tracked in fp32 and only the final update is cast back down.
    adam_m = torch.zeros_like(latents, dtype=torch.float32)
    adam_v = torch.zeros_like(latents, dtype=torch.float32)
    # Per-particle step counters in particle-filter mode (a duplicated particle
    # carries its ancestor's counter forward after resampling); a single shared
    # counter otherwise.
    adam_t = [0] * args.num_particles if args.particle_filter else 0
    target_clip_np = all_clip_embeddings.cpu().numpy()
    softmax_man_prompt   = target_groups[-1][1]   # last group (most masculine)
    softmax_woman_prompt = target_groups[0][1]    # first group (most feminine)

    if args.loss_fn == "mmd":
        loss_fn = partial(compute_mmd, bandwidth_scale=args.bandwidth_scale,
                          kernel_alpha=args.kernel_alpha)
    else:
        loss_fn = LOSS_FNS[args.loss_fn]

    # ── Baseline visualisation (before any correction) ──────────────────────
    # In particle-filter mode this only shows particle 0's baseline -- the full
    # particle cloud's step-0 state is shown by the first per-step
    # visualize_particle_step call inside the main loop instead.
    with torch.no_grad():
        baseline_latents = latents[0:1].detach() if args.particle_filter else latents.detach()
        baseline_noise_pred = predict_noise_cfg(
            architect.unet, architect.scheduler, baseline_latents,
            timesteps_to_run[0], cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
        )
        baseline_pred_x0 = compute_pred_x0_direct(
            architect.scheduler, baseline_noise_pred, timesteps_to_run[0], baseline_latents,
        )
        baseline_px = architect.vae.decode(
            (baseline_pred_x0 / architect.vae.config.scaling_factor).to(architect.vae.dtype)
        ).sample
        baseline_px_norm = torch.clamp((baseline_px + 1.0) / 2.0, 0.0, 1.0)

        sprinter.vae.to(dtype=torch.float16)
        baseline_var_images = [
            sprinter(
                prompt=args.sprinter_variation_prompt,
                image=baseline_px_norm,
                num_inference_steps=2,
                guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.controlnet_scale,
                output_type="pil",
            ).images[0]
            for _ in range(n_eval)
        ]
        sprinter.vae.to(dtype=torch.float32)

        var_tensors = torch.cat(
            [TF.to_tensor(img).unsqueeze(0) for img in baseline_var_images], dim=0
        ).to(device)
        clip_model.to(device)
        baseline_clip_flat = encode_images_clip(
            var_tensors, clip_model, clip_processor
        ).cpu().numpy()
        clip_model.to("cpu")

        sd_baseline = {
            "step": 0, "timestep": timesteps_to_run[0].item(),
            "mmd_loss": 0.0, "zeta_i": 0.0,
            "latents_step_cpu":         baseline_latents.cpu(),
            "pred_x0_cpu":              baseline_pred_x0.detach().cpu(),
            "variation_clip_flat":      baseline_clip_flat,
        }
        if args.run_unguided:
            sd_baseline["latents_step_regular_cpu"] = baseline_latents.cpu()
            sd_baseline["pred_x0_regular_cpu"]       = baseline_pred_x0.detach().cpu()

    visualize_step(sd_baseline, architect, sprinter, target_clip_np,
                   num_cond=4, save_path=os.path.join(steps_dir, "step_baseline.png"),
                   pca_fixed=pca_fixed, group_names=group_names, group_sizes=group_sizes,
                   controlnet_scale=args.controlnet_scale)
    print("✅ Baseline visualisation saved.", flush=True)

    # ── 9. MLGD-F guidance loop ─────────────────────────────────────────────
    dps_start_time = time.time()
    for i, t in enumerate(timesteps_to_run):
        print(f"\n{'='*60}", flush=True)
        print(f"Step {i+1}/{len(timesteps_to_run)}  (t={t})", flush=True)
        print(f"{'='*60}", flush=True)

        if args.particle_filter:
            B = args.num_particles
            particle_losses = torch.zeros(B)
            new_latents = torch.empty_like(latents)

            # Linear beta annealing over the guidance trajectory (common in SMC:
            # start close to the untilted prior, sharpen the tilt toward the
            # target as steps progress).
            frac = i / max(len(timesteps_to_run) - 1, 1)
            beta_t = args.beta_min + (args.beta_max - args.beta_min) * frac

            def vae_decode_checkpoint(lat):
                return architect.vae.decode(lat.to(architect.vae.dtype)).sample

            for b in range(B):
                latents_step_b = latents[b:b + 1].detach().requires_grad_(True)

                noise_pred_b = predict_noise_cfg(
                    architect.unet, architect.scheduler,
                    latents_step_b, t, cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
                )
                pred_x0_b = compute_pred_x0_direct(architect.scheduler, noise_pred_b, t, latents_step_b)
                pred_x0_scaled_b = pred_x0_b / architect.vae.config.scaling_factor

                pixel_x0_b = torch.utils.checkpoint.checkpoint(
                    vae_decode_checkpoint, pred_x0_scaled_b, use_reentrant=False
                )
                pixel_x0_norm_b = torch.clamp((pixel_x0_b + 1.0) / 2.0, 0.0, 1.0)

                grad_b, loss_b, zeta_b, loss_norm_b, vl_clip_flat_b = run_dps_step_clip(
                    latents=latents_step_b,
                    latents_step=latents_step_b,
                    noise_pred=noise_pred_b,
                    pixel_x0_norm=pixel_x0_norm_b,
                    sprinter=sprinter,
                    all_clip_embeddings=all_clip_embeddings,
                    num_variations=args.num_variations,
                    variation_batch_size=1,
                    base_zeta_prime=args.base_zeta,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    vae=sprinter.vae,
                    vae_scaling_factor=sprinter.vae.config.scaling_factor,
                    variation_prompt=args.sprinter_variation_prompt,
                    loss_fn=loss_fn,
                    loss_scale=args.loss_scale,
                    controlnet_scale=args.controlnet_scale,
                    backsel_k=args.backsel_k,
                    backsel_rule=args.backsel_rule,
                    backsel_generator=backsel_generator,
                    witness_floor=args.witness_floor,
                    witness_temperature=args.witness_temperature,
                    witness_replacement=args.witness_replacement,
                    witness_bandwidth_scale=args.bandwidth_scale,
                    witness_kernel_alpha=args.kernel_alpha,
                    uniform_normalize_by_nsel=args.uniform_normalize_by_nsel,
                )

                grad_norm_b = grad_b.norm().item()
                zeta_val_b  = zeta_b.item() if isinstance(zeta_b, torch.Tensor) else zeta_b
                mmd_display_b = mmd_report(loss_b) if args.loss_fn == "mmd" else loss_b
                particle_losses[b] = loss_norm_b.item()
                print(f"  [P{b}] MMD={mmd_display_b.item():.6f}  ζi={zeta_val_b:.4f}  "
                      f"∥∇∥={grad_norm_b:.6f}", flush=True)

                if torch.isnan(grad_b).any():
                    print(f"  ⚠️  NaN in gradient at step {i} particle {b} — skipping correction",
                          flush=True)
                    correction_b = torch.zeros_like(latents_step_b)
                elif args.use_adam:
                    grad_f32_b = grad_b.float()
                    adam_t[b] += 1
                    adam_m[b:b + 1].mul_(args.adam_beta1).add_(grad_f32_b, alpha=1 - args.adam_beta1)
                    adam_v[b:b + 1].mul_(args.adam_beta2).addcmul_(
                        grad_f32_b, grad_f32_b, value=1 - args.adam_beta2
                    )
                    m_hat = adam_m[b:b + 1] / (1 - args.adam_beta1 ** adam_t[b])
                    v_hat = adam_v[b:b + 1] / (1 - args.adam_beta2 ** adam_t[b])
                    adam_update_b = args.adam_lr * m_hat / (v_hat.sqrt() + args.adam_eps)
                    correction_b = -adam_update_b.to(latents_step_b.dtype)
                else:
                    correction_b = -zeta_b * grad_b

                correction_norm_b = correction_b.norm().item()

                step_gradients.append({
                    "step":            i + 1,
                    "particle":        b,
                    "timestep":        t.item(),
                    "gradient_norm":   grad_norm_b,
                    "mmd_loss":        mmd_display_b.item(),
                    "zeta_i":          zeta_val_b,
                    "loss_norm":       loss_norm_b.item(),
                    "correction_norm": correction_norm_b,
                })

                wandb.log({
                    f"particle_{b}/mmd_loss":        mmd_display_b.item(),
                    f"particle_{b}/gradient_norm":   grad_norm_b,
                    f"particle_{b}/zeta":            zeta_val_b,
                    f"particle_{b}/correction_norm": correction_norm_b,
                }, commit=False)

                with torch.no_grad():
                    sd_b = {
                        "step":                i + 1,
                        "timestep":            t.item(),
                        "mmd_loss":            mmd_display_b.item(),
                        "zeta_i":              zeta_val_b,
                        "latents_step_cpu":    latents_step_b.detach().cpu(),
                        "pred_x0_cpu":         pred_x0_b.detach().cpu(),
                        "variation_clip_flat": vl_clip_flat_b,
                    }

                # One separate figure (with its own PCA plot vs. the shared
                # target) per particle -- not one figure shared across all B.
                visualize_step(
                    sd_b, architect, sprinter, target_clip_np,
                    num_cond=5, save_path=os.path.join(steps_dir, f"step_{i:03d}_particle_{b}.png"),
                    pca_fixed=pca_fixed, group_names=group_names, group_sizes=group_sizes,
                    controlnet_scale=args.controlnet_scale,
                    wandb_key=f"particle_{b}/step_visualization", commit=False,
                )

                new_latents[b:b + 1] = denoise_step(
                    architect.scheduler, noise_pred_b, t, latents_step_b, correction=correction_b
                )

                del grad_b, loss_b, mmd_display_b, loss_norm_b, zeta_b, correction_b
                del pixel_x0_b, pixel_x0_norm_b, pred_x0_b, noise_pred_b, latents_step_b
                gc.collect(); torch.cuda.empty_cache()

            # ── SMC reweighting + adaptive (systematic) resampling ───────────
            # Tilted target: Q(x) ∝ P(x) * exp(-beta_t * loss(x)) -- low-loss
            # particles get upweighted. Resample only when ESS < B/2, the
            # standard adaptive trigger, to avoid resampling (and its added
            # variance) every single step.
            log_w = -beta_t * particle_losses
            log_w = log_w - log_w.max()
            w = torch.exp(log_w)
            w = w / w.sum()
            ess = 1.0 / (w ** 2).sum().item()

            resampled_idx = None
            if ess < B / 2:
                idx = systematic_resample(w, generator=backsel_generator)
                new_latents = new_latents[idx].clone()
                if args.use_adam:
                    adam_m = adam_m[idx].clone()
                    adam_v = adam_v[idx].clone()
                    adam_t = [adam_t[j] for j in idx.tolist()]
                resampled_idx = idx.tolist()
                print(f"  [PF] Resampled at step {i+1}: ESS={ess:.2f} < {B/2:.1f}  "
                      f"idx={resampled_idx}", flush=True)
            else:
                print(f"  [PF] No resample at step {i+1}: ESS={ess:.2f} >= {B/2:.1f}", flush=True)

            wandb.log({
                "pf/beta_t":     beta_t,
                "pf/ess":        ess,
                "pf/mean_loss":  particle_losses.mean().item(),
                "pf/min_loss":   particle_losses.min().item(),
                "pf/resampled":  int(resampled_idx is not None),
            }, commit=True)

            latents = new_latents
            continue

        latents_step = latents.detach().requires_grad_(True)

        noise_pred = predict_noise_cfg(
            architect.unet, architect.scheduler,
            latents_step, t, cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
        )
        pred_x0 = compute_pred_x0_direct(architect.scheduler, noise_pred, t, latents_step)

        if args.run_unguided:
            # Extra full UNet forward pass every step -- off by default.
            latents_step_regular = latents_regular.detach()
            with torch.no_grad():
                noise_pred_regular = predict_noise_cfg(
                    architect.unet, scheduler_regular,
                    latents_step_regular, t, cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
                )
                pred_x0_regular = compute_pred_x0_direct(
                    scheduler_regular, noise_pred_regular, t, latents_step_regular
                )

        pred_x0_scaled = pred_x0 / architect.vae.config.scaling_factor

        def vae_decode_checkpoint(lat):
            return architect.vae.decode(lat.to(architect.vae.dtype)).sample

        pixel_x0      = torch.utils.checkpoint.checkpoint(
            vae_decode_checkpoint, pred_x0_scaled, use_reentrant=False
        )
        pixel_x0_norm = torch.clamp((pixel_x0 + 1.0) / 2.0, 0.0, 1.0)

        grad, mmd_loss, zeta_i, loss_norm, vl_clip_flat = run_dps_step_clip(
            latents=latents,
            latents_step=latents_step,
            noise_pred=noise_pred,
            pixel_x0_norm=pixel_x0_norm,
            sprinter=sprinter,
            all_clip_embeddings=all_clip_embeddings,
            num_variations=args.num_variations,
            variation_batch_size=1,
            base_zeta_prime=args.base_zeta,
            clip_model=clip_model,
            clip_processor=clip_processor,
            vae=sprinter.vae,
            vae_scaling_factor=sprinter.vae.config.scaling_factor,
            variation_prompt=args.sprinter_variation_prompt,
            loss_fn=loss_fn,
            loss_scale=args.loss_scale,
            controlnet_scale=args.controlnet_scale,
            backsel_k=args.backsel_k,
            backsel_rule=args.backsel_rule,
            backsel_generator=backsel_generator,
            witness_floor=args.witness_floor,
            witness_temperature=args.witness_temperature,
            witness_replacement=args.witness_replacement,
            witness_bandwidth_scale=args.bandwidth_scale,
            witness_kernel_alpha=args.kernel_alpha,
            uniform_normalize_by_nsel=args.uniform_normalize_by_nsel,
        )

        grad_norm = grad.norm().item()
        zeta_val  = zeta_i.item() if isinstance(zeta_i, torch.Tensor) else zeta_i
        # Display-only: mmd_loss is the plain squared MMD^2_U statistic (what zeta_i's
        # scaling and the gradient actually used) when --loss_fn=mmd; report its square
        # root for a human-readable number. Left as-is for --loss_fn=swd, which is
        # already a real-valued distance, not a squared statistic.
        mmd_display = mmd_report(mmd_loss) if args.loss_fn == "mmd" else mmd_loss
        print(f"  MMD={mmd_display.item():.6f}  ζi={zeta_val:.4f}  ∥∇∥={grad_norm:.6f}", flush=True)

        adam_log = {}
        if torch.isnan(grad).any():
            print(f"  ⚠️  NaN in gradient at step {i} — skipping correction", flush=True)
            correction = torch.zeros_like(latents_step)
        elif args.use_adam:
            # zeta_i is intentionally unused here -- it's a scalar 1/loss rescale,
            # Adam already adaptively rescales per-coordinate via its own moment
            # estimates, and applying both would double-normalize the step.
            # Moments accumulate in float32 even though latents_step is float16,
            # to avoid v underflowing/overflowing under fp16 accumulation.
            grad_f32 = grad.float()
            adam_t += 1
            adam_m.mul_(args.adam_beta1).add_(grad_f32, alpha=1 - args.adam_beta1)
            adam_v.mul_(args.adam_beta2).addcmul_(grad_f32, grad_f32, value=1 - args.adam_beta2)
            m_hat = adam_m / (1 - args.adam_beta1 ** adam_t)
            v_hat = adam_v / (1 - args.adam_beta2 ** adam_t)
            adam_update = args.adam_lr * m_hat / (v_hat.sqrt() + args.adam_eps)
            correction = -adam_update.to(latents_step.dtype)
            print(f"  [adam] t={adam_t}  ∥update∥={adam_update.norm().item():.6f}  "
                  f"∥m∥={adam_m.norm().item():.6f}  mean(v)={adam_v.mean().item():.6e}",
                  flush=True)
            adam_log = {
                "adam/step_norm": adam_update.norm().item(),
                "adam/m_norm":    adam_m.norm().item(),
                "adam/v_mean":    adam_v.mean().item(),
                "adam/t":         adam_t,
            }
        else:
            correction = -zeta_i * grad

        correction_norm = correction.norm().item()

        step_gradients.append({
            "step":            i + 1,
            "timestep":        t.item(),
            "gradient_norm":   grad_norm,
            "mmd_loss":        mmd_display.item(),
            "zeta_i":          zeta_val,
            "loss_norm":       loss_norm.item(),
            "correction_norm": correction_norm,
            **({"adam_step_norm": adam_log["adam/step_norm"]} if adam_log else {}),
        })

        wandb_log = {
            "step":            i + 1,
            "mmd_loss":        mmd_display.item(),
            "gradient_norm":   grad_norm,
            "zeta":            zeta_val,
            "correction_norm": correction_norm,
            **adam_log,
        }

        if args.run_unguided and i % eval_interval == 0:
            # unguided_mmd is already mmd_report()'d inside evaluate_distribution_mmd,
            # so this stays an apples-to-apples comparison with mmd_display.
            unguided_mmd, _, _ = evaluate_distribution_mmd(
                pred_x0_regular.detach(), architect.vae, architect.image_processor,
                sprinter, clip_model, clip_processor,
                all_clip_embeddings, args.sprinter_eval_prompt,
                n_eval=n_eval, device=device, controlnet_scale=args.controlnet_scale,
            )
            wandb_log["intermediate/unguided_cond_mmd"] = unguided_mmd
            wandb_log["intermediate/mlgd_f_cond_mmd"]   = mmd_display.item()
            wandb_log["intermediate/cond_mmd_delta"]     = mmd_display.item() - unguided_mmd
            print(f"  [eval] mlgd_f={mmd_display.item():.6f}  unguided={unguided_mmd:.6f}  "
                  f"delta={mmd_display.item()-unguided_mmd:.6f}", flush=True)

        wandb.log(wandb_log, commit=False)

        with torch.no_grad():
            sd = {
                "step":                     i + 1,
                "timestep":                 t.item(),
                "mmd_loss":                 mmd_display.item(),
                "zeta_i":                   zeta_val,
                "latents_step_cpu":         latents_step.detach().cpu(),
                "pred_x0_cpu":              pred_x0.detach().cpu(),
                "variation_clip_flat":      vl_clip_flat,
            }
            if args.run_unguided:
                sd["latents_step_regular_cpu"] = latents_step_regular.detach().cpu()
                sd["pred_x0_regular_cpu"]       = pred_x0_regular.detach().cpu()
            step_vis_data.append(sd)

        visualize_step(sd, architect, sprinter, target_clip_np,
                       num_cond=5, save_path=os.path.join(steps_dir, f"step_{i:03d}.png"),
                       pca_fixed=pca_fixed, group_names=group_names, group_sizes=group_sizes,
                       controlnet_scale=args.controlnet_scale)

        latents = denoise_step(
            architect.scheduler, noise_pred, t, latents_step, correction=correction
        )
        if args.run_unguided:
            with torch.no_grad():
                latents_regular = denoise_step(
                    scheduler_regular, noise_pred_regular, t, latents_step_regular
                )
            del pred_x0_regular, latents_step_regular, noise_pred_regular

        del grad, mmd_loss, mmd_display, loss_norm, zeta_i, correction
        del pixel_x0, pixel_x0_norm, pred_x0
        gc.collect(); torch.cuda.empty_cache()

    final_particle_losses = None
    if args.particle_filter:
        # `latents` still holds all B final particles here. Reduce to a single
        # "best" particle (lowest last-step training loss) for the rest of the
        # pipeline below, which is written for a single trajectory -- but keep
        # every particle's final scribble/loss around too, saved separately.
        final_particle_losses = particle_losses.clone()
        best_particle_idx = int(torch.argmin(final_particle_losses))
        print(f"\n✅ P-MLGD-F complete! Best particle: {best_particle_idx} "
              f"(loss={final_particle_losses[best_particle_idx].item():.6f})", flush=True)
        all_particle_latents = latents.clone()
        latents = latents[best_particle_idx:best_particle_idx + 1].clone()
    else:
        del latents_step, noise_pred
        torch.cuda.empty_cache()
        print(f"\n✅ MLGD-F complete! {len(step_vis_data)} steps stored.", flush=True)

    # ── 10. Final MMD evaluation ────────────────────────────────────────────
    # Uses a larger, fixed sample size (FINAL_MMD_N=250 on both sides) than the
    # per-step n_eval, for a lower-variance final number independent of however
    # many target images this run happened to build (N_total). Target side is
    # resampled from the existing all_clip_embeddings pool (bootstrap if it has
    # fewer than 250 rows) rather than regenerated from scratch.
    FINAL_MMD_N = 250
    final_mmd_generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    final_target_clip = sample_target_embeddings(
        all_clip_embeddings, FINAL_MMD_N, generator=final_mmd_generator
    )

    regular_mmd = None
    regular_eval_photos = []
    if args.run_unguided:
        print(f"Computing final MMD (regular, n={FINAL_MMD_N})...", flush=True)
        regular_mmd, regular_eval_photos, _ = evaluate_distribution_mmd(
            latents_regular, architect.vae, architect.image_processor,
            sprinter, clip_model, clip_processor,
            final_target_clip, eval_prompt=args.sprinter_eval_prompt,
            n_eval=FINAL_MMD_N, device=device, controlnet_scale=args.controlnet_scale,
        )

    print(f"Computing final MMD (MLGD-F, n={FINAL_MMD_N})...", flush=True)
    mlgd_f_mmd, mlgd_f_eval_photos, _ = evaluate_distribution_mmd(
        latents, architect.vae, architect.image_processor,
        sprinter, clip_model, clip_processor,
        final_target_clip, eval_prompt=args.sprinter_eval_prompt,
        n_eval=FINAL_MMD_N, device=device, controlnet_scale=args.controlnet_scale,
    )

    print(f"MLGD-F MMD  : {mlgd_f_mmd:.6f}",  flush=True)
    final_log = {"final/mlgd_f_mmd_n250": mlgd_f_mmd}
    if args.run_unguided:
        print(f"Regular MMD : {regular_mmd:.6f}", flush=True)
        print(f"Delta (↓ better for MLGD-F): {regular_mmd - mlgd_f_mmd:.6f}", flush=True)
        final_log["final/regular_mmd_n250"] = regular_mmd
        final_log["final/mmd_delta_n250"]   = regular_mmd - mlgd_f_mmd
    wandb.log(final_log, commit=False)

    # ── 11. Final visualisations ────────────────────────────────────────────
    with torch.no_grad():
        final_mlgd_f_pil = latent_to_pil(latents, architect.vae, architect.image_processor)
    final_mlgd_f_pil.save(os.path.join(args.output_dir, "final_scribble_mlgd_f.png"))

    if args.particle_filter:
        # Every particle's final scribble + loss, not just the best one picked
        # above for the single-trajectory pipeline below.
        particles_dir = os.path.join(args.output_dir, "particles")
        os.makedirs(particles_dir, exist_ok=True)
        with torch.no_grad():
            for b in range(all_particle_latents.shape[0]):
                pil_b = latent_to_pil(
                    all_particle_latents[b:b + 1], architect.vae, architect.image_processor
                )
                pil_b.save(os.path.join(particles_dir, f"final_scribble_particle_{b}.png"))
        with open(os.path.join(particles_dir, "final_losses.json"), "w") as f:
            json.dump({
                "final_particle_losses": final_particle_losses.tolist(),
                "best_particle_idx":     best_particle_idx,
            }, f, indent=2)
        print(f"✅ All {all_particle_latents.shape[0]} final particle scribbles + "
              f"losses saved to {particles_dir}/", flush=True)

    final_regular_pil = None
    if args.run_unguided:
        with torch.no_grad():
            final_regular_pil = latent_to_pil(latents_regular, architect.vae, architect.image_processor)
        final_regular_pil.save(os.path.join(args.output_dir, "final_scribble_regular.png"))

        heatmap_path = os.path.join(args.output_dir, "scribble_heatmap.png")
        compare_scribbles_heatmap(final_mlgd_f_pil, final_regular_pil, save_path=heatmap_path)
        print("✅ Scribble heatmap saved.", flush=True)

        plot_row(regular_eval_photos, f"Regular final photos  (MMD={regular_mmd:.4f})",
                 save_path=os.path.join(args.output_dir, "final_photos_regular.png"))

    plot_row(mlgd_f_eval_photos,  f"MLGD-F final photos   (MMD={mlgd_f_mmd:.4f})",
             save_path=os.path.join(args.output_dir, "final_photos_mlgd_f.png"))

    photo_groups = [("photos_mlgd_f", mlgd_f_eval_photos)]
    if args.run_unguided:
        photo_groups.append(("photos_regular", regular_eval_photos))
    for folder, photos in photo_groups:
        photo_dir = os.path.join(args.output_dir, folder)
        os.makedirs(photo_dir, exist_ok=True)
        for idx, photo in enumerate(photos):
            photo.save(os.path.join(photo_dir, f"photo_{idx:03d}.png"))

    # ── 12. wandb final logs ─────────────────────────────────────────────────
    # Only the final MLGD-F scribble goes to wandb here, per the "only source
    # scribble / source portrait / final MLGD-F scribble / per-step plot" wandb
    # policy for this script -- the regular scribble and the heatmap are still
    # saved locally (above) but not uploaded, and mlgd_f_eval_photos/
    # regular_eval_photos (250 each) are also local-only (too expensive to
    # upload every run).
    wandb_final = {
        "final_mlgd_f_mmd":      mlgd_f_mmd,
        "final_scribble_mlgd_f": wandb.Image(final_mlgd_f_pil),
    }
    if args.run_unguided:
        wandb_final["final_regular_mmd"]        = regular_mmd
        wandb_final["mmd_delta"]                = regular_mmd - mlgd_f_mmd
        wandb_final["mmd_relative_improvement"] = (regular_mmd - mlgd_f_mmd) / (regular_mmd + 1e-8)
    wandb.log(wandb_final)
    wandb.summary["final_mlgd_f_mmd"] = mlgd_f_mmd
    if args.run_unguided:
        wandb.summary["final_regular_mmd"] = regular_mmd
        wandb.summary["mmd_delta"]         = regular_mmd - mlgd_f_mmd
    wandb.summary["final_grad_norm"] = step_gradients[-1]["gradient_norm"]

    # ── Save numpy arrays ───────────────────────────────────────────────────
    npy_dir = os.path.join(args.output_dir, "npy")
    os.makedirs(npy_dir, exist_ok=True)

    save_image_list_npy(mlgd_f_eval_photos,  os.path.join(npy_dir, "photos_mlgd_f.npy"))
    if args.run_unguided:
        save_image_list_npy(regular_eval_photos, os.path.join(npy_dir, "photos_regular.npy"))
    for name, imgs in target_images_per_group.items():
        safe_name = name.lower().replace(" ", "_")
        save_image_list_npy(imgs, os.path.join(npy_dir, f"targets_{safe_name}.npy"))
    save_image_list_npy([source_image],      os.path.join(npy_dir, "source_portrait.npy"))
    save_image_list_npy([scribble_pil],      os.path.join(npy_dir, "scribble.npy"))
    save_image_list_npy([final_mlgd_f_pil],  os.path.join(npy_dir, "final_scribble_mlgd_f.npy"))
    if args.run_unguided:
        save_image_list_npy([final_regular_pil], os.path.join(npy_dir, "final_scribble_regular.npy"))
    print("✅ Image arrays saved to npy/", flush=True)

    for name, imgs in target_images_per_group.items():
        safe_name = name.lower().replace(" ", "_")
        photo_dir = os.path.join(args.output_dir, f"targets_{safe_name}")
        os.makedirs(photo_dir, exist_ok=True)
        for idx, photo in enumerate(imgs):
            photo.save(os.path.join(photo_dir, f"photo_{idx:03d}.png"))
    print("✅ Individual target portraits saved.", flush=True)

    # ── CLIP softmax / SWD (disabled — run offline via analysis.py) ─────────
    mlgd_f_stats  = {}
    regular_stats = {}
    swd_mlgd_f    = None
    swd_regular   = None

    optimization_time_sec = time.time() - dps_start_time

    # ── Save metrics.json ───────────────────────────────────────────────────
    npy_manifest = {
        "photos_mlgd_f": "npy/photos_mlgd_f.npy",
        **{f"targets_{n.lower().replace(' ','_')}": f"npy/targets_{n.lower().replace(' ','_')}.npy"
           for n in target_images_per_group},
        "source_portrait":       "npy/source_portrait.npy",
        "scribble":              "npy/scribble.npy",
        "final_scribble_mlgd_f": "npy/final_scribble_mlgd_f.npy",
    }
    if args.run_unguided:
        npy_manifest["photos_regular"]         = "npy/photos_regular.npy"
        npy_manifest["final_scribble_regular"] = "npy/final_scribble_regular.npy"

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({
            "args":  vars(args),
            "steps": step_gradients,
            "final_mlgd_f_mmd":       mlgd_f_mmd,
            "final_regular_mmd":      regular_mmd if args.run_unguided else None,
            "mmd_delta":              (regular_mmd - mlgd_f_mmd) if args.run_unguided else None,
            "final_mlgd_f_swd":       swd_mlgd_f,   # None — run analysis.py to compute
            "final_regular_swd":      swd_regular,  # None — run analysis.py to compute
            "swd_delta":              None,         # computed in analysis.py
            "optimization_time_sec":  optimization_time_sec,
            "mlgd_f_gender":          mlgd_f_stats,
            "regular_gender":         regular_stats,
            "npy": npy_manifest,
            # clip embeddings computed by analysis.py (run offline)
        }, f, indent=2)
    print(f"✅ metrics.json saved.  Optimization time: {optimization_time_sec/60:.1f} min",
          flush=True)

    wandb.finish()
    print(f"\n✅ All outputs saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
