"""
run_mlgd_f.py — MLGD-F pipeline entry point.

Cluster-ready script version of main.ipynb (scribble_cond_loss branch).

Pipeline:
    1. Load Architect + Sprinter + CLIP.
    2. Generate target distribution (man/woman portraits) and encode to CLIP.
    3. Extract HED scribble from one portrait (SDEdit-style init).
    4. Noise the scribble latent to start_step and run MLGD-F guidance,
       comparing against the regular (unguided) path at each step.
    5. Evaluate final MMD/SWD for both paths and log everything to wandb.

Usage:
    python scripts/run_mlgd_f.py --output_dir output/run_001 --seed 1
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
from metrics import compute_mmd, compute_swd, evaluate_distribution_mmd
from models import load_models, setup_gradient_checkpointing
from visualization import plot_row, visualize_step

LOSS_FNS = {"mmd": compute_mmd, "swd": compute_swd}


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

    # Variations / eval
    p.add_argument("--num_variations", type=int, default=6)
    p.add_argument("--eval_interval",  type=int, default=0,
                   help="Evaluate intermediate MMD every N steps (0 = auto ~5 checkpoints)")

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
                   default="xinsir/controlnet-scribble-sdxl-1.0")
    p.add_argument("--sprinter_model_id",   type=str,
                   default="stabilityai/sdxl-turbo")
    p.add_argument("--architect_model_id",  type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0")

    p.add_argument("--seed", type=int, default=None)
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
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
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

    # ── 3. Parse target prompts ────────────────────────────────────────────────
    _DEFAULT_COLORS  = ["royalblue", "crimson", "limegreen", "orange",
                        "mediumpurple", "gold", "deepskyblue", "hotpink"]
    _DEFAULT_MARKERS = ["o", "x", "^", "s", "D", "P", "v", "<"]

    if args.target_prompts:
        target_groups = []
        for idx, spec in enumerate(args.target_prompts):
            parts = spec.split(":", 2)
            if len(parts) != 3:
                raise ValueError(
                    f"--target_prompts entry '{spec}' must be 'name:prompt:n'"
                )
            name, prompt_text, n_str = parts
            target_groups.append((
                name.strip(), prompt_text.strip(), int(n_str),
                _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)],
                _DEFAULT_MARKERS[idx % len(_DEFAULT_MARKERS)],
            ))
    else:
        # default: balanced man / woman split (10 + 10)
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

    # ── 4. Generate initial targets (on oval scribble) for HED extraction ──
    print("Generating initial targets for HED scribble extraction...", flush=True)
    target_images_per_group_init = {}
    with torch.no_grad():
        for name, prompt_text, n, _, _ in target_groups:
            imgs, _ = generate_and_store_cs(
                sprinter, prompt_text,
                sobel_cond_pil, n, batch_size=2, cn_scale=args.controlnet_scale,
            )
            target_images_per_group_init[name] = imgs

    # Extract HED scribble from first image of first group
    print("Extracting HED scribble...", flush=True)
    source_image = target_images_per_group_init[group_names[0]][min(2, target_groups[0][2] - 1)]
    scribble_pil = extract_scribble_hed(source_image)
    source_image.save(os.path.join(args.output_dir, "source_portrait.png"))
    scribble_pil.save(os.path.join(args.output_dir, "scribble.png"))
    sobel_cond_pil = scribble_pil
    print("✅ HED scribble ready.", flush=True)

    # ── 5. Regenerate targets conditioned on the HED scribble ──────────────
    print(f"Regenerating {N_total} targets conditioned on HED scribble...", flush=True)
    target_images_per_group = {}
    with torch.no_grad():
        for name, prompt_text, n, _, _ in target_groups:
            imgs, _ = generate_and_store_cs(
                sprinter, prompt_text,
                sobel_cond_pil, n, batch_size=2, cn_scale=args.controlnet_scale,
            )
            target_images_per_group[name] = imgs
            plot_row(imgs, f"Target: {name}",
                     save_path=os.path.join(args.output_dir, f"target_samples_{name}.png"))

    # ── 6. Encode targets to CLIP ───────────────────────────────────────────
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
    print(f"Target CLIP embeddings: {all_clip_embeddings.shape}  groups: {list(zip(group_names, group_sizes))}", flush=True)

    # Target PCA — fit on all groups, one scatter per group
    pca_fixed = PCA(n_components=2)
    pca_fixed.fit(all_clip_embeddings.cpu().numpy())
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, _, _, color, marker), embs in zip(target_groups, clip_embs_per_group.values()):
        coords = pca_fixed.transform(embs.cpu().numpy())
        ax.scatter(coords[:, 0], coords[:, 1], c=color, label=name,
                   marker=marker, alpha=0.7, s=50)
    ax.set_title("PCA of Target CLIP Embeddings"); ax.legend(); ax.grid(True, alpha=0.3)
    pca_path = os.path.join(args.output_dir, "target_clip_pca.png")
    fig.savefig(pca_path, dpi=100, bbox_inches="tight"); plt.close(fig)

    # ── 7. wandb init ───────────────────────────────────────────────────────
    eval_interval = (args.eval_interval if args.eval_interval > 0
                     else max(1, (args.n_steps - args.start_step) // 5))

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,  # None = use whoever is logged in
        config={
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
        },
    )
    print(f"✅ wandb run: {run.name}", flush=True)

    wandb.log({
        "scribble":        wandb.Image(scribble_pil),
        "source_portrait": wandb.Image(source_image),
        "target_clip_pca": wandb.Image(pca_path),
        **{f"target_samples/{name}": [wandb.Image(p) for p in imgs]
           for name, imgs in target_images_per_group.items()},
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
    scheduler_regular = copy.deepcopy(architect.scheduler)

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

    timesteps_to_run = timesteps[start_step:]
    print(f"✅ Ready. Starting from step {start_step}/{n_steps}  (t={t_start.item():.0f})", flush=True)
    print(f"   Running {len(timesteps_to_run)} MLGD-F steps...", flush=True)

    step_gradients = []
    step_vis_data  = []
    target_clip_np = all_clip_embeddings.cpu().numpy()

    target_clip_np      = all_clip_embeddings.cpu().numpy()
    softmax_man_prompt   = target_groups[-1][1]   # last group (most masculine)
    softmax_woman_prompt = target_groups[0][1]    # first group (most feminine)

    if args.loss_fn == "mmd":
        loss_fn = partial(compute_mmd, bandwidth_scale=args.bandwidth_scale,
                          kernel_alpha=args.kernel_alpha)
    else:
        loss_fn = LOSS_FNS[args.loss_fn]

    # ── Baseline visualisation (before any correction) ──────────────────────
    with torch.no_grad():
        baseline_noise_pred = predict_noise_cfg(
            architect.unet, architect.scheduler, latents.detach(),
            timesteps_to_run[0], cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
        )
        baseline_pred_x0 = compute_pred_x0_direct(
            architect.scheduler, baseline_noise_pred, timesteps_to_run[0], latents.detach(),
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
            "latents_step_cpu":         latents.detach().cpu(),
            "latents_step_regular_cpu": latents.detach().cpu(),
            "pred_x0_cpu":              baseline_pred_x0.detach().cpu(),
            "pred_x0_regular_cpu":      baseline_pred_x0.detach().cpu(),
            "variation_clip_flat":      baseline_clip_flat,
        }

    visualize_step(sd_baseline, architect, sprinter, target_clip_np,
                   num_cond=4, save_path=os.path.join(steps_dir, "step_baseline.png"),
                   pca_fixed=pca_fixed, group_names=group_names, group_sizes=group_sizes)
    print("✅ Baseline visualisation saved.", flush=True)

    # ── 9. MLGD-F guidance loop ─────────────────────────────────────────────
    dps_start_time = time.time()
    for i, t in enumerate(timesteps_to_run):
        print(f"\n{'='*60}", flush=True)
        print(f"Step {i+1}/{len(timesteps_to_run)}  (t={t})", flush=True)
        print(f"{'='*60}", flush=True)

        latents_step         = latents.detach().requires_grad_(True)
        latents_step_regular = latents_regular.detach()

        noise_pred = predict_noise_cfg(
            architect.unet, architect.scheduler,
            latents_step, t, cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
        )
        with torch.no_grad():
            noise_pred_regular = predict_noise_cfg(
                architect.unet, scheduler_regular,
                latents_step_regular, t, cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
            )

        pred_x0 = compute_pred_x0_direct(architect.scheduler, noise_pred, t, latents_step)
        with torch.no_grad():
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
        )

        grad_norm = grad.norm().item()
        zeta_val  = zeta_i.item() if isinstance(zeta_i, torch.Tensor) else zeta_i
        print(f"  MMD={mmd_loss.item():.6f}  ζi={zeta_val:.4f}  ∥∇∥={grad_norm:.6f}", flush=True)

        if torch.isnan(grad).any():
            print(f"  ⚠️  NaN in gradient at step {i} — skipping correction", flush=True)
            correction = torch.zeros_like(latents_step)
        else:
            correction = -zeta_i * grad

        step_gradients.append({
            "step":            i + 1,
            "timestep":        t.item(),
            "gradient_norm":   grad_norm,
            "mmd_loss":        mmd_loss.item(),
            "zeta_i":          zeta_val,
            "loss_norm":       loss_norm.item(),
            "correction_norm": zeta_val * grad_norm,
        })

        wandb_log = {
            "step":            i + 1,
            "mmd_loss":        mmd_loss.item(),
            "gradient_norm":   grad_norm,
            "zeta":            zeta_val,
            "correction_norm": zeta_val * grad_norm,
        }

        if i % eval_interval == 0:
            unguided_mmd, _, _ = evaluate_distribution_mmd(
                pred_x0_regular.detach(), architect.vae, architect.image_processor,
                sprinter, clip_model, clip_processor,
                all_clip_embeddings, args.sprinter_eval_prompt,
                n_eval=n_eval, device=device,
            )
            wandb_log["intermediate/unguided_cond_mmd"] = unguided_mmd
            wandb_log["intermediate/mlgd_f_cond_mmd"]   = mmd_loss.item()
            wandb_log["intermediate/cond_mmd_delta"]     = mmd_loss.item() - unguided_mmd
            print(f"  [eval] mlgd_f={mmd_loss.item():.6f}  unguided={unguided_mmd:.6f}  "
                  f"delta={mmd_loss.item()-unguided_mmd:.6f}", flush=True)

        wandb.log(wandb_log, commit=False)

        with torch.no_grad():
            sd = {
                "step":                     i + 1,
                "timestep":                 t.item(),
                "mmd_loss":                 mmd_loss.item(),
                "zeta_i":                   zeta_val,
                "latents_step_cpu":         latents_step.detach().cpu(),
                "latents_step_regular_cpu": latents_step_regular.detach().cpu(),
                "pred_x0_cpu":              pred_x0.detach().cpu(),
                "pred_x0_regular_cpu":      pred_x0_regular.detach().cpu(),
                "variation_clip_flat":      vl_clip_flat,
            }
            step_vis_data.append(sd)

        visualize_step(sd, architect, sprinter, target_clip_np,
                       num_cond=5, save_path=os.path.join(steps_dir, f"step_{i:03d}.png"),
                       pca_fixed=pca_fixed, group_names=group_names, group_sizes=group_sizes)

        latents = denoise_step(
            architect.scheduler, noise_pred, t, latents_step, correction=correction
        )
        with torch.no_grad():
            latents_regular = denoise_step(
                scheduler_regular, noise_pred_regular, t, latents_step_regular
            )

        del grad, mmd_loss, loss_norm, zeta_i, correction
        del pixel_x0, pixel_x0_norm, pred_x0, pred_x0_regular
        del latents_step_regular, noise_pred_regular
        gc.collect(); torch.cuda.empty_cache()

    del latents_step, noise_pred
    torch.cuda.empty_cache()
    print(f"\n✅ MLGD-F complete! {len(step_vis_data)} steps stored.", flush=True)

    # ── 10. Final MMD evaluation ────────────────────────────────────────────
    print("Computing final MMD (regular)...", flush=True)
    regular_mmd, regular_eval_photos, _ = evaluate_distribution_mmd(
        latents_regular, architect.vae, architect.image_processor,
        sprinter, clip_model, clip_processor,
        all_clip_embeddings, eval_prompt=args.sprinter_eval_prompt,
        n_eval=n_eval, device=device,
    )

    print("Computing final MMD (MLGD-F)...", flush=True)
    mlgd_f_mmd, mlgd_f_eval_photos, _ = evaluate_distribution_mmd(
        latents, architect.vae, architect.image_processor,
        sprinter, clip_model, clip_processor,
        all_clip_embeddings, eval_prompt=args.sprinter_eval_prompt,
        n_eval=n_eval, device=device,
    )

    print(f"Regular MMD : {regular_mmd:.6f}", flush=True)
    print(f"MLGD-F MMD  : {mlgd_f_mmd:.6f}",  flush=True)
    print(f"Delta (↓ better for MLGD-F): {regular_mmd - mlgd_f_mmd:.6f}", flush=True)

    # ── 11. Final visualisations ────────────────────────────────────────────
    with torch.no_grad():
        final_mlgd_f_pil  = latent_to_pil(latents,         architect.vae, architect.image_processor)
        final_regular_pil = latent_to_pil(latents_regular, architect.vae, architect.image_processor)

    final_mlgd_f_pil.save(os.path.join(args.output_dir, "final_scribble_mlgd_f.png"))
    final_regular_pil.save(os.path.join(args.output_dir, "final_scribble_regular.png"))

    heatmap_path = os.path.join(args.output_dir, "scribble_heatmap.png")
    compare_scribbles_heatmap(final_mlgd_f_pil, final_regular_pil, save_path=heatmap_path)
    print("✅ Scribble heatmap saved.", flush=True)

    plot_row(regular_eval_photos, f"Regular final photos  (MMD={regular_mmd:.4f})",
             save_path=os.path.join(args.output_dir, "final_photos_regular.png"))
    plot_row(mlgd_f_eval_photos,  f"MLGD-F final photos   (MMD={mlgd_f_mmd:.4f})",
             save_path=os.path.join(args.output_dir, "final_photos_mlgd_f.png"))

    for folder, photos in [("photos_regular", regular_eval_photos),
                            ("photos_mlgd_f",  mlgd_f_eval_photos)]:
        photo_dir = os.path.join(args.output_dir, folder)
        os.makedirs(photo_dir, exist_ok=True)
        for idx, photo in enumerate(photos):
            photo.save(os.path.join(photo_dir, f"photo_{idx:03d}.png"))

    # ── 12. wandb final logs ────────────────────────────────────────────────
    wandb.log({
        "final_mlgd_f_mmd":         mlgd_f_mmd,
        "final_regular_mmd":        regular_mmd,
        "mmd_delta":                regular_mmd - mlgd_f_mmd,
        "mmd_relative_improvement": (regular_mmd - mlgd_f_mmd) / (regular_mmd + 1e-8),
        "final_scribble_mlgd_f":    wandb.Image(final_mlgd_f_pil),
        "final_scribble_regular":   wandb.Image(final_regular_pil),
        "mlgd_f_eval_photos":       [wandb.Image(p) for p in mlgd_f_eval_photos],
        "regular_eval_photos":      [wandb.Image(p) for p in regular_eval_photos],
        "scribble_heatmap":         wandb.Image(heatmap_path),
    })
    wandb.summary["final_mlgd_f_mmd"]  = mlgd_f_mmd
    wandb.summary["final_regular_mmd"] = regular_mmd
    wandb.summary["mmd_delta"]         = regular_mmd - mlgd_f_mmd
    wandb.summary["final_grad_norm"]   = step_gradients[-1]["gradient_norm"]

    # ── Save numpy arrays ───────────────────────────────────────────────────
    npy_dir = os.path.join(args.output_dir, "npy")
    os.makedirs(npy_dir, exist_ok=True)

    save_image_list_npy(mlgd_f_eval_photos,  os.path.join(npy_dir, "photos_mlgd_f.npy"))
    save_image_list_npy(regular_eval_photos, os.path.join(npy_dir, "photos_regular.npy"))
    for name, imgs in target_images_per_group.items():
        safe_name = name.lower().replace(" ", "_")
        save_image_list_npy(imgs, os.path.join(npy_dir, f"targets_{safe_name}.npy"))
    save_image_list_npy([source_image],      os.path.join(npy_dir, "source_portrait.npy"))
    save_image_list_npy([scribble_pil],      os.path.join(npy_dir, "scribble.npy"))
    save_image_list_npy([final_mlgd_f_pil],  os.path.join(npy_dir, "final_scribble_mlgd_f.npy"))
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
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({
            "args":  vars(args),
            "steps": step_gradients,
            "final_mlgd_f_mmd":       mlgd_f_mmd,
            "final_regular_mmd":      regular_mmd,
            "mmd_delta":              regular_mmd - mlgd_f_mmd,
            "final_mlgd_f_swd":       swd_mlgd_f,   # None — run analysis.py to compute
            "final_regular_swd":      swd_regular,  # None — run analysis.py to compute
            "swd_delta":              None,         # computed in analysis.py
            "optimization_time_sec":  optimization_time_sec,
            "mlgd_f_gender":          mlgd_f_stats,
            "regular_gender":         regular_stats,
            "npy": {
                "photos_mlgd_f":          "npy/photos_mlgd_f.npy",
                "photos_regular":         "npy/photos_regular.npy",
                **{f"targets_{n.lower().replace(' ','_')}": f"npy/targets_{n.lower().replace(' ','_')}.npy"
                   for n in target_images_per_group},
                "source_portrait":        "npy/source_portrait.npy",
                "scribble":               "npy/scribble.npy",
                "final_scribble_mlgd_f":  "npy/final_scribble_mlgd_f.npy",
                "final_scribble_regular": "npy/final_scribble_regular.npy",
                # clip embeddings computed by analysis.py (run offline)
            },
        }, f, indent=2)
    print(f"✅ metrics.json saved.  Optimization time: {optimization_time_sec/60:.1f} min",
          flush=True)

    wandb.finish()
    print(f"\n✅ All outputs saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
