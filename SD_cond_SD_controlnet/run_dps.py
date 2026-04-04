"""
DPS CLIP-MMD Pipeline — Script version of main.ipynb (scribble_cond_loss branch).

Mirrors the notebook exactly:
  - SDEdit-style init (encode HED scribble → latent, noise to start_step)
  - HED scribble conditioning
  - CLIP-MMD DPS guidance with variation_prompt
  - Intermediate MMD evaluation every eval_interval steps
  - Final MMD evaluation for both DPS and regular paths
  - Full wandb logging (input images, step visualizations, final eval photos, PCA)
"""

import argparse
import copy
import gc
import json
import time
import numpy as np
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA

# Ensure SD_cond_SD_controlnet/ is on the path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from clip_utils import encode_images_clip, load_clip_model
from generation import (
    compute_pred_x0_direct,
    denoise_step,
    generate_and_store_cs,
    predict_noise_cfg,
    run_dps_step_clip,
)
from image_utils import build_base_image, latent_to_pil, sobel_proxy
from metrics import compute_mmd, evaluate_distribution_mmd, compute_swd
from models import load_models, setup_gradient_checkpointing
from visualization import plot_row, visualize_step
from analysis import compare_scribbles_heatmap
LOSS_FNS = {"mmd": compute_mmd, "swd": compute_swd}

def parse_args():
    p = argparse.ArgumentParser(description="DPS CLIP-MMD Pipeline (main.ipynb script version)")
    p.add_argument("--output_dir", type=str, default="SD_cond_SD_controlnet/output/dps_run")
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--architect_unet_path", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="measure_MMD_between_uncond_dps")
    p.add_argument("--wandb_entity", type=str, default="conditional-matching")

    # Scheduler / loop
    p.add_argument("--n_steps", type=int, default=30)
    p.add_argument("--start_step", type=int, default=15,
                   help="SDEdit start step — DPS runs from here to n_steps")

    # Guidance
    p.add_argument("--base_zeta", type=float, default=1.0)
    p.add_argument("--guidance_scale", type=float, default=0.0,
                   help="CFG scale for architect (0.0 = unconditional)")
    p.add_argument("--controlnet_scale", type=float, default=0.5)
    p.add_argument("--loss_fn", type=str, default="mmd", choices=["mmd", "swd"])
    p.add_argument("--bandwidth_scale", type=float, default=1.0,
                   help="Scale factor for MMD bandwidth (< 1 = sharper kernel)")
    p.add_argument("--loss_scale", type=float, default=1.0,
                   help="Multiply loss by this factor before grad computation"),
    p.add_argument("--kernel_alpha", type=float, default=1.0,
                   help="Generalized RBF exponent. >1 = sharper falloff, penalizes inter-mode points more.")

    # Variations / eval
    p.add_argument("--num_variations", type=int, default=6)
    p.add_argument("--n_targets", type=int, default=20,
                   help="Total target images (split evenly man/woman)")
    p.add_argument("--n_eval", type=int, default=10,
                   help="Sprinter photos per MMD evaluation")
    p.add_argument("--eval_interval", type=int, default=0,
                   help="Evaluate intermediate MMD every N steps (0 = auto ~5 checkpoints)")

    # Prompts
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--sprinter_variation_prompt", type=str,
                   default="a superrealistic professional photograph of")
    p.add_argument("--sprinter_target_man_prompt", type=str,
                   default="a superrealistic portrait photograph of a man, studio lighting")
    p.add_argument("--sprinter_target_woman_prompt", type=str,
                   default="a superrealistic portrait photograph of a woman, studio lighting")
    p.add_argument("--sprinter_eval_prompt", type=str,
                   default="a superrealistic professional photograph of")

    # Models
    p.add_argument("--controlnet_model_id", type=str,
                   default="xinsir/controlnet-scribble-sdxl-1.0")
    p.add_argument("--sprinter_model_id", type=str, default="stabilityai/sdxl-turbo")
    p.add_argument("--architect_model_id", type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0")

    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def compute_clip_softmax(pil_list, clip_model, clip_processor,
                         man_prompt, woman_prompt, device):
    """
    For each PIL image compute softmax probability over [man_prompt, woman_prompt].
    Returns a list of dicts: [{"p_male": float, "p_female": float, "label": str}, ...]
    label = "male" if p_male > 0.5 else "female"
    """
    import torch.nn.functional as F

    # encode text prompts once
    text_inputs = clip_processor(
        text=[man_prompt, woman_prompt],
        return_tensors="pt",
        padding=True,
    ).to(device)

    clip_model.to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    results = []
    # encode images in batches of 8
    batch_size = 8
    all_image_features = []
    for start in range(0, len(pil_list), batch_size):
        batch = pil_list[start:start + batch_size]
        tensors = torch.cat(
            [TF.to_tensor(img).unsqueeze(0) for img in batch], dim=0
        ).to(device)
        with torch.no_grad():
            img_features = encode_images_clip(tensors, clip_model, clip_processor)
        all_image_features.append(img_features)

    image_features = torch.cat(all_image_features, dim=0)  # [N, 768]

    # cosine similarity → softmax
    logits = (image_features @ text_features.T) * 100.0  # scale like CLIP
    probs = F.softmax(logits, dim=-1).cpu().numpy()       # [N, 2]

    for p_male, p_female in probs:
        results.append({
            "p_male":   float(p_male),
            "p_female": float(p_female),
            "label":    "male" if p_male > 0.5 else "female",
        })

    clip_model.to("cpu")
    return results, image_features.cpu().numpy()  # also return embeddings

def pil_images_to_tensor(pil_list, device):
    tensors = [TF.to_tensor(img).unsqueeze(0) for img in pil_list]
    return torch.cat(tensors, dim=0).to(device)
def save_image_list_npy(pil_list, path):
    """Save a list of PIL images as [N,H,W,3] uint8 numpy array."""
    arr = np.stack([np.array(img) for img in pil_list], axis=0)
    np.save(path, arr)

def extract_scribble_hed(pil_image):
    from controlnet_aux import HEDdetector
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
    return hed(pil_image, scribble=True)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    steps_dir = os.path.join(args.output_dir, "steps")
    os.makedirs(steps_dir, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # ── wandb ──────────────────────────────────────────────────────────────────
    import wandb

    # ── 1. Load models ─────────────────────────────────────────────────────────
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

    # ── 2. Base oval image + Sobel (used for initial target generation) ─────────
    base_image_pil, base_tensor = build_base_image(device)
    with torch.no_grad():
        sobel_cond_tensor = sobel_proxy(base_tensor, device)
        sobel_cond_pil = T.ToPILImage()(sobel_cond_tensor.squeeze(0).cpu())

    # ── 3. Generate initial target distributions ────────────────────────────────
    N = args.n_targets
    n_half = N // 2
    print(f"Generating {N} initial target images ({n_half} man + {n_half} woman)...", flush=True)

    with torch.no_grad():
        man_images_init, _ = generate_and_store_cs(
            sprinter, args.sprinter_target_man_prompt,
            sobel_cond_pil, n_half, batch_size=2, cn_scale=args.controlnet_scale,
        )
        woman_images_init, _ = generate_and_store_cs(
            sprinter, args.sprinter_target_woman_prompt,
            sobel_cond_pil, n_half, batch_size=2, cn_scale=args.controlnet_scale,
        )

    # ── 4. Extract HED scribble from one of the generated portraits ─────────────
    print("Extracting HED scribble...", flush=True)
    source_image = man_images_init[2]
    scribble_pil = extract_scribble_hed(source_image)
    source_image.save(os.path.join(args.output_dir, "source_portrait.png"))
    scribble_pil.save(os.path.join(args.output_dir, "scribble.png"))

    # From here on, use scribble_pil as the controlnet conditioning
    sobel_cond_pil = scribble_pil
    print("✅ HED scribble ready.", flush=True)

    # ── 5. Regenerate targets conditioned on the HED scribble ──────────────────
    print(f"Regenerating {N} target images conditioned on HED scribble...", flush=True)
    with torch.no_grad():
        man_images, _ = generate_and_store_cs(
            sprinter, args.sprinter_target_man_prompt,
            sobel_cond_pil, n_half, batch_size=2, cn_scale=args.controlnet_scale,
        )
        woman_images, _ = generate_and_store_cs(
            sprinter, args.sprinter_target_woman_prompt,
            sobel_cond_pil, n_half, batch_size=2, cn_scale=args.controlnet_scale,
        )

    plot_row(man_images, "Man Portrait Samples",
             save_path=os.path.join(args.output_dir, "target_samples_man.png"))
    plot_row(woman_images, "Woman Portrait Samples",
             save_path=os.path.join(args.output_dir, "target_samples_woman.png"))

    # ── 6. Encode targets to CLIP ──────────────────────────────────────────────
    print("Encoding targets to CLIP...", flush=True)
    with torch.no_grad():
        man_clip_embs = encode_images_clip(
            pil_images_to_tensor(man_images, device), clip_model, clip_processor)
        woman_clip_embs = encode_images_clip(
            pil_images_to_tensor(woman_images, device), clip_model, clip_processor)
    all_clip_embeddings = torch.cat([man_clip_embs, woman_clip_embs], dim=0)
    print(f"Target CLIP embeddings: {all_clip_embeddings.shape}", flush=True)

    # Sanity checks
    norms = all_clip_embeddings.norm(dim=-1)
    intra_sim = (man_clip_embs @ man_clip_embs.T).mean().item()
    inter_sim = (man_clip_embs @ woman_clip_embs.T).mean().item()
    print(f"  Norms min/max: {norms.min():.4f} / {norms.max():.4f}")
    print(f"  Intra-class sim (man↔man):   {intra_sim:.4f}")
    print(f"  Inter-class sim (man↔woman): {inter_sim:.4f}")
    assert intra_sim > inter_sim, "Classes not separated in CLIP space!"
    print("✅ Classes separable in CLIP space.", flush=True)

    # Target CLIP PCA
    _pca = PCA(n_components=2)
    _coords = _pca.fit_transform(all_clip_embeddings.cpu().numpy())
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(_coords[:n_half, 0], _coords[:n_half, 1], c='dodgerblue', label='Man', alpha=0.7)
    ax.scatter(_coords[n_half:, 0], _coords[n_half:, 1], c='crimson', label='Woman', alpha=0.7)
    ax.set_title("PCA of Target CLIP Embeddings"); ax.legend(); ax.grid(True, alpha=0.3)
    pca_path = os.path.join(args.output_dir, "target_clip_pca.png")
    fig.savefig(pca_path, dpi=100, bbox_inches='tight'); plt.close(fig)
    pca_fixed = _pca
    del _coords

    # ── 7. wandb init (after we have config details) ───────────────────────────
    eval_interval = args.eval_interval if args.eval_interval > 0 else max(1, (args.n_steps - args.start_step) // 5)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "prompt":                       args.prompt,
            "negative_prompt":              args.negative_prompt,
            "n_targets":                    N,
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
            "n_eval":                       args.n_eval,
            "eval_interval":                eval_interval,
            "lora_path":                    args.lora_path,
            "architect_unet_path":          args.architect_unet_path,
            "architect_model":              args.architect_model_id,
            "sprinter_model":               args.sprinter_model_id,
            "sprinter_variation_prompt":    args.sprinter_variation_prompt,
            "sprinter_target_man_prompt":   args.sprinter_target_man_prompt,
            "sprinter_target_woman_prompt": args.sprinter_target_woman_prompt,
            "sprinter_eval_prompt":         args.sprinter_eval_prompt,
            "loss_fn":              args.loss_fn,
            "loss_scale":           args.loss_scale,
            "bandwidth_scale":      args.bandwidth_scale,
            "kernel_alpha": args.kernel_alpha,
        },
    )
    print(f"✅ wandb run: {run.name}", flush=True)

    # Log input images
    wandb.log({
        "scribble":             wandb.Image(scribble_pil),
        "source_portrait":      wandb.Image(source_image),
        "target_samples_man":   [wandb.Image(p) for p in man_images],
        "target_samples_woman": [wandb.Image(p) for p in woman_images],
        "target_clip_pca":      wandb.Image(pca_path),
    })
    print("✅ Input images logged to wandb.", flush=True)

    # ── 8. Prepare DPS ─────────────────────────────────────────────────────────
    height, width = 512, 512
    n_steps = args.n_steps
    start_step = args.start_step
    prompt = args.prompt
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
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=device,
            do_classifier_free_guidance=True,
            num_images_per_prompt=1,
        )

    architect.scheduler.set_timesteps(n_steps, device=device)
    timesteps = architect.scheduler.timesteps

    scheduler_regular = copy.deepcopy(architect.scheduler)

    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]], dtype=prompt_embeds.dtype, device=device
    )
    added_cond_kwargs = {
        "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0),
        "time_ids":    add_time_ids.repeat(2, 1),
    }
    cfg_encoder_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # ── SDEdit-style init: encode HED scribble → latent, noise to start_step ──
    with torch.no_grad():
        scribble_tensor = TF.to_tensor(scribble_pil).unsqueeze(0).to(device).to(torch.float32)
        scribble_tensor = (scribble_tensor * 2.0) - 1.0  # [0,1] → [-1,1]
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
    print(f"   Running {len(timesteps_to_run)} DPS steps...", flush=True)

    step_gradients = []
    step_vis_data  = []
    target_clip_np = all_clip_embeddings.cpu().numpy()
    from functools import partial
    if args.loss_fn == "mmd":
        loss_fn = partial(compute_mmd, bandwidth_scale=args.bandwidth_scale,kernel_alpha=args.kernel_alpha)
    else:
        loss_fn = LOSS_FNS[args.loss_fn]
    # ── Baseline visualization (step 0, before any DPS correction) ────────────
    with torch.no_grad():
        baseline_noise_pred = predict_noise_cfg(
            architect.unet, architect.scheduler,
            latents.detach(), timesteps_to_run[0],
            cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
        )
        baseline_pred_x0 = compute_pred_x0_direct(
            architect.scheduler, baseline_noise_pred, timesteps_to_run[0], latents.detach()
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
                num_inference_steps=2, guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.controlnet_scale, output_type="pil",

            ).images[0]

            for _ in range(args.n_eval)
        ]
        sprinter.vae.to(dtype=torch.float32)

        var_tensors = torch.cat([TF.to_tensor(img).unsqueeze(0) for img in baseline_var_images], dim=0).to(device)
        clip_model.to(device)
        baseline_clip_flat = encode_images_clip(var_tensors, clip_model, clip_processor).cpu().numpy()
        clip_model.to("cpu")

        sd_baseline = {
            "step": 0,
            "timestep": timesteps_to_run[0].item(),
            "mmd_loss": 0.0,
            "zeta_i": 0.0,
            "latents_step_cpu": latents.detach().cpu(),
            "latents_step_regular_cpu": latents.detach().cpu(),
            "pred_x0_cpu": baseline_pred_x0.detach().cpu(),
            "pred_x0_regular_cpu": baseline_pred_x0.detach().cpu(),
            "variation_clip_flat": baseline_clip_flat,
        }

    baseline_save_path = os.path.join(steps_dir, "step_baseline.png")
    visualize_step(sd_baseline, architect, sprinter, target_clip_np,
                   num_cond=4, save_path=baseline_save_path, pca_fixed=pca_fixed)
    print("✅ Baseline visualization saved.", flush=True)

    # ── 9. DPS loop ────────────────────────────────────────────────────────────
    dps_start_time = time.time()
    for i, t in enumerate(timesteps_to_run):
        print(f"\n{'='*60}", flush=True)
        print(f"Step {i+1}/{len(timesteps_to_run)}  (t={t})", flush=True)
        print(f"{'='*60}", flush=True)

        latents_step         = latents.detach().requires_grad_(True)
        latents_step_regular = latents_regular.detach()

        # Noise prediction
        noise_pred = predict_noise_cfg(
            architect.unet, architect.scheduler,
            latents_step, t, cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
        )
        with torch.no_grad():
            noise_pred_regular = predict_noise_cfg(
                architect.unet, scheduler_regular,
                latents_step_regular, t, cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
            )

        # pred_x0 — pure formula, no scheduler.step() side effects
        pred_x0 = compute_pred_x0_direct(architect.scheduler, noise_pred, t, latents_step)
        with torch.no_grad():
            pred_x0_regular = compute_pred_x0_direct(
                scheduler_regular, noise_pred_regular, t, latents_step_regular)

        # Decode pred_x0 → pixel space (keep grad)
        pred_x0_scaled = pred_x0 / architect.vae.config.scaling_factor

        def vae_decode_checkpoint(lat):
            return architect.vae.decode(lat.to(architect.vae.dtype)).sample

        pixel_x0 = torch.utils.checkpoint.checkpoint(
            vae_decode_checkpoint, pred_x0_scaled, use_reentrant=False)
        pixel_x0_norm = torch.clamp((pixel_x0 + 1.0) / 2.0, 0.0, 1.0)

        # CLIP-MMD + gradient
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
            "step":            i+1,
            "timestep":        t.item(),
            "gradient_norm":   grad_norm,
            "mmd_loss":        mmd_loss.item(),
            "zeta_i":          zeta_val,
            "loss_norm":       loss_norm.item(),
            "correction_norm": zeta_val * grad_norm,
        })

        wandb_log = {
            "step":            i+1,
            "mmd_loss":        mmd_loss.item(),
            "gradient_norm":   grad_norm,
            "zeta":            zeta_val,
            "correction_norm": zeta_val * grad_norm,
        }

        # Intermediate MMD evaluation
        if i % eval_interval == 0:
            unguided_mmd, _, _ = evaluate_distribution_mmd(
                pred_x0_regular.detach(), architect.vae, architect.image_processor,
                sprinter, clip_model, clip_processor,
                all_clip_embeddings, args.sprinter_eval_prompt,
                n_eval=args.n_eval, device=device,
            )
            wandb_log["intermediate/unguided_cond_mmd"] = unguided_mmd
            wandb_log["intermediate/guided_cond_mmd"]   = mmd_loss.item()
            wandb_log["intermediate/cond_mmd_delta"]    = mmd_loss.item() - unguided_mmd
            print(f"  [eval] guided={mmd_loss.item():.6f}  "
                  f"unguided={unguided_mmd:.6f}  "
                  f"delta={mmd_loss.item()-unguided_mmd:.6f}", flush=True)

        wandb.log(wandb_log, commit=False)

        # Store step data for visualization
        with torch.no_grad():
            sd = {
                "step":                     i+1,
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

        step_save_path = os.path.join(steps_dir, f"step_{i:03d}.png")
        visualize_step(sd, architect, sprinter, target_clip_np,
                       num_cond=5, save_path=step_save_path, pca_fixed=pca_fixed)

        # Scheduler step
        latents = denoise_step(architect.scheduler, noise_pred, t, latents_step,
                               correction=correction)
        with torch.no_grad():
            latents_regular = denoise_step(scheduler_regular, noise_pred_regular,
                                           t, latents_step_regular)

        # Cleanup
        del grad, mmd_loss, loss_norm, zeta_i, correction
        del pixel_x0, pixel_x0_norm, pred_x0, pred_x0_regular
        del latents_step_regular, noise_pred_regular
        gc.collect(); torch.cuda.empty_cache()

    del latents_step, noise_pred
    torch.cuda.empty_cache()
    print(f"\n✅ CLIP-MMD DPS Complete! {len(step_vis_data)} steps stored.", flush=True)

    # ── 10. Final MMD evaluation ───────────────────────────────────────────────
    print("Computing final MMD (regular)...", flush=True)
    regular_mmd, regular_eval_photos, _ = evaluate_distribution_mmd(
        latents_regular, architect.vae, architect.image_processor,
        sprinter, clip_model, clip_processor,
        all_clip_embeddings, eval_prompt=args.sprinter_eval_prompt,
        n_eval=args.n_eval, device=device,
    )

    print("Computing final MMD (DPS)...", flush=True)
    dps_mmd, dps_eval_photos, _ = evaluate_distribution_mmd(
        latents, architect.vae, architect.image_processor,
        sprinter, clip_model, clip_processor,
        all_clip_embeddings, eval_prompt=args.sprinter_eval_prompt,
        n_eval=args.n_eval, device=device,
    )

    print(f"Regular MMD : {regular_mmd:.6f}", flush=True)
    print(f"DPS MMD     : {dps_mmd:.6f}", flush=True)
    print(f"Delta (↓ better for DPS): {regular_mmd - dps_mmd:.6f}", flush=True)

    # ── 11. Final visualizations ───────────────────────────────────────────────
    with torch.no_grad():
        final_dps_pil     = latent_to_pil(latents,         architect.vae, architect.image_processor)
        final_regular_pil = latent_to_pil(latents_regular, architect.vae, architect.image_processor)

    final_dps_pil.save(os.path.join(args.output_dir, "final_scribble_lgd_cm.png"))
    final_regular_pil.save(os.path.join(args.output_dir, "final_scribble_regular.png"))
    heatmap_path = os.path.join(args.output_dir, "scribble_heatmap.png")
    compare_scribbles_heatmap(final_dps_pil, final_regular_pil, save_path=heatmap_path)
    print("✅ Scribble heatmap saved.", flush=True)
    # Save eval photo rows
    plot_row(regular_eval_photos, f"Regular final photos  (MMD={regular_mmd:.4f})",
             save_path=os.path.join(args.output_dir, "final_photos_regular.png"))
    plot_row(dps_eval_photos, f"LGD-CM final photos   (MMD={dps_mmd:.4f})",
             save_path=os.path.join(args.output_dir, "final_photos_lgd_cm.png"))

    # Save individual photos for downstream evaluation (e.g. gender classifier)
    for folder, photos in [("photos_regular", regular_eval_photos),
                           ("photos_lgd_cm", dps_eval_photos)]:
        photo_dir = os.path.join(args.output_dir, folder)
        os.makedirs(photo_dir, exist_ok=True)
        for idx, photo in enumerate(photos):
            photo.save(os.path.join(photo_dir, f"photo_{idx:03d}.png"))






    # ── 12. wandb final logs ───────────────────────────────────────────────────
    wandb.log({
        "final_dps_mmd":            dps_mmd,
        "final_regular_mmd":        regular_mmd,
        "mmd_delta":                regular_mmd - dps_mmd,
        "mmd_relative_improvement": (regular_mmd - dps_mmd) / (regular_mmd + 1e-8),
        "final_scribble_lgd_cm":    wandb.Image(final_dps_pil),
        "final_scribble_regular":   wandb.Image(final_regular_pil),
        "lgd_cm_eval_photos":       [wandb.Image(p) for p in dps_eval_photos],
        "regular_eval_photos":      [wandb.Image(p) for p in regular_eval_photos],
        "scribble_heatmap": wandb.Image(heatmap_path),
    })

    wandb.summary["final_dps_mmd"]            = dps_mmd
    wandb.summary["final_regular_mmd"]        = regular_mmd
    wandb.summary["mmd_delta"]                = regular_mmd - dps_mmd
    wandb.summary["final_grad_norm"]          = step_gradients[-1]["gradient_norm"]

    # ── Save image arrays (.npy) ───────────────────────────────────────
    npy_dir = os.path.join(args.output_dir, "npy")
    os.makedirs(npy_dir, exist_ok=True)

    save_image_list_npy(dps_eval_photos,
                        os.path.join(npy_dir, "photos_lgd_cm.npy"))
    save_image_list_npy(regular_eval_photos,
                        os.path.join(npy_dir, "photos_regular.npy"))
    save_image_list_npy(man_images,
                        os.path.join(npy_dir, "targets_man.npy"))
    save_image_list_npy(woman_images,
                        os.path.join(npy_dir, "targets_woman.npy"))
    save_image_list_npy([source_image],
                        os.path.join(npy_dir, "source_portrait.npy"))
    save_image_list_npy([scribble_pil],
                        os.path.join(npy_dir, "scribble.npy"))
    print("✅ Image arrays saved to npy/", flush=True)

    # save individual target portraits
    for folder, photos in [("targets_man", man_images),
                           ("targets_woman", woman_images)]:
        photo_dir = os.path.join(args.output_dir, folder)
        os.makedirs(photo_dir, exist_ok=True)
        for idx, photo in enumerate(photos):
            photo.save(os.path.join(photo_dir, f"photo_{idx:03d}.png"))
    print("✅ Individual target portraits saved.", flush=True)

    # ── CLIP softmax probabilities ─────────────────────────────────────
    print("Computing CLIP softmax probabilities...", flush=True)
    clip_model.to(device)

    lgd_cm_softmax, lgd_cm_clip_embs = compute_clip_softmax(
        dps_eval_photos, clip_model, clip_processor,
        args.sprinter_target_man_prompt,
        args.sprinter_target_woman_prompt,
        device,
    )
    regular_softmax, regular_clip_embs = compute_clip_softmax(
        regular_eval_photos, clip_model, clip_processor,
        args.sprinter_target_man_prompt,
        args.sprinter_target_woman_prompt,
        device,
    )
    target_softmax, target_clip_embs = compute_clip_softmax(
        man_images + woman_images, clip_model, clip_processor,
        args.sprinter_target_man_prompt,
        args.sprinter_target_woman_prompt,
        device,
    )
    clip_model.to("cpu")
    print("✅ CLIP softmax done.", flush=True)

    # save CLIP embeddings as .npy
    np.save(os.path.join(npy_dir, "clip_lgd_cm.npy"), lgd_cm_clip_embs)
    np.save(os.path.join(npy_dir, "clip_regular.npy"), regular_clip_embs)
    np.save(os.path.join(npy_dir, "clip_targets.npy"), target_clip_embs)
    np.save(os.path.join(npy_dir, "clip_targets_man.npy"),
            target_clip_embs[:len(man_images)])
    np.save(os.path.join(npy_dir, "clip_targets_woman.npy"),
            target_clip_embs[len(man_images):])
    print("✅ CLIP embeddings saved to npy/", flush=True)

    # ── Gender counts + mean confidence ───────────────────────────────
    def gender_stats(softmax_list):
        males = [x for x in softmax_list if x["label"] == "male"]
        females = [x for x in softmax_list if x["label"] == "female"]
        return {
            "n_male": len(males),
            "n_female": len(females),
            "mean_conf_male": float(np.mean([x["p_male"] for x in males]))
            if males else None,
            "mean_conf_female": float(np.mean([x["p_female"] for x in females]))
            if females else None,
            "per_image": softmax_list,
        }

    lgd_cm_stats = gender_stats(lgd_cm_softmax)
    regular_stats = gender_stats(regular_softmax)

    print(f"  LGD-CM  : {lgd_cm_stats['n_male']} male, "
          f"{lgd_cm_stats['n_female']} female", flush=True)
    print(f"  Regular : {regular_stats['n_male']} male, "
          f"{regular_stats['n_female']} female", flush=True)

    # ── SWD to target ──────────────────────────────────────────────────
    print("Computing SWD to target...", flush=True)
    lgd_cm_clip_t = torch.from_numpy(lgd_cm_clip_embs).float()
    regular_clip_t = torch.from_numpy(regular_clip_embs).float()
    target_clip_t = torch.from_numpy(target_clip_embs).float()

    with torch.no_grad():
        swd_lgd_cm = compute_swd(lgd_cm_clip_t, target_clip_t).item()
        swd_regular = compute_swd(regular_clip_t, target_clip_t).item()
    print(f"  SWD LGD-CM : {swd_lgd_cm:.6f}", flush=True)
    print(f"  SWD Regular: {swd_regular:.6f}", flush=True)

    # ── Timing ────────────────────────────────────────────────────────
    dps_end_time = time.time()
    optimization_time_sec = dps_end_time - dps_start_time

    # ── Save enriched metrics.json ─────────────────────────────────────
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            # run config
            "args": vars(args),

            # per-step DPS metrics
            "steps": step_gradients,

            # final MMD
            "final_lgd_cm_mmd": dps_mmd,
            "final_regular_mmd": regular_mmd,
            "mmd_delta": regular_mmd - dps_mmd,

            # final SWD
            "final_lgd_cm_swd": swd_lgd_cm,
            "final_regular_swd": swd_regular,
            "swd_delta": swd_regular - swd_lgd_cm,

            # timing
            "optimization_time_sec": optimization_time_sec,

            # gender stats (counts + mean confidence + per-image probs)
            "lgd_cm_gender": lgd_cm_stats,
            "regular_gender": regular_stats,

            # paths to saved arrays (relative to output_dir)
            "npy": {
                "photos_lgd_cm": "npy/photos_lgd_cm.npy",
                "photos_regular": "npy/photos_regular.npy",
                "targets_man": "npy/targets_man.npy",
                "targets_woman": "npy/targets_woman.npy",
                "source_portrait": "npy/source_portrait.npy",
                "scribble": "npy/scribble.npy",
                "clip_lgd_cm": "npy/clip_lgd_cm.npy",
                "clip_regular": "npy/clip_regular.npy",
                "clip_targets": "npy/clip_targets.npy",
                "clip_targets_man": "npy/clip_targets_man.npy",
                "clip_targets_woman": "npy/clip_targets_woman.npy",
            },
        }, f, indent=2)
    print(f"✅ metrics.json saved.", flush=True)
    print(f"   Optimization time: {optimization_time_sec / 60:.1f} min", flush=True)

    wandb.finish()
    print(f"\n✅ All outputs saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
