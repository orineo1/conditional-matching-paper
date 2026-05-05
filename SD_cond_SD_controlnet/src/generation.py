"""
generation.py — noise prediction, pred_x0, DPS gradient step (CLIP-MMD).

Key design (matches working notebook + dps_loop.py / run_gender.py / run_age.py):
  - run_dps_step_clip accepts loss_fn and loss_scale (required by CLI scripts)
  - controlnet_conditioning_scale is a parameter, not hardcoded
  - Gradient flows through ALL variations: sprinter → VAE decode → CLIP → MMD
  - Debug prints are clean, structured, and always-on (strip later if needed)
"""

import gc

import numpy as np
import torch
import torch.utils.checkpoint


# ── Sprinter generation helpers ───────────────────────────────────────────────

def generate_and_store(pipe, prompt, sobel_cond_pil, num_samples, batch_size=2):
    original_vae_dtype = pipe.vae.dtype
    pipe.vae.to(dtype=torch.float16)
    all_images, all_lats = [], []

    def latents_callback(p, step_index, timestep, cb_kwargs):
        if step_index == p.num_timesteps - 1:
            p._current_latents = cb_kwargs["latents"].detach().cpu().numpy()
        return cb_kwargs

    for i in range(0, num_samples, batch_size):
        curr = min(batch_size, num_samples - i)
        result = pipe(
            prompt=[prompt] * curr,
            image=[sobel_cond_pil] * curr,
            num_inference_steps=2,
            guidance_scale=0.0,
            controlnet_conditioning_scale=1.0,
            callback_on_step_end=latents_callback,
        )
        all_images.extend(result.images)
        all_lats.append(pipe._current_latents.reshape(curr, -1))
        print(f"  Progress: {len(all_images)}/{num_samples}", end="\r", flush=True)

    print(flush=True)
    pipe.vae.to(dtype=original_vae_dtype)
    return all_images, np.vstack(all_lats)


def generate_and_store_cs(pipe, prompt, cond_pil, num_samples,
                          batch_size=2, cn_scale=0.5):
    """generate_and_store with configurable controlnet_conditioning_scale."""
    original_vae_dtype = pipe.vae.dtype
    pipe.vae.to(dtype=torch.float16)
    all_images, all_lats = [], []

    def latents_callback(p, step_index, timestep, cb_kwargs):
        if step_index == p.num_timesteps - 1:
            p._current_latents = cb_kwargs["latents"].detach().cpu().numpy()
        return cb_kwargs

    for i in range(0, num_samples, batch_size):
        curr = min(batch_size, num_samples - i)
        result = pipe(
            prompt=[prompt] * curr,
            image=[cond_pil] * curr,
            num_inference_steps=2,
            guidance_scale=0.0,
            controlnet_conditioning_scale=cn_scale,
            callback_on_step_end=latents_callback,
        )
        all_images.extend(result.images)
        all_lats.append(pipe._current_latents.reshape(curr, -1))
        print(f"  Progress: {len(all_images)}/{num_samples}", end="\r", flush=True)

    print(flush=True)
    pipe.vae.to(dtype=original_vae_dtype)
    return all_images, np.vstack(all_lats)


# ── Noise prediction & pred_x0 ────────────────────────────────────────────────

def predict_noise_cfg(unet, scheduler, latents_in, t,
                      encoder_states, added_cond, gs):
    lmi = scheduler.scale_model_input(torch.cat([latents_in] * 2), t)
    np_out = unet(
        lmi, t,
        encoder_hidden_states=encoder_states,
        added_cond_kwargs=added_cond,
        return_dict=False,
    )[0]
    np_u, np_t = np_out.chunk(2)
    return np_u + gs * (np_t - np_u)


def compute_pred_x0(scheduler, noise_pred, t, latents_in):
    so = scheduler.step(noise_pred, t, latents_in, return_dict=True)
    if hasattr(so, "pred_original_sample") and so.pred_original_sample is not None:
        return so.pred_original_sample
    alpha = scheduler.alphas_cumprod[t.long().cpu()].to(latents_in.device)
    return (latents_in - (1 - alpha) ** 0.5 * noise_pred) / alpha ** 0.5


def compute_pred_x0_direct(scheduler, noise_pred, t, latents_in):
    """Compute pred_x0 without calling scheduler.step() (avoids step_index side effects)."""
    if hasattr(scheduler, "alphas_cumprod"):
        alpha = scheduler.alphas_cumprod[t.long().cpu()].to(latents_in.device)
    else:
        sigma = scheduler.sigmas[scheduler.step_index]
        alpha = (1 / (sigma ** 2 + 1)).to(latents_in.device)
    return (latents_in - (1 - alpha) ** 0.5 * noise_pred) / alpha ** 0.5


def denoise_step(scheduler, noise_pred, t, latents_in, correction=None):
    x_t_minus_1 = scheduler.step(
        noise_pred, t, latents_in, return_dict=True
    ).prev_sample
    if correction is not None:
        x_t_minus_1 = x_t_minus_1 + correction
    return x_t_minus_1.detach()


# ── Legacy latent-space DPS step (kept for reference) ────────────────────────

def run_dps_step(latents, latents_step, noise_pred, pixel_x0_norm,
                 sprinter, all_latents, num_variations,
                 variation_batch_size, base_zeta_prime):
    from metrics import compute_mmd

    variation_latents_list = []
    print(f"  Generating {num_variations} latent variations...", flush=True)

    for start_idx in range(0, num_variations, variation_batch_size):
        end_idx = min(start_idx + variation_batch_size, num_variations)
        bs = end_idx - start_idx
        ctrl_batch = pixel_x0_norm[0].unsqueeze(0).repeat(bs, 1, 1, 1)

        def sprinter_forward(ctrl):
            return sprinter(
                prompt=["a superrealistic professional photograph of"] * ctrl.shape[0],
                image=ctrl,
                num_inference_steps=2,
                guidance_scale=0.0,
                controlnet_conditioning_scale=0.8,
                output_type="latent",
                return_dict=True,
            ).images

        with torch.cuda.amp.autocast():
            vl = torch.utils.checkpoint.checkpoint(
                sprinter_forward, ctrl_batch, use_reentrant=False
            )
        variation_latents_list.append(vl)

    variation_latents = torch.cat(variation_latents_list, dim=0)
    torch.cuda.empty_cache()

    mmd_squared = compute_mmd(
        variation_latents,
        torch.tensor(all_latents, device=variation_latents.device),
    )
    mmd_loss = torch.sqrt(mmd_squared.abs() + 1e-8)
    loss_norm = mmd_loss.detach()
    zeta_i = base_zeta_prime / loss_norm

    grad = torch.autograd.grad(
        mmd_loss, latents_step, retain_graph=False, create_graph=False
    )[0]
    vl_flat = variation_latents.detach().cpu().numpy().reshape(
        variation_latents.shape[0], -1
    )
    del variation_latents_list, variation_latents
    return grad, mmd_loss, zeta_i, loss_norm, vl_flat


# ── CLIP-MMD DPS step ─────────────────────────────────────────────────────────

def run_dps_step_clip(
    *,
    latents_step,            # [1, 4, H, W]  requires_grad=True
    pixel_x0_norm,           # [1, 3, H, W]  in [0,1], grad flows through this
    sprinter,
    all_clip_embeddings,     # [N, 768]  target embeddings (will be detached)
    num_variations,
    variation_batch_size,
    base_zeta_prime,
    clip_model,
    clip_processor,
    vae,                     # sprinter.vae
    vae_scaling_factor,
    variation_prompt,
    loss_fn,                 # callable(x, y) → scalar  (compute_mmd or compute_swd)
    loss_scale=1.0,          # multiply loss before grad (default 1.0)
    controlnet_scale=0.5,    # controlnet_conditioning_scale for sprinter
):
    """
    Compute the CLIP-MMD DPS gradient.

    Gradient path:
        latents_step
          → UNet (frozen) → noise_pred
          → diffusion formula → pred_x0
          → architect VAE decode → pixel_x0_norm          ← enters here
          → sprinter × num_variations (frozen)
          → sprinter VAE decode → pixels
          → CLIP encode (frozen) → [num_variations, 768]
          → loss_fn vs target embeddings
          → autograd.grad → gradient on latents_step

    Returns:
        grad            gradient w.r.t. latents_step
        loss_value      raw loss (before loss_scale)
        zeta_i          adaptive step size  base_zeta / loss_norm
        loss_norm       detached loss norm (for logging)
        vl_clip_flat    [num_variations, 768] numpy  (variation CLIP embeddings)
    """
    from src.clip_utils import encode_images_clip

    device = pixel_x0_norm.device
    clip_model.to(device)

    # ── DIAG A: graph entry-point ──────────────────────────────────────────────
    print(f"  [DIAG-A] pixel_x0_norm.requires_grad = {pixel_x0_norm.requires_grad}",
          flush=True)
    print(f"  [DIAG-A] pixel_x0_norm.dtype         = {pixel_x0_norm.dtype}",
          flush=True)
    print(f"  [DIAG-A] pixel_x0_norm.grad_fn       = {pixel_x0_norm.grad_fn}",
          flush=True)
    print(f"  [DIAG-A] latents_step.requires_grad  = {latents_step.requires_grad}",
          flush=True)

    # Quick end-to-end grad smoke-test through pixel_x0_norm (cheap, no sprinter)
    try:
        _test = pixel_x0_norm.float().sum()
        _tg = torch.autograd.grad(_test, latents_step, retain_graph=True)[0]
        print(f"  [DIAG-A] ✅ grad flows pixel_x0_norm → latents_step  "
              f"norm={_tg.norm().item():.6f}", flush=True)
        del _tg, _test
    except Exception as _e:
        print(f"  [DIAG-A] ❌ GRAD BROKEN at pixel_x0_norm → latents_step: {_e}",
              flush=True)
        print(f"           Check architect VAE cast to float32 before decode.",
              flush=True)

    # ── Generate variations: sprinter → VAE decode → CLIP ─────────────────────
    variation_clip_list = []

    for start_idx in range(0, num_variations, variation_batch_size):
        end_idx = min(start_idx + variation_batch_size, num_variations)
        bs = end_idx - start_idx
        ctrl_batch = pixel_x0_norm[0].unsqueeze(0).repeat(bs, 1, 1, 1)

        def sprinter_vae_clip_forward(ctrl):
            # Sprinter runs in no-grad (frozen); output latents are detached
            var_latents = sprinter(
                prompt=[variation_prompt] * ctrl.shape[0],
                image=ctrl,
                num_inference_steps=2,
                guidance_scale=0.0,
                controlnet_conditioning_scale=controlnet_scale,
                output_type="latent",
                return_dict=True,
            ).images  # [bs, 4, h, w]

            # ── DIAG B: sprinter output ────────────────────────────────────────
            print(f"    [DIAG-B] var_latents.dtype         = {var_latents.dtype}",
                  flush=True)
            print(f"    [DIAG-B] var_latents.requires_grad = {var_latents.requires_grad}",
                  flush=True)

            # VAE decode in float32 (no autocast) so grad can flow through ctrl
            with torch.amp.autocast("cuda", enabled=False):
                var_pixels_raw = vae.decode(
                    var_latents.float() / vae_scaling_factor
                ).sample

            # ── DIAG C: pixels after sprinter VAE ─────────────────────────────
            print(f"    [DIAG-C] var_pixels_raw.dtype         = {var_pixels_raw.dtype}",
                  flush=True)
            print(f"    [DIAG-C] var_pixels_raw.requires_grad = {var_pixels_raw.requires_grad}",
                  flush=True)
            print(f"    [DIAG-C] var_pixels_raw.grad_fn       = {var_pixels_raw.grad_fn}",
                  flush=True)

            var_pixels = torch.clamp((var_pixels_raw.float() + 1.0) / 2.0, 0.0, 1.0)
            clip_emb = encode_images_clip(var_pixels, clip_model, clip_processor)

            # ── DIAG D: CLIP embedding ─────────────────────────────────────────
            print(f"    [DIAG-D] clip_emb.requires_grad = {clip_emb.requires_grad}",
                  flush=True)
            print(f"    [DIAG-D] clip_emb.grad_fn       = {clip_emb.grad_fn}",
                  flush=True)
            print(f"    [DIAG-D] clip_emb NaN count     = {torch.isnan(clip_emb).sum().item()}",
                  flush=True)

            return clip_emb  # [bs, 768]  carries grad through ctrl → pixel_x0_norm

        # Gradient checkpoint: recompute forward on backward to save VRAM
        with torch.amp.autocast("cuda", enabled=False):
            var_clip = torch.utils.checkpoint.checkpoint(
                sprinter_vae_clip_forward, ctrl_batch, use_reentrant=False
            )

        variation_clip_list.append(var_clip)

    variation_clip_embs = torch.cat(variation_clip_list, dim=0)  # [num_variations, 768]
    torch.cuda.empty_cache()

    # ── DIAG E: MMD inputs ────────────────────────────────────────────────────
    print(f"  [DIAG-E] variation_clip_embs.shape         = {variation_clip_embs.shape}",
          flush=True)
    print(f"  [DIAG-E] variation_clip_embs.requires_grad = {variation_clip_embs.requires_grad}",
          flush=True)
    print(f"  [DIAG-E] variation_clip_embs.grad_fn       = {variation_clip_embs.grad_fn}",
          flush=True)
    print(f"  [DIAG-E] all_clip_embeddings.requires_grad = {all_clip_embeddings.requires_grad}",
          flush=True)

    # ── Compute loss ──────────────────────────────────────────────────────────
    # Target embeddings are always detached; grad only flows through generated side.
    loss_value = loss_fn(variation_clip_embs, all_clip_embeddings.detach())

    # ── DIAG F: loss ──────────────────────────────────────────────────────────
    print(f"  [DIAG-F] loss_value.item()        = {loss_value.item():.8f}", flush=True)
    print(f"  [DIAG-F] loss_scale               = {loss_scale}", flush=True)
    print(f"  [DIAG-F] loss_value.requires_grad = {loss_value.requires_grad}", flush=True)
    print(f"  [DIAG-F] loss_value.grad_fn       = {loss_value.grad_fn}", flush=True)

    # Apply loss_scale (amplifies or dampens gradient signal)
    scaled_loss = loss_value * loss_scale

    loss_norm = loss_value.detach()           # unscaled norm for adaptive zeta
    zeta_i = base_zeta_prime / (loss_norm + 1e-8)

    # ── Backprop ──────────────────────────────────────────────────────────────
    grad = torch.autograd.grad(
        scaled_loss, latents_step,
        retain_graph=False,
        create_graph=False,
    )[0]

    # ── DIAG G: gradient ──────────────────────────────────────────────────────
    print(f"  [DIAG-G] grad.norm()          = {grad.norm().item():.8f}", flush=True)
    print(f"  [DIAG-G] grad.isnan()         = {torch.isnan(grad).any().item()}", flush=True)
    print(f"  [DIAG-G] grad.abs().max()     = {grad.abs().max().item():.8f}", flush=True)
    print(f"  [DIAG-G] zeta_i               = {zeta_i.item():.6f}", flush=True)
    print(f"  [DIAG-G] correction_norm      = {(zeta_i * grad.norm()).item():.8f}",
          flush=True)

    vl_clip_flat = variation_clip_embs.detach().cpu().numpy()  # [num_variations, 768]
    del variation_clip_list, variation_clip_embs
    clip_model.to("cpu")

    return grad, loss_value, zeta_i, loss_norm, vl_clip_flat
