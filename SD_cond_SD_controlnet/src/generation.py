import gc

import numpy as np
import torch

from src.metrics import compute_mmd


def generate_and_store_cs(pipe, prompt, cond_pil, num_samples,
                           batch_size=2, cn_scale=0.5):
    """
    Generate num_samples images from the sprinter conditioned on cond_pil.

    cond_pil may be None for the initial unconditioned generation pass
    (before a real scribble is available). In that case a blank white image
    is used as a neutral ControlNet conditioning input.

    Returns: (pil_images, latents_numpy [N, flat_dim])
    """
    if cond_pil is None:
        from PIL import Image
        cond_pil = Image.new("RGB", (512, 512), color=(255, 255, 255))

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
        print(f"  Progress: {len(all_images)}/{num_samples}", end="\r")

    print()
    pipe.vae.to(dtype=original_vae_dtype)
    return all_images, np.vstack(all_lats)


def predict_noise_cfg(unet, scheduler, latents_in, t,
                      encoder_states, added_cond, gs):
    """Classifier-free guidance noise prediction."""
    lmi = scheduler.scale_model_input(torch.cat([latents_in] * 2), t)
    np_out = unet(
        lmi, t,
        encoder_hidden_states=encoder_states,
        added_cond_kwargs=added_cond,
        return_dict=False,
    )[0]
    np_u, np_t = np_out.chunk(2)
    return np_u + gs * (np_t - np_u)


def compute_pred_x0_direct(scheduler, noise_pred, t, latents_in):
    """
    Compute pred_x0 without calling scheduler.step() (avoids step_index side effects).
    Uses the diffusion formula: x0 = (x_t - sqrt(1-alpha) * eps) / sqrt(alpha)
    """
    if hasattr(scheduler, "alphas_cumprod"):
        alpha = scheduler.alphas_cumprod[t.long().cpu()].to(latents_in.device)
    else:
        sigma = scheduler.sigmas[scheduler.step_index]
        alpha = (1 / (sigma ** 2 + 1)).to(latents_in.device)
    return (latents_in - (1 - alpha) ** 0.5 * noise_pred) / alpha ** 0.5


def denoise_step(scheduler, noise_pred, t, latents_in, correction=None):
    """Apply one scheduler denoising step, optionally adding a DPS correction."""
    x_t_minus_1 = scheduler.step(
        noise_pred, t, latents_in, return_dict=True
    ).prev_sample
    if correction is not None:
        x_t_minus_1 = x_t_minus_1 + correction
    return x_t_minus_1.detach()


def run_dps_step_clip(
    latents_step,
    pixel_x0_norm,
    sprinter,
    all_clip_embeddings,
    num_variations,
    variation_batch_size,
    base_zeta_prime,
    clip_model,
    clip_processor,
    vae,
    vae_scaling_factor,
    variation_prompt,
    loss_fn=compute_mmd,
    loss_scale=1.0,
):
    """
    Compute DPS gradient via CLIP-MMD guidance.

    Gradient flows: latents_step -> UNet -> pred_x0 -> VAE(fp32) -> pixel_x0_norm
                    -> sprinter (num_variations times) -> VAE decode -> CLIP -> MMD

    pixel_x0_norm is passed directly into each checkpoint call so the autograd
    graph from latents_step is preserved all the way to the loss.

    Returns:
        grad, loss_scaled, zeta_i, loss_norm, variation_clip_embs_np
    """
    from src.clip_utils import encode_images_clip

    device = pixel_x0_norm.device
    clip_model.to(device)

    print(f"      [grad] pixel_x0_norm.requires_grad={pixel_x0_norm.requires_grad}  "
          f"grad_fn={pixel_x0_norm.grad_fn}", flush=True)

    variation_clip_list = []
    for _ in range(num_variations):

        # pixel_x0_norm is passed as the checkpoint input so grad flows through it
        def sprinter_vae_clip_forward(ctrl):
            var_latents = sprinter(
                prompt=[variation_prompt],
                image=ctrl,
                num_inference_steps=2,
                guidance_scale=0.0,
                controlnet_conditioning_scale=0.5,
                output_type="latent",
                return_dict=True,
            ).images
            var_pixels = vae.decode(
                (var_latents.float() / vae_scaling_factor).to(vae.dtype)
            ).sample
            var_pixels = (var_pixels.float() + 1.0) / 2.0
            var_pixels = var_pixels.clamp(0.0, 1.0)
            with torch.amp.autocast("cuda", enabled=False):
                return encode_images_clip(var_pixels.float(), clip_model, clip_processor)

        var_clip = torch.utils.checkpoint.checkpoint(
            sprinter_vae_clip_forward, pixel_x0_norm, use_reentrant=False
        )
        variation_clip_list.append(var_clip)

    print(f"      [grad] var_clip.requires_grad={var_clip.requires_grad}  "
          f"grad_fn={var_clip.grad_fn}", flush=True)

    variation_clip_embs = torch.cat(variation_clip_list, dim=0)
    torch.cuda.empty_cache()

    loss_value = loss_fn(variation_clip_embs, all_clip_embeddings.detach())
    loss_scaled = loss_value * loss_scale

    print(f"      [grad] loss_scaled.requires_grad={loss_scaled.requires_grad}  "
          f"grad_fn={loss_scaled.grad_fn}", flush=True)

    loss_norm = loss_scaled.detach()
    zeta_i = base_zeta_prime / loss_norm

    grad = torch.autograd.grad(
        loss_scaled, latents_step,
        retain_graph=False, create_graph=False,
    )[0]

    variation_clip_np = variation_clip_embs.detach().cpu().numpy()
    del variation_clip_list, variation_clip_embs

    return grad, loss_scaled, zeta_i, loss_norm, variation_clip_np
