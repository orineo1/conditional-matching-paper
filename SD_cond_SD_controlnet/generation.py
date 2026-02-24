import torch
import numpy as np
import gc
from metrics import compute_mmd

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
            prompt=[prompt] * curr, image=[sobel_cond_pil] * curr,
            num_inference_steps=2, guidance_scale=0.0,
            controlnet_conditioning_scale=1.0,
            callback_on_step_end=latents_callback,
        )
        all_images.extend(result.images)
        all_lats.append(pipe._current_latents.reshape(curr, -1))
        print(f"  Progress: {len(all_images)}/{num_samples}", end="\r")

    print()
    pipe.vae.to(dtype=original_vae_dtype)
    return all_images, np.vstack(all_lats)


def predict_noise_cfg(unet, scheduler, latents_in, t, encoder_states, added_cond, gs):
    lmi = scheduler.scale_model_input(torch.cat([latents_in] * 2), t)
    np_out = unet(lmi, t, encoder_hidden_states=encoder_states,
                  added_cond_kwargs=added_cond, return_dict=False)[0]
    np_u, np_t = np_out.chunk(2)
    return np_u + gs * (np_t - np_u)


def compute_pred_x0(scheduler, noise_pred, t, latents_in):
    so = scheduler.step(noise_pred, t, latents_in, return_dict=True)
    if hasattr(so, 'pred_original_sample') and so.pred_original_sample is not None:
        return so.pred_original_sample
    alpha = scheduler.alphas_cumprod[t.long().cpu()].to(latents_in.device)
    return (latents_in - (1 - alpha)**0.5 * noise_pred) / alpha**0.5


def denoise_step(scheduler, noise_pred, t, latents_in, correction=None):
    x_t_minus_1 = scheduler.step(noise_pred, t, latents_in, return_dict=True).prev_sample
    if correction is not None:
        x_t_minus_1 = x_t_minus_1 + correction
    return x_t_minus_1.detach()


def run_dps_step(latents, latents_step, noise_pred, pixel_x0_norm,
                 sprinter, all_latents, num_variations, variation_batch_size, base_zeta_prime):
    variation_latents_list = []
    print(f"  Generating {num_variations} variations...")

    for start_idx in range(0, num_variations, variation_batch_size):
        end_idx = min(start_idx + variation_batch_size, num_variations)
        bs = end_idx - start_idx
        ctrl_batch = pixel_x0_norm[0].unsqueeze(0).repeat(bs, 1, 1, 1)

        def sprinter_forward(ctrl):
            return sprinter(
                prompt=["a superrealistic professional photograph of"] * ctrl.shape[0],
                image=ctrl, num_inference_steps=2, guidance_scale=0.0,
                controlnet_conditioning_scale=0.8, output_type="latent", return_dict=True,
            ).images

        with torch.cuda.amp.autocast():
            vl = torch.utils.checkpoint.checkpoint(sprinter_forward, ctrl_batch, use_reentrant=False)
        variation_latents_list.append(vl)
        torch.cuda.empty_cache()

    variation_latents = torch.cat(variation_latents_list, dim=0)
    mmd_squared = compute_mmd(variation_latents, all_latents)
    mmd_loss = torch.sqrt(mmd_squared + 1e-8)
    loss_norm = mmd_loss.detach()
    zeta_i = base_zeta_prime / loss_norm

    grad = torch.autograd.grad(mmd_loss, latents_step, retain_graph=False, create_graph=False)[0]
    vl_flat = variation_latents.detach().cpu().numpy().reshape(variation_latents.shape[0], -1)
    del variation_latents_list, variation_latents

    return grad, mmd_loss, zeta_i, loss_norm, vl_flat