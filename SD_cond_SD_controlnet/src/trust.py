"""
trust.py — SD adapter for the shared noise-level trust region (``tfg.trust``).

The DPS correction  Delta = -zeta_i * grad  is added to x_{t_prev} (the output
of the DDIM step, see generation.denoise_step).  Under the VP forward process
every latent element of x_{t_prev} carries noise of std sqrt(1 - abar_{t_prev}),
so the cap (``tfg.trust.noise_cap``, convention ``noise_prev_rms``) is

    ||Delta||_2 <= cap_t = tau * sqrt(1 - abar_{t_prev}) * sqrt(numel(Delta)),

the per-element RMS of the step may not exceed ``tau`` noise stds.  Design
notes (why sqrt(numel), relation to the synthetic ``trust_noise1`` rule) are in
``experiments/model-optimization/sd/PIPELINE.md`` sec 2 and ``tfg/trust.py``.

SD-specific here: ``prev_alpha_bar`` reads abar_{t_prev} from a diffusers
DDIMScheduler exactly as ``DDIMScheduler.step`` does, and ``apply_trust`` keeps
the pipeline's "cap <= 0 means disabled" convention.
"""

import math

import torch

import _tfg_path  # noqa: F401  (makes `tfg` importable)
from tfg.trust import clip_step, noise_cap


def prev_alpha_bar(scheduler, t):
    """abar of the timestep the DDIM step lands on (mirrors DDIMScheduler.step)."""
    t_int = int(t.item() if torch.is_tensor(t) else t)
    prev_t = t_int - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    if prev_t >= 0:
        return float(scheduler.alphas_cumprod[prev_t])
    return float(scheduler.final_alpha_cumprod)


def trust_cap(tau, abar_prev, numel):
    """cap_t = tau * sqrt(1 - abar_prev) * sqrt(numel)  (= tfg.trust.noise_cap)."""
    return noise_cap(tau, math.sqrt(max(1.0 - float(abar_prev), 0.0)), numel=numel)


def apply_trust(correction, cap):
    """
    Rescale `correction` so that ||correction||_2 <= cap (direction preserved).

    Returns (correction_clipped, scale, norm_before) with scale = min(1, cap/norm).
    cap <= 0 disables the clip (pipeline convention).
    """
    norm = float(correction.norm().item())
    if cap <= 0.0 or norm <= cap or norm == 0.0:
        return correction, 1.0, norm
    return clip_step(correction, cap, eps=0.0), cap / norm, norm
