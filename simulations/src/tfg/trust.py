"""Noise-level trust region on the applied guidance step -- the ONE implementation
shared by the synthetic engine (``engine.GeneralizedTFG._step_clip``) and the
Stable-Diffusion pipeline (``SD_cond_SD_controlnet/src/trust.py`` adapter).

Two conventions of the same rule:

* ``noise`` (synthetic, promoted ``trust_noise1``):
      ||Delta_t|| <= tau * sqrt(1 - alphabar_t)                       (numel = 1)
* ``noise_prev_rms`` (SD latent convention, ``sd/PIPELINE.md`` sec 2):
      ||Delta||   <= tau * sqrt(1 - alphabar_{t_prev}) * sqrt(numel)
  i.e. the per-element RMS of the step may not exceed ``tau`` noise stds of the
  state the step is added to.  ``sqrt(numel)`` turns the per-element std into
  the L2 norm of an isotropic noise vector; synthetic ``tau=1`` corresponds to
  ``tau = 1/sqrt(numel)`` under this convention.

Both are pure rescalings of the direction: ``Delta * min(1, cap / ||Delta||)``.
"""

import math

import torch


def noise_cap(tau, sqrt_one_minus_ab, numel=1, min_noise=0.0):
    """``cap = tau * max(sqrt(1 - abar), min_noise) * sqrt(numel)``.

    ``min_noise`` floors the noise amplitude so a step landing on a noise-free
    state (``abar = 1``) keeps a small non-zero cap instead of 0.
    """
    s = max(float(sqrt_one_minus_ab), float(min_noise))
    return float(tau) * s * math.sqrt(float(numel))


def clip_step(delta, cap, eps=1e-12):
    """``delta * clamp(cap / (||delta|| + eps), max=1)`` -- direction preserved,
    norm capped at ``cap`` (torch expression, differentiable, used verbatim by
    the engine so the promoted rule's numerics are unchanged)."""
    nrm = delta.norm()
    factor = torch.clamp(cap / (nrm + eps), max=1.0)
    return delta * factor
