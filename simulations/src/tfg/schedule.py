"""Diffusion noise schedule, held in float64.

The repository's :class:`Diffusion.DiffusionModel` builds ``betas``/``alphas``/
``baralphas`` as hard-coded float32 *plain attributes* (Diffusion.py:26-30).
They are not registered buffers, so ``model.double()`` does not convert them
and they do not follow ``.to(device)``.  For an equivalence test targeting
1e-7, schedule constants in float32 dominate the error budget, so the reference
engine and the generalised engine share one schedule object built here in
float64 instead of reading the model's attributes.

Indexing convention used throughout this package, chosen to match TFG
Algorithm 1 rather than the repository's loop:

    ``alphabar[t]`` for ``t`` in ``0..T``, with ``alphabar[0] == 1`` (clean
    data) and ``alphabar[T]`` the noisiest level.  Step ``t`` maps ``x_t`` to
    ``x_{t-1}``, and the loop runs ``t = T, T-1, ..., 1`` so that the final
    state is ``x_0``.

Note this differs from ``Optimization.py:98``, whose loop
``range(diffusion_steps-1, 0, -1)`` stops at ``t=1`` and therefore never
produces ``x_0``.  That discrepancy is recorded in the Checkpoint 0 report; we
do not reproduce it.
"""

import math

import torch


class DiffusionSchedule:
    """Cosine (Nichol & Dhariwal) schedule over ``T`` steps, in float64."""

    def __init__(self, T=100, s=0.008, max_beta=0.999, min_alphabar=None,
                 dtype=torch.float64, device="cpu"):
        self.T = int(T)
        self.dtype = dtype
        self.device = torch.device(device)

        t = torch.arange(self.T + 1, dtype=dtype, device=device)
        f = torch.cos(((t / self.T) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        raw = f / f[0]

        # The raw cosine gives alphabar[T] == 0 exactly, because the argument
        # is exactly pi/2 at t = T.  That makes 1/sqrt(alphabar_T) unbounded,
        # and since x_{0|t} = (x_t - sqrt(1-ab_t) eps) / sqrt(ab_t) and line 9
        # divides Delta_t by sqrt(ab_t) as well, the whole trajectory blows up.
        # Nichol & Dhariwal handle this by clipping the per-step beta, which is
        # what we do -- and nothing else.  This is the model's actual schedule.
        #
        # `min_alphabar` is an OPTIONAL, TEST-ONLY numerical guard and defaults
        # to None (no floor).  It must never be set for an experiment-facing
        # schedule: at T=100 a floor of 1e-4 raises alphabar_T from 2.4e-7 to
        # 1e-4 (412x) and changes alpha_T from 0.001 to 0.41, i.e. it
        # substantially changes the diffusion process itself.
        betas = (1.0 - raw[1:] / raw[:-1]).clamp(min=0.0, max=max_beta)
        alphabar = torch.cumprod(1.0 - betas, dim=0)
        alphabar = torch.cat([torch.ones(1, dtype=dtype, device=device), alphabar])
        alphabar = alphabar.clamp(max=1.0)
        if min_alphabar is not None:
            alphabar = alphabar.clamp(min=float(min_alphabar))
        self.alphabar = alphabar
        self.min_alphabar = min_alphabar
        # Recompute betas FROM the final clamped alphabar so the two are always
        # mutually consistent.  Deriving them before the clamp left
        # cumprod(1-betas) disagreeing with alphabar by up to 1e-4, which would
        # silently give a different schedule to anything reading `betas`.
        self.betas = 1.0 - self.alphabar[1:] / self.alphabar[:-1]

    # -- accessors ---------------------------------------------------------

    def ab(self, t):
        """alphabar_t."""
        return self.alphabar[t]

    def alpha(self, t):
        """alpha_t = alphabar_t / alphabar_{t-1}, the per-step retention."""
        return self.alphabar[t] / self.alphabar[t - 1]

    def sqrt_ab(self, t):
        return torch.sqrt(self.alphabar[t])

    def sqrt_one_minus_ab(self, t):
        return torch.sqrt(1.0 - self.alphabar[t])

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = torch.device(device)
            self.alphabar = self.alphabar.to(self.device)
        if dtype is not None:
            self.dtype = dtype
            self.alphabar = self.alphabar.to(dtype)
        return self


def constant_vector(value, T, dtype=torch.float64, device="cpu"):
    """A time-indexed guidance strength that is constant in ``t``.

    Returned as a length ``T+1`` tensor so it can be indexed by ``t`` directly.
    """
    return torch.full((T + 1,), float(value), dtype=dtype, device=device)


def structured_vector(scalar, structure, schedule):
    """Build ``rho``/``mu`` as ``scalar * s(t)`` for the three TFG structures.

    TFG (Ye et al., 2024, Eq. 8) normalises each structure so that
    ``sum_t s(t) == T``:

        increase:  s(t) = alpha_t   / mean_t(alpha_t)
        decrease:  s(t) = (1-alpha_t) / mean_t(1-alpha_t)
        constant:  s(t) = 1
    """
    T = schedule.T
    dtype, device = schedule.dtype, schedule.device
    s = torch.ones(T + 1, dtype=dtype, device=device)

    if structure == "constant":
        pass
    elif structure in ("increase", "decrease"):
        alphas = torch.stack([schedule.alpha(t) for t in range(1, T + 1)])
        raw = alphas if structure == "increase" else (1.0 - alphas)
        raw = raw / raw.mean()
        s[1:] = raw
    else:
        raise ValueError(f"unknown structure {structure!r}")

    return float(scalar) * s
