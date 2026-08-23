"""Deployable estimate of the simplified rule, using sampled gradients only.

    lambda_hat_t = Vhat_t / (Vhat_t + Vhat_{t+1} + Dhat_t^2)

No ground-truth gradient is available at run time, so every term must come from
quantities the sampler already produces, or from samples that are explicitly
charged to the compute budget.

Estimator
---------
**Within-step variance, at zero extra conditional cost.** Split the ``n_t``
conditional samples into two disjoint halves and form the two half-gradients
``g_A``, ``g_B``. Each is an estimate from ``n_t/2`` samples, so

    Var[g_half] ~ 2 * Var[g_full]      (variance scales like 1/n)

and ``(g_A - g_B)`` has variance ``2 * Var[g_half] ~ 4 * Var[g_full]``. Hence

    Vhat_t = ||g_A - g_B||^2 / 4

is an unbiased-in-scale estimate of the full-sample gradient variance built
entirely from samples that were going to be drawn anyway. The full gradient
used for the update remains the one computed on all ``n_t`` samples, so the
guidance itself is unchanged.

**Drift.** ``D_t^2 = ||g_t - g_{t+1}||^2`` is a difference of TRUE gradients,
which we cannot see. The deployable surrogate uses the two estimates we do have
and removes their sampling variance:

    Dhat_t^2 = max(0, ||ghat_t - ghat_{t+1}||^2 - Vhat_t - Vhat_{t+1})

since E||ghat_t - ghat_{t+1}||^2 = D_t^2 + V_t + V_{t+1} under independence and
unbiasedness -- the same two assumptions the simplified rule already makes.
Clipping at 0 is necessary because the subtraction can go negative on noise.

**Previous-step variance.** ``Vhat_{t+1}`` is carried over from the previous
step's own split estimate; nothing extra is drawn.

Compute accounting
------------------
The split costs **zero additional conditional samples**: it reuses the ``n_t``
draws. It costs two extra kernel evaluations per step (the two half-MMDs) and
two extra backward passes. Those are reported separately as kernel/denoiser
overhead rather than hidden.

``n_t = 1`` cannot be split, so the rule falls back to ``lambda = 0`` (no reuse)
at that step; this is recorded rather than silently patched.
"""

import torch


class DeployableLambda:
    """Online estimator of lambda_hat_t. Carries state across outer steps."""

    def __init__(self, lambda_max=0.95, warmup_steps=1):
        self.lambda_max = float(lambda_max)
        self.warmup_steps = int(warmup_steps)
        self.prev_V = None
        self.prev_g = None
        self.n_steps = 0
        self.extra_kernel_evals = 0
        self.extra_backward = 0
        self.log = []

    @staticmethod
    def split_variance(g_a, g_b):
        """Vhat for the FULL-sample gradient from two half-sample gradients."""
        return float(((g_a - g_b) ** 2).sum()) / 4.0

    def update(self, g_full, g_a=None, g_b=None):
        """Return ``(lambda_hat, diagnostics)`` for the current step.

        ``g_full`` is the gradient actually used for guidance; ``g_a``/``g_b``
        are the two half-sample gradients (``None`` when n_t < 2).
        """
        self.n_steps += 1
        if g_a is None or g_b is None:
            V_t = None
        else:
            V_t = self.split_variance(g_a, g_b)
            self.extra_kernel_evals += 2
            self.extra_backward += 2

        lam, why = 0.0, "ok"
        D2 = None
        if V_t is None:
            why = "n_t<2: cannot split, no reuse"
        elif self.prev_g is None or self.prev_V is None:
            why = "first step: no cache"
        elif self.n_steps <= self.warmup_steps:
            why = "warmup"
        else:
            raw = float(((g_full - self.prev_g) ** 2).sum())
            D2 = max(0.0, raw - V_t - self.prev_V)
            denom = V_t + self.prev_V + D2
            lam = 0.0 if denom <= 0 else V_t / denom
            lam = min(self.lambda_max, max(0.0, lam))

        diag = {"lambda_hat": lam, "V_t": V_t, "V_prev": self.prev_V,
                "D2_hat": D2, "reason": why}
        self.log.append(diag)
        self.prev_V = V_t
        self.prev_g = None if g_full is None else g_full.detach().clone()
        return lam, diag

    def overhead(self):
        return {"extra_conditional_samples": 0,
                "extra_kernel_evals": self.extra_kernel_evals,
                "extra_backward_passes": self.extra_backward,
                "steps": self.n_steps}


def oracle_lambda_simplified(V_t, V_next, D2):
    denom = V_t + V_next + D2
    return 0.0 if denom <= 0 else min(1.0, max(0.0, V_t / denom))
