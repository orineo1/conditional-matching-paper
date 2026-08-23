"""AdamDPS adaptive-moment stabilisation of the guidance gradient.

Transcribed from: Belardi, Lovelace, Weinberger, Gomes,
"Adaptive Moments are Surprisingly Effective for Plug-and-Play Diffusion
Sampling", ICLR 2026 (arXiv:2603.16797v2), Eqs. (10)-(11) and Algorithm 1.

Verbatim from the paper:

    m_k = beta1 * m_{k-1} + (1 - beta1) * g_t
    v_k = beta2 * v_{k-1} + (1 - beta2) * g_t^2          (element-wise square)

    ghat_t = mhat_k / (sqrt(vhat_k) + delta)

    with bias-corrected moments  mhat_k = m_k / (1 - beta1^k),
                                 vhat_k = v_k / (1 - beta2^k),
    and delta a small constant for numerical stability.

Algorithm 1 (AdamDPS) then applies it as

    g_t = -grad_{x_t} L(f_phi(x_{0|t}), y)
    ghat_t, m, v, k = AdaptiveMomentEstimate(g_t, m, v, k, beta1, beta2)
    x_s = Sample(x_{0|t}, x_t, t, s) + rho * ghat_t

Two points of care when porting to this repository:

  * The paper's ``g_t`` is the NEGATIVE gradient of the loss (an ascent
    direction), and the update ADDS ``rho * ghat_t``. The repository's
    ``optimize_LGD`` computes ``grad = dL/dx`` and SUBTRACTS it. Both are
    descent; ``AdamGuidance.step`` takes the repository's ``dL/dx`` and returns
    the quantity to SUBTRACT, so the sign convention of the surrounding loop is
    preserved. This is asserted in the tests.
  * ``k`` counts Adam steps, incremented once per applied update, and drives the
    bias correction. It is NOT the diffusion timestep.

PARITY WITH THE OFFICIAL RELEASE
--------------------------------
Checked against https://github.com/christianbelardi/adam-guidance,
commit 21f878a08279ac8399cb58c36a599c511d087fb0 (clean),
``methods/adam_dps.py`` sha256 ab49ae19b5d9ccf670eed5799b503dd902f61c3ed052338477fec23455e25ffd.
``tests/test_adam_official_parity.py`` runs the upstream
``adaptive_moment_estimate`` verbatim and asserts agreement to atol 1e-12.

Two gaps were found against our first transcription, both fixed here:

  1. ``beta2`` default is **0.995** upstream (``utils/configs.py``), not 0.999.
  2. Upstream scales the guidance by ``1 / sqrt(alpha_t)`` with
     ``alpha_t = alpha_prod_t / alpha_prod_t_prev``
     (``x_prev += guidance / alpha_t ** 0.5``). **Algorithm 1 in the paper omits
     this factor.** We follow the code; ``inv_sqrt_alpha`` exposes the choice and
     it is applied only when an ``alpha_t`` is supplied.

Upstream also clamps ``x0`` to ``clip_sample_range`` before the DDIM step. That is
an image-domain detail (pixel range) with no counterpart in our unbounded 1-D x,
so it is deliberately not ported.
"""

import torch


class AdamGuidance:
    """Adaptive-moment stabilisation, faithful to arXiv:2603.16797v2 Eqs. (10)-(11)."""

    def __init__(self, beta1=0.9, beta2=0.995, delta=1e-8, rho=1.0,
                 inv_sqrt_alpha=True):
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.delta = float(delta)
        self.rho = float(rho)
        # The official code divides the guidance by sqrt(alpha_t) (adam_dps.py:
        # ``x_prev += guidance / alpha_t ** 0.5``). Algorithm 1 in the paper
        # omits this factor. We follow the CODE, and expose the switch.
        self.inv_sqrt_alpha = bool(inv_sqrt_alpha)
        self.m = None
        self.v = None
        self.k = 0

    def reset(self):
        self.m = None
        self.v = None
        self.k = 0

    def step(self, grad_loss, alpha_t=None):
        """Return the stabilised quantity to SUBTRACT from the iterate.

        ``grad_loss`` is ``dL/dx`` in the repository's convention.
        """
        g = grad_loss.detach()
        if self.m is None:
            self.m = torch.zeros_like(g)
            self.v = torch.zeros_like(g)
        self.k += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * g * g
        m_hat = self.m / (1.0 - self.beta1 ** self.k)
        v_hat = self.v / (1.0 - self.beta2 ** self.k)
        out = self.rho * m_hat / (torch.sqrt(v_hat) + self.delta)
        if self.inv_sqrt_alpha and alpha_t is not None:
            out = out / (alpha_t ** 0.5)
        return out

    def state(self):
        return {"k": self.k, "beta1": self.beta1, "beta2": self.beta2,
                "delta": self.delta, "rho": self.rho,
                "m_norm": None if self.m is None else float(self.m.norm()),
                "v_norm": None if self.v is None else float(self.v.norm())}
