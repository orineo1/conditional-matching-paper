"""Exact cached-target MMD for the distributional predictor (opt-in).

Minimal port of ``experiments/model-optimization/exact_loss/fast_mmd.py::MMDFixedTarget``
(Agent 2; verified EXACT by the campaign verifier).  Computes the SAME value and
the SAME first-order gradient as ``LossFunctions.MMDLoss(RBF(...))``::

    K   = sum_k exp(-D / (bw * m_k)),  m_k = mul_factor ** (k - n_kernels//2)
    bw  = fixed, or  D.sum() / (N^2 - N) on the STACKED (X;Y), N = n + m,
          NOT detached (the gradient flows through the bandwidth, as in the reference)
    MMD^2_V = K_xx.mean() - 2 K_xy.mean() + K_yy.mean()       (biased V-statistic)

but caches every Y-only quantity (Y, ||y||^2, D_yy and, for a fixed bandwidth,
the scalar K_yy mean) so that each call only evaluates the (n x n) and (n x m)
blocks; the reference re-evaluates the (m x m) block -- by far the largest for
n << m -- on every call and through autograd.  Squared distances use
``||x||^2 + ||y||^2 - 2 x.y`` clamped at 0 (``dist='mm'``; ``torch.cdist`` itself
uses this formula above 25 rows).  With an adaptive bandwidth the YY term
depends on bw: its value and d/d(bw) are computed from the cached D_yy under
``no_grad`` and re-attached first-order exactly (``f(bw0) + f'(bw0)(bw - bw0)``) --
exact value and exact first derivative, not a valid second derivative.

Equality to the reference is asserted at 1e-12 (float64) in
``tests/test_fast_mmd_integration.py``.
"""

import torch


def _sq_dists(A, B, A_sq=None, B_sq=None):
    A_sq = (A * A).sum(-1, keepdim=True) if A_sq is None else A_sq
    B_sq = (B * B).sum(-1, keepdim=True) if B_sq is None else B_sq
    return (A_sq + B_sq.transpose(-1, -2) - 2.0 * (A @ B.transpose(-1, -2))).clamp_min(0.0)


class MMDFixedTarget:
    """``loss(X) -> MMD^2_V(X, Y)`` against a fixed target ``Y`` (m, d)."""

    def __init__(self, Y, bandwidth=None, n_kernels=5, mul_factor=2.0):
        Y = Y.detach()
        self.Y, self.m = Y, Y.shape[0]
        self.dtype, self.device = Y.dtype, Y.device
        self.n_kernels, self.mul_factor = int(n_kernels), float(mul_factor)
        self.adaptive = bandwidth is None
        # reference dtype semantics: multipliers take the DEFAULT dtype
        self.mults = self.mul_factor ** (torch.arange(self.n_kernels, device=self.device)
                                         - self.n_kernels // 2)
        self.bandwidth = (None if self.adaptive
                          else torch.as_tensor(bandwidth, dtype=self.dtype, device=self.device))
        self.Y_sq = (Y * Y).sum(-1, keepdim=True)
        with torch.no_grad():
            self.D_yy = _sq_dists(Y, Y, self.Y_sq, self.Y_sq)
            self.D_yy_sum = self.D_yy.sum()
            if not self.adaptive:
                self.YY_fixed = self._K(self.D_yy, self._scales(self.bandwidth)).mean()

    def _scales(self, bw):
        return bw * self.mults

    def _K(self, D, scales):
        shape = (-1,) + (1,) * D.dim()
        return torch.exp(-(D.unsqueeze(0) / scales.reshape(shape))).sum(0)

    def _yy_term(self, bw):
        if not self.adaptive:
            return self.YY_fixed
        bw0 = bw.detach()
        with torch.enable_grad():
            b = bw0.clone().requires_grad_(True)
            f = self._K(self.D_yy, self._scales(b)).mean()
            (df,) = torch.autograd.grad(f, b)
        return f.detach() + df.detach() * (bw - bw0)

    def __call__(self, X):
        X = X.to(self.dtype)
        n = X.shape[0]
        X_sq = (X * X).sum(-1, keepdim=True)
        D_xx = _sq_dists(X, X, X_sq, X_sq)
        D_xy = _sq_dists(X, self.Y, X_sq, self.Y_sq)
        if self.adaptive:
            N = n + self.m
            bw = (D_xx.sum() + 2.0 * D_xy.sum() + self.D_yy_sum) / (N ** 2 - N)
        else:
            bw = self.bandwidth
        scales = self._scales(bw)
        xx = self._K(D_xx, scales).mean()
        xy = self._K(D_xy, scales).sum() / (n * self.m)
        return xx - 2.0 * xy + self._yy_term(bw)
