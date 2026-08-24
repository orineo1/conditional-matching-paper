"""Mathematically exact, opt-in accelerations of the repository MMD loss.

Reference (``simulations/src/LossFunctions.py``)::

    K  = sum_k exp(-D / (bw * m_k)),   D = cdist(Z, Z)^2,  Z = vstack(X, Y)
    m_k = mul_factor ** (k - n_kernels // 2),   k = 0..n_kernels-1
    bw  = bandwidth  (fixed)   or   D.sum() / (N^2 - N),  N = n + m  (adaptive,
          NOT detached: the bandwidth is a function of X and Y jointly)
    MMD^2_V = K[:n,:n].mean() - 2 K[:n,n:].mean() + K[n:,n:].mean()   (biased V-stat)

SD pipeline (``SD_cond_SD_controlnet/src/metrics.py::compute_mmd``)::

    k(x,y) = exp(-(||x-y||^2 / (2 bw^2)) ** alpha)            (single kernel)
    bw     = sqrt(median(D_xy[D_xy > 0]) / 2) * bandwidth_scale, DETACHED
    MMD^2_U = unbiased U-statistic (diagonals removed; xx term 0 when n == 1)
    loss   = sqrt(|MMD^2_U| + 1e-8);   adaptive zeta_i = base_zeta / loss.detach()

Both are covered by one parameterisation: ``K(D) = sum_k exp(-(D / s_k) ** alpha)``
with scale vector ``s``.  Repo: ``s_k = bw * m_k``, ``alpha = 1``.  SD: ``s = 2 bw^2``.

Everything here is *exact*: same value, same first-order gradient w.r.t. X (and,
in the adaptive case, the same gradient flow through the bandwidth), up to
floating-point reordering (~1e-15 relative in float64).  Nothing is approximated.

Variants (all opt-in, selected by constructor arguments):

* ``MMDFixedTarget`` -- caches Y, ||y||^2, the Y-Y squared distances and, for a
  fixed bandwidth, the scalar YY kernel mean.  Per call it only evaluates the
  XX (n x n) and XY (n x m) blocks.  The reference evaluates the YY (m x m) block
  -- by far the largest when n << m -- on every call, through ``exp`` and
  (needlessly) through autograd.
* ``dist="cdist" | "mm"`` -- squared distances by ``torch.cdist(.)**2`` or by
  ``||x||^2 + ||y||^2 - 2 x y^T`` clamped at 0 (note ``torch.cdist`` itself
  switches to the matmul formula for > 25 rows, so the reference is already
  matmul-based for m = 250).
* ``kernel_eval="exp" | "powchain"`` -- ``powchain`` is valid for ``alpha == 1``
  and integer ``mul_factor``: with multipliers 1/4,1/2,1,2,4 all five kernels are
  powers of one base, ``E = exp(-D / (4 bw))``, ``K = E + E^2 + E^4 + E^8 + E^16``,
  so one ``exp`` plus 4 squarings replaces 5 ``exp`` calls over an (n x m) block.
* ``chunk`` -- blockwise evaluation of the XY block over Y, with a fused
  autograd.Function that never stores the (K, n, m) kernel tensor: memory
  O(n d + chunk * n) instead of O(K n m).
* ``batched(Xb)`` -- B independent sample sets (B, n, d) against the same Y in one
  call, returning (B,) losses (for LGD's M perturbations / independent restarts).
* ``bandwidth=None`` -- the reference adaptive rule on the STACKED matrix,
  including the gradient through the bandwidth.  The YY kernel block then does
  depend on X (through bw); its value and d/d(bw) are computed from the cached
  Y-Y distances under ``no_grad`` and re-attached to the graph with a first-order
  exact re-attachment (``f(bw0) + f'(bw0) (bw - bw0)``) -- exact value and exact
  first derivative, but NOT a valid second derivative (``create_graph=True``
  through the YY term would see zero curvature).  Use ``reattach_yy="autograd"``
  for full higher-order correctness (costs the m x m autograd graph).
* ``unbiased=True`` -- U-statistic (SD convention); ``sd_output=True`` applies the
  SD ``sqrt(|MMD^2| + 1e-8)``.

Which experiments use what
--------------------------
* synthetic Exp 2-7 (``simulations/experiments/_guided.py``): ``MMDLoss(RBF(bandwidth=fixed))``,
  biased V-statistic, 5 bandwidths, mul_factor 2, bandwidth frozen from S_G
  (``_common.fixed_bandwidth``), Y = S_G (250 x d) constant, n in 1..32.
* SD pipeline (``run_mlgd_f.py`` / ``metrics.compute_mmd``): UNBIASED U-statistic,
  single generalised kernel with ``kernel_alpha``, detached median bandwidth
  estimated from X and Y each call, ``sqrt(|.|+1e-8)`` output, float32.
* MNIST (``MNIST/run_mlgdf.py``): sliced Wasserstein (50 random projections, L1
  of sorted projections), no MMD in the guidance loop.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

__all__ = [
    "repo_scales", "sd_scale", "sd_median_bandwidth", "sq_dists", "kernel_from_d2",
    "MMDFixedTarget", "mmd_reference_like", "mmd_batched_stacked",
]


# --------------------------------------------------------------------------- #
# kernel / scale helpers
# --------------------------------------------------------------------------- #
def repo_scales(bandwidth, n_kernels: int = 5, mul_factor: float = 2.0,
                dtype=torch.float64, device="cpu"):
    """``s_k = bandwidth * mul_factor ** (k - n_kernels // 2)`` (LossFunctions.RBF).

    Reproduces the reference's dtype semantics on purpose: the multipliers are
    built with ``mul_factor ** (torch.arange(n) - n//2)`` which takes the *default*
    dtype (float32 unless ``torch.set_default_dtype(float64)``), and a Python-float
    or 0-dim bandwidth is promoted to THAT dtype, i.e. the reference rounds the
    bandwidth to float32 when run under the default dtype even if X is float64.
    """
    mults = mul_factor ** (torch.arange(n_kernels, device=device) - n_kernels // 2)
    if torch.is_tensor(bandwidth):
        bw = bandwidth.reshape(()) if bandwidth.dim() <= 1 else bandwidth
    else:
        bw = torch.as_tensor(bandwidth, dtype=dtype, device=device)
    return bw * mults


def sd_scale(bandwidth, dtype=torch.float64, device="cpu"):
    """``s = 2 * bw**2`` so that ``exp(-(D/s)**alpha)`` is the SD generalised RBF."""
    bw = bandwidth if torch.is_tensor(bandwidth) else torch.as_tensor(
        bandwidth, dtype=dtype, device=device)
    return (2.0 * bw ** 2).reshape(1)


def sd_median_bandwidth(X, Y, bandwidth_scale: float = 1.0, subsample: int = 1000):
    """Detached median heuristic of ``metrics.compute_mmd`` (first ``ss`` rows of each)."""
    ss = min(subsample, X.shape[0], Y.shape[0])
    with torch.no_grad():
        d = sq_dists(X[:ss].detach(), Y[:ss], mode="mm", clamp=False)
        d = d[d > 0]
        bw = (torch.sqrt(torch.median(d) / 2) if d.numel() > 0
              else torch.tensor(1.0, dtype=X.dtype, device=X.device))
    return bw.detach() * bandwidth_scale


def sq_dists(A, B, mode: str = "cdist", clamp: bool = True, A_sq=None, B_sq=None):
    """Pairwise squared Euclidean distances (.., n, d) x (.., m, d) -> (.., n, m)."""
    if mode == "cdist":
        return torch.cdist(A, B, p=2) ** 2
    if mode == "mm":
        if A_sq is None:
            A_sq = (A * A).sum(-1, keepdim=True)
        if B_sq is None:
            B_sq = (B * B).sum(-1, keepdim=True)
        D = A_sq + B_sq.transpose(-1, -2) - 2.0 * (A @ B.transpose(-1, -2))
        return D.clamp_min(0.0) if clamp else D
    raise ValueError(mode)


def _powchain_ok(scales, alpha, mul_factor):
    if alpha != 1.0 or float(mul_factor) != float(int(mul_factor)) or len(scales) < 2:
        return False
    return True


def kernel_from_d2(D, scales, alpha: float = 1.0, kernel_eval: str = "exp",
                   mul_factor: float = 2.0):
    """``sum_k exp(-(D / s_k) ** alpha)``, elementwise over D (any shape).

    ``kernel_eval="exp"``: broadcasts to (K, *D.shape) -- what the reference does.
    ``kernel_eval="powchain"`` (alpha == 1, integer mul_factor): one exp at the
    LARGEST scale, then repeated integer powers.  Exact up to a few ulp.
    ``kernel_eval="loop"``: accumulates kernel by kernel (no (K, ...) tensor).
    """
    if kernel_eval == "powchain" and _powchain_ok(scales, alpha, mul_factor):
        s_sorted, _ = torch.sort(scales, descending=True)
        E = torch.exp(-D / s_sorted[0])
        K = E
        cur = E
        f = int(mul_factor)
        for _ in range(1, len(scales)):
            # scales are s_0 / f^j  ->  exp(-D f^j / s_0) = E ** (f^j)
            cur = cur ** f if f != 2 else cur * cur
            K = K + cur
        return K
    if kernel_eval == "loop":
        K = None
        for s in scales:
            z = D / s
            term = torch.exp(-(z ** alpha if alpha != 1.0 else z))
            K = term if K is None else K + term
        return K
    # "exp": reference broadcasting
    shape = (-1,) + (1,) * D.dim()
    z = D.unsqueeze(0) / scales.reshape(shape)
    if alpha != 1.0:
        z = z ** alpha
    return torch.exp(-z).sum(0)


# --------------------------------------------------------------------------- #
# fused / chunked XY kernel sum (no (K, n, m) tensor kept for backward)
# --------------------------------------------------------------------------- #
class _ChunkedKernelSum(torch.autograd.Function):
    """S = sum_{i,j,k} exp(-(||x_i - y_j||^2 / s_k)^alpha) computed in chunks over Y.

    Returns grads w.r.t. X and w.r.t. the scale vector s (for adaptive bandwidth).
    First-order exact; stores only (n, d) + (K,) for backward.
    """

    @staticmethod
    def forward(ctx, X, Y, scales, alpha, chunk, dist_mode, X_sq, Y_sq):
        n, d = X.shape
        m = Y.shape[0]
        S = X.new_zeros(())
        gX = torch.zeros_like(X)
        gs = torch.zeros_like(scales)
        need_gs = scales.requires_grad
        for j0 in range(0, m, chunk):
            Yb = Y[j0:j0 + chunk]
            D = sq_dists(X, Yb, mode=dist_mode, A_sq=X_sq,
                         B_sq=None if Y_sq is None else Y_sq[j0:j0 + chunk])
            W = torch.zeros_like(D)       # sum_k dK/dD  (negative)
            for k, s in enumerate(scales):
                z = D / s
                if alpha == 1.0:
                    e = torch.exp(-z)
                    dz = e / s                       # -dK/dD per kernel
                    if need_gs:
                        gs[k] += (e * z).sum() / s
                else:
                    za = z ** alpha
                    e = torch.exp(-za)
                    dz = e * alpha * z ** (alpha - 1.0) / s
                    if need_gs:
                        gs[k] += (e * alpha * za).sum() / s
                S = S + e.sum()
                W = W - dz
            # dD/dx_i = 2 (x_i - y_j)  ->  gX_i += 2 * sum_j W_ij (x_i - y_j)
            gX += 2.0 * (W.sum(1, keepdim=True) * X - W @ Yb)
        ctx.save_for_backward(gX, gs)
        return S

    @staticmethod
    def backward(ctx, g):
        gX, gs = ctx.saved_tensors
        return g * gX, None, g * gs, None, None, None, None, None


def chunked_kernel_sum(X, Y, scales, alpha=1.0, chunk=512, dist_mode="mm",
                       X_sq=None, Y_sq=None):
    return _ChunkedKernelSum.apply(X, Y, scales, float(alpha), int(chunk), dist_mode,
                                   X_sq, Y_sq)


# --------------------------------------------------------------------------- #
# main class
# --------------------------------------------------------------------------- #
class MMDFixedTarget:
    """MMD against a fixed target Y with cached Y-only quantities.  Exact.

    Parameters
    ----------
    Y : (m, d) target samples (detached, cached).
    bandwidth : float | tensor | None.  None -> reference adaptive rule on the stacked
        (X;Y) matrix with gradient flowing through the bandwidth.
    kernel : "repo" (5 RBFs, ``s_k = bw * 2^(k-2)``, alpha forced 1 unless given)
             or "sd" (single ``s = 2 bw^2``, exponent alpha).
    unbiased : U-statistic (SD) instead of V-statistic (repo).
    sd_output : return ``sqrt(|MMD^2| + 1e-8)`` (SD convention).
    dist : "cdist" | "mm" for XX / XY squared distances.
    kernel_eval : "exp" | "powchain" | "loop".
    chunk : None, or block size over Y for the fused chunked XY evaluation.
    reattach_yy : "linear" (first-order exact, cheap) | "autograd" (full graph) --
        only relevant for the adaptive bandwidth, where YY depends on bw.
    """

    def __init__(self, Y, bandwidth=None, *, kernel="repo", n_kernels=5, mul_factor=2.0,
                 alpha=1.0, unbiased=False, sd_output=False, dist="cdist",
                 kernel_eval="exp", chunk: Optional[int] = None, reattach_yy="linear"):
        Y = Y.detach()
        self.Y, self.m, self.d = Y, Y.shape[0], Y.shape[1]
        self.dtype, self.device = Y.dtype, Y.device
        self.kernel, self.n_kernels, self.mul_factor = kernel, n_kernels, float(mul_factor)
        self.alpha = float(alpha)
        self.unbiased, self.sd_output = unbiased, sd_output
        self.dist, self.kernel_eval, self.chunk, self.reattach_yy = dist, kernel_eval, chunk, reattach_yy
        self.adaptive = bandwidth is None
        self.bandwidth = None if self.adaptive else torch.as_tensor(
            bandwidth, dtype=self.dtype, device=self.device)
        # cached Y-only quantities
        self.Y_sq = (Y * Y).sum(-1, keepdim=True)                       # (m, 1)
        with torch.no_grad():
            self.D_yy = sq_dists(Y, Y, mode=dist, B_sq=self.Y_sq, A_sq=self.Y_sq)
            self.D_yy_sum = self.D_yy.sum()
            if not self.adaptive:
                self._scales_fixed = self._scales(self.bandwidth)
                self.YY_fixed = self._yy_stat(kernel_from_d2(
                    self.D_yy, self._scales_fixed, self.alpha, "exp"))
        self._yy_cache = {}

    # -- helpers ---------------------------------------------------------- #
    def _scales(self, bw):
        if self.kernel == "repo":
            return repo_scales(bw, self.n_kernels, self.mul_factor, self.dtype, self.device)
        if self.kernel == "sd":
            return sd_scale(bw, self.dtype, self.device)
        raise ValueError(self.kernel)

    def _K(self, D, scales):
        return kernel_from_d2(D, scales, self.alpha, self.kernel_eval, self.mul_factor)

    def _yy_stat(self, Kyy):
        m = self.m
        if self.unbiased:
            return (Kyy.sum() - Kyy.diagonal().sum()) / (m * (m - 1)) if m > 1 else Kyy.new_zeros(())
        return Kyy.mean()

    def _xx_stat(self, Kxx, n):
        if self.unbiased:
            return ((Kxx.sum() - Kxx.diagonal(dim1=-2, dim2=-1).sum(-1)) / (n * (n - 1))
                    if n > 1 else Kxx.sum(dim=(-2, -1)) * 0.0)
        return Kxx.mean(dim=(-2, -1))

    def _yy_term(self, bw):
        """YY statistic as a function of (possibly graph-attached) bandwidth."""
        if not self.adaptive:
            return self.YY_fixed
        if self.reattach_yy == "autograd":
            return self._yy_stat(self._K(self.D_yy, self._scales(bw)))
        # first-order exact re-attachment: value f(bw0), gradient f'(bw0)
        bw0 = bw.detach()
        with torch.enable_grad():
            b = bw0.clone().requires_grad_(True)
            f = self._yy_stat(self._K(self.D_yy, self._scales(b)))
            (df,) = torch.autograd.grad(f, b)
        return f.detach() + df.detach() * (bw - bw0)

    def _bandwidth(self, X, D_xx, D_xy):
        """Reference adaptive rule: mean off-diagonal squared distance of vstack(X, Y)."""
        if not self.adaptive:
            return self.bandwidth
        n = X.shape[0]
        N = n + self.m
        if D_xy is None:   # chunked path: sum_ij ||x_i - y_j||^2 in closed form
            xy_sum = (self.m * (X * X).sum() + n * self.Y_sq.sum()
                      - 2.0 * (X.sum(0) * self.Y.sum(0)).sum())
        else:
            xy_sum = D_xy.sum()
        tot = D_xx.sum() + 2.0 * xy_sum + self.D_yy_sum
        return tot / (N ** 2 - N)

    def _finish(self, xx, xy, yy):
        mmd2 = xx - 2.0 * xy + yy
        if self.sd_output:
            return torch.sqrt(mmd2.abs() + 1e-8)
        return mmd2

    # -- single sample set -------------------------------------------------- #
    def __call__(self, X):
        X = X.to(self.dtype)
        n = X.shape[0]
        X_sq = (X * X).sum(-1, keepdim=True) if self.dist == "mm" else None
        D_xx = sq_dists(X, X, mode=self.dist, A_sq=X_sq, B_sq=X_sq)
        if self.chunk is None:
            D_xy = sq_dists(X, self.Y, mode=self.dist, A_sq=X_sq, B_sq=self.Y_sq)
        else:
            D_xy = None
        bw = self._bandwidth(X, D_xx, D_xy)
        scales = self._scales_fixed if not self.adaptive else self._scales(bw)
        xx = self._xx_stat(self._K(D_xx, scales), n)
        if D_xy is not None:
            xy = self._K(D_xy, scales).sum() / (n * self.m)
        else:
            xy = chunked_kernel_sum(X, self.Y, scales, self.alpha, self.chunk,
                                    self.dist, X_sq, self.Y_sq) / (n * self.m)
        return self._finish(xx, xy, self._yy_term(bw))

    # -- B independent sample sets ----------------------------------------- #
    def batched(self, Xb):
        """Xb: (B, n, d) -> (B,) losses, each identical to ``self(Xb[b])``."""
        Xb = Xb.to(self.dtype)
        B, n, _ = Xb.shape
        X_sq = (Xb * Xb).sum(-1, keepdim=True) if self.dist == "mm" else None
        D_xx = sq_dists(Xb, Xb, mode=self.dist, A_sq=X_sq, B_sq=X_sq)       # (B, n, n)
        D_xy = sq_dists(Xb, self.Y.expand(B, -1, -1), mode=self.dist,
                        A_sq=X_sq, B_sq=self.Y_sq.expand(B, -1, -1))       # (B, n, m)
        if self.adaptive:
            N = n + self.m
            bw = (D_xx.sum((-2, -1)) + 2.0 * D_xy.sum((-2, -1)) + self.D_yy_sum) / (N ** 2 - N)
            out = []
            for b in range(B):       # scales differ per set; K-loop per element
                scales = self._scales(bw[b])
                xx = self._xx_stat(self._K(D_xx[b], scales), n)
                xy = self._K(D_xy[b], scales).sum() / (n * self.m)
                out.append(self._finish(xx, xy, self._yy_term(bw[b])))
            return torch.stack(out)
        scales = self._scales_fixed
        xx = self._xx_stat(self._K(D_xx, scales), n)                          # (B,)
        xy = self._K(D_xy, scales).sum((-2, -1)) / (n * self.m)               # (B,)
        return self._finish(xx, xy, self.YY_fixed)


# --------------------------------------------------------------------------- #
# stacked variants (same structure as the reference, for ablation)
# --------------------------------------------------------------------------- #
def mmd_reference_like(X, Y, bandwidth=None, n_kernels=5, mul_factor=2.0, dist="cdist",
                       kernel_eval="exp"):
    """Reference algorithm (stacked (X;Y), YY recomputed) with swappable distance /
    kernel evaluation.  Used to isolate the effect of each micro-change."""
    Z = torch.vstack([X, Y])
    D = sq_dists(Z, Z, mode=dist)
    if bandwidth is None:
        N = Z.shape[0]
        bandwidth = D.sum() / (N ** 2 - N)
    scales = repo_scales(bandwidth, n_kernels, mul_factor, Z.dtype, Z.device)
    K = kernel_from_d2(D, scales, 1.0, kernel_eval, mul_factor)
    n = X.shape[0]
    return K[:n, :n].mean() - 2 * K[:n, n:].mean() + K[n:, n:].mean()


def mmd_batched_stacked(Xb, Y, bandwidth, n_kernels=5, mul_factor=2.0):
    """Naive batched baseline: loop of the reference over B sets (for benchmarking)."""
    return torch.stack([mmd_reference_like(x, Y, bandwidth, n_kernels, mul_factor)
                        for x in Xb])
