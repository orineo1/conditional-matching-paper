"""Approximate / linear-time distributional objectives for CDM guidance (Agent 3).

Every class here exposes ``loss(X)`` for a differentiable ``X`` of shape
``(n, d)`` against a FIXED target that is baked in at construction time.  All
target-only work (random features of the target, landmarks, sorted
projections, the constant ``E k(y, y')`` term) is done ONCE, because in the
guidance loop the target never changes while ``X`` changes at every diffusion
step.  The cost that matters is therefore the per-call cost in ``n``.

Reference objective (``Reference``): ``simulations/src/LossFunctions.py``
``MMDLoss(RBF(bandwidth=bw))`` -- 5 bandwidths ``bw * 2^k``, ``k=-2..2``,
kernel ``sum_k exp(-||x-y||^2 / (bw m_k))``, biased V-statistic on the stacked
``(X; Y)`` kernel matrix.  In Gaussian-kernel notation ``sigma_k^2 = bw m_k / 2``.

All random features are drawn from a ``torch.Generator`` seeded at
construction and then FROZEN (common random numbers across the diffusion
trajectory, see THEORY.md).  ``resample()`` redraws them on purpose.

Candidates
----------
* ``RFFMMD``         random Fourier features, multi-bandwidth, D features/bw
                     (``orthogonal=True`` gives ORF blocks)
* ``NystromMMD``     Nystrom features from L target landmarks
* ``SubsampledTargetMMD`` linear-time cross term: each call uses B of the m
                     targets (fixed or re-drawn), exact XX term
* ``SlicedW2``       sliced 2-Wasserstein with fixed projections (not an MMD)
* ``PopulationGMMMMD`` exact semi-population MMD^2(empirical X, GMM target)
* ``TabulatedKME1D`` d=1 only: tabulated kernel mean embedding + interpolation
* ``Reference``      the repository loss, target-constant block cached
"""
import math
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
_SIM = _HERE.parents[3] / "simulations"
for p in (_SIM / "src", _SIM / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tfg._compat import ensure_ot_stub  # noqa: E402

ensure_ot_stub()
from LossFunctions import MMDLoss, RBF  # noqa: E402
from tfg import gmm_mmd  # noqa: E402

DT = torch.float64


def multipliers(n_kernels=5, mul_factor=2.0):
    ks = torch.arange(n_kernels, dtype=DT) - n_kernels // 2
    return mul_factor ** ks


def sigmas_from_bandwidth(bw, n_kernels=5, mul_factor=2.0):
    """sigma_k with exp(-d^2/(bw m_k)) = exp(-d^2/(2 sigma_k^2))."""
    return torch.sqrt(bw * multipliers(n_kernels, mul_factor) / 2.0)


def sqdist(A, B):
    return torch.cdist(A, B, p=2) ** 2


def multi_kernel(A, B, bw, n_kernels=5, mul_factor=2.0):
    d2 = sqdist(A, B)
    m = multipliers(n_kernels, mul_factor).to(d2)
    return torch.exp(-d2[None] / (bw * m[:, None, None])).sum(0)


class _Base:
    name = "base"
    # hardware-independent per-call cost model, filled by subclasses
    def cost_per_call(self, n):
        raise NotImplementedError

    def loss(self, X):
        raise NotImplementedError

    def __call__(self, X):
        return self.loss(X)

    def resample(self, seed):
        """Redraw any randomness (features / subsets). No-op for exact ones."""
        return self


# --------------------------------------------------------------------------
class Reference(_Base):
    """Repository ``MMDLoss`` exactly as used by ``_guided.py``."""
    name = "reference"

    def __init__(self, Y, bw, n_kernels=5, mul_factor=2.0):
        self.Y = Y.detach().to(DT)
        self.bw = float(bw)
        self.m, self.d = self.Y.shape
        self.nk = n_kernels
        self.mmd = MMDLoss(kernel=RBF(n_kernels=n_kernels, mul_factor=mul_factor,
                                      bandwidth=self.bw, device="cpu"), device="cpu")

    def loss(self, X):
        return self.mmd(X.to(DT), self.Y)

    def cost_per_call(self, n):
        return (n + self.m) ** 2 * (self.d + self.nk)


class ReferenceCachedYY(_Base):
    """Same number as ``Reference`` but the target-target block is a cached
    constant: cost O(n (n + m) d).  This is the honest baseline an
    approximation must beat (Agent 2's exact simplification)."""
    name = "exact_cachedYY"

    def __init__(self, Y, bw, n_kernels=5, mul_factor=2.0):
        self.Y = Y.detach().to(DT)
        self.bw = float(bw)
        self.nk, self.mf = n_kernels, mul_factor
        self.m, self.d = self.Y.shape
        with torch.no_grad():
            self.yy = multi_kernel(self.Y, self.Y, self.bw, n_kernels, mul_factor).mean()

    def loss(self, X):
        X = X.to(DT)
        xx = multi_kernel(X, X, self.bw, self.nk, self.mf).mean()
        xy = multi_kernel(X, self.Y, self.bw, self.nk, self.mf).mean()
        return xx - 2 * xy + self.yy

    def cost_per_call(self, n):
        return n * (n + self.m) * (self.d + self.nk)


# --------------------------------------------------------------------------
class RFFMMD(_Base):
    """Random Fourier feature MMD^2, summed over the 5 bandwidths.

    phi_k(x) = sqrt(2/D) [cos(W x / sigma_k), sin(W x / sigma_k)]  with
    W ~ N(0, I_d) of shape (D/2, d) SHARED across bandwidths (same frequencies,
    rescaled), so the 5 approximations are coupled by common random numbers.
    MMD^2 ~= sum_k || mean phi_k(X) - mean phi_k(Y) ||^2 ; the target mean
    feature is precomputed once.  Unbiased for each k(x,y) given W, but
    MMD^2 (a quadratic) is biased upward by O(1/D) (see THEORY.md).

    ``orthogonal=True``: orthogonal random features (Yu et al. 2016) -- W is
    built from orthogonal blocks with chi-distributed row norms.  Only
    meaningful for d >= D/2-ish, i.e. CLIP-768, not d=1.
    """
    name = "rff"

    def __init__(self, Y, bw, D=256, seed=0, n_kernels=5, mul_factor=2.0,
                 orthogonal=False):
        assert D % 2 == 0, "D must be even (cos/sin pairs)"
        self.Y = Y.detach().to(DT)
        self.bw = float(bw)
        self.D, self.seed = D, seed
        self.orthogonal = orthogonal
        self.m, self.d = self.Y.shape
        self.sig = sigmas_from_bandwidth(self.bw, n_kernels, mul_factor)  # (K,)
        self.nk = n_kernels
        self.name = "orf" if orthogonal else "rff"
        self.resample(seed)

    def _draw_W(self, g):
        half = self.D // 2
        if not self.orthogonal:
            return torch.randn(half, self.d, generator=g, dtype=DT)
        blocks = []
        left = half
        while left > 0:
            G = torch.randn(self.d, self.d, generator=g, dtype=DT)
            Q, _ = torch.linalg.qr(G)
            # chi(d) row norms so that rows match Gaussian norm distribution
            S = torch.randn(self.d, self.d, generator=g, dtype=DT).norm(dim=1)
            blocks.append(S[:, None] * Q)
            left -= self.d
        return torch.cat(blocks, 0)[:half]

    def resample(self, seed):
        g = torch.Generator().manual_seed(int(seed))
        self.W = self._draw_W(g)                                  # (D/2, d)
        # scaled frequencies for each bandwidth: (K, D/2, d)
        self.Wk = torch.stack([self.W / s for s in self.sig])
        with torch.no_grad():
            self.phiY = self.features(self.Y).mean(1)             # (K, D)
        return self

    def features(self, X):
        """(K, N, D) features, sqrt(2/D) scaled so ||phi||^2 -> 1 per k."""
        proj = torch.einsum("kfd,nd->knf", self.Wk, X)             # (K, N, D/2)
        return math.sqrt(2.0 / self.D) * torch.cat([proj.cos(), proj.sin()], -1)

    def loss(self, X):
        phiX = self.features(X.to(DT)).mean(1)                    # (K, D)
        return ((phiX - self.phiY) ** 2).sum()

    def cost_per_call(self, n):
        return n * self.D * (self.d + 2) * self.nk // 2 + self.nk * self.D


# --------------------------------------------------------------------------
class NystromMMD(_Base):
    """Nystrom feature MMD^2 with L landmarks drawn from the target.

    For each bandwidth k: K_ZZ = k(Z,Z), phi(x) = K_ZZ^{-1/2} k(Z, x) (pseudo
    inverse sqrt with eigenvalue floor ``eps``).  MMD^2 ~= sum_k
    ||mean phi(X) - mean phi(Y)||^2, the target mean precomputed.  This is the
    exact MMD of the kernel projected onto span{k(z, .)} -- a deterministic,
    downward-biased approximation whose error is the part of the witness
    function outside the landmark span.
    """
    name = "nystrom"

    def __init__(self, Y, bw, L=64, seed=0, n_kernels=5, mul_factor=2.0,
                 eps=1e-10):
        self.Y = Y.detach().to(DT)
        self.bw, self.L, self.eps = float(bw), int(L), eps
        self.m, self.d = self.Y.shape
        self.nk, self.mf = n_kernels, mul_factor
        self.mult = multipliers(n_kernels, mul_factor)
        self.resample(seed)

    def resample(self, seed):
        g = torch.Generator().manual_seed(int(seed))
        idx = torch.randperm(self.m, generator=g)[: self.L]
        self.Z = self.Y[idx]
        self.Q = []          # per-bandwidth (L, r) maps so phi = k(x,Z) @ Q
        with torch.no_grad():
            d2 = sqdist(self.Z, self.Z)
            for mk in self.mult:
                K = torch.exp(-d2 / (self.bw * float(mk)))
                evals, evecs = torch.linalg.eigh(K)
                keep = evals > self.eps * evals.max()
                self.Q.append(evecs[:, keep] / evals[keep].sqrt())
            self.phiY = self._feat(self.Y)
            self.phiY = [f.mean(0) for f in self.phiY]
        return self

    def _feat(self, X):
        d2 = sqdist(X, self.Z)
        return [torch.exp(-d2 / (self.bw * float(mk))) @ q
                for mk, q in zip(self.mult, self.Q)]

    def loss(self, X):
        fx = self._feat(X.to(DT))
        return sum(((f.mean(0) - py) ** 2).sum() for f, py in zip(fx, self.phiY))

    def cost_per_call(self, n):
        return n * self.L * (self.d + self.nk * (1 + self.L))


# --------------------------------------------------------------------------
class SubsampledTargetMMD(_Base):
    """Linear-time-in-m estimator: exact XX term (n^2, cheap for n <= 32),
    cross term against B <= m targets, YY replaced by the full-target constant
    (computed once; irrelevant to the gradient anyway).

    ``mode='fixed'``: one subset for the whole trajectory (common random
    numbers, = exact MMD against a smaller target set).
    ``mode='fresh'``: call ``resample`` each step -> unbiased for the cross
    term at every step but noisier (B-test style).
    """
    name = "subsample"

    def __init__(self, Y, bw, B=64, seed=0, n_kernels=5, mul_factor=2.0):
        self.Y = Y.detach().to(DT)
        self.bw, self.B = float(bw), int(B)
        self.m, self.d = self.Y.shape
        self.nk, self.mf = n_kernels, mul_factor
        with torch.no_grad():
            self.yy = multi_kernel(self.Y, self.Y, self.bw, n_kernels, mul_factor).mean()
        self.resample(seed)

    def resample(self, seed):
        g = torch.Generator().manual_seed(int(seed))
        self.Ys = self.Y[torch.randperm(self.m, generator=g)[: self.B]]
        return self

    def loss(self, X):
        X = X.to(DT)
        xx = multi_kernel(X, X, self.bw, self.nk, self.mf).mean()
        xy = multi_kernel(X, self.Ys, self.bw, self.nk, self.mf).mean()
        return xx - 2 * xy + self.yy

    def cost_per_call(self, n):
        return n * (n + self.B) * (self.d + self.nk)


# --------------------------------------------------------------------------
class SlicedW2(_Base):
    """Sliced 2-Wasserstein^2 with P FIXED unit projections.

    For each projection, X's sorted projected values are matched to the
    target quantiles at levels (i + 0.5)/n (target sorted once).  NOT an
    MMD: a different objective, included because the MNIST pipeline guides
    with (resampled-projection) SWD.  In d=1 every projection is +-identity,
    so it reduces to the 1-D W2^2.  Gradient flows through the sort.
    """
    name = "sliced_w2"

    def __init__(self, Y, bw=None, P=32, seed=0):
        self.Y = Y.detach().to(DT)
        self.m, self.d = self.Y.shape
        self.P = int(P)
        self.resample(seed)

    def resample(self, seed):
        g = torch.Generator().manual_seed(int(seed))
        if self.d == 1:
            self.proj = torch.ones(1, 1, dtype=DT)
            self.P = 1
        else:
            pr = torch.randn(self.P, self.d, generator=g, dtype=DT)
            self.proj = pr / pr.norm(dim=1, keepdim=True)
        self.Ysorted = torch.sort(self.Y @ self.proj.T, dim=0)[0]   # (m, P)
        return self

    def _target_quantiles(self, n):
        q = (torch.arange(n, dtype=DT) + 0.5) / n * (self.m - 1)
        lo = q.floor().long().clamp(max=self.m - 1)
        hi = (lo + 1).clamp(max=self.m - 1)
        w = (q - lo.to(DT))[:, None]
        return (1 - w) * self.Ysorted[lo] + w * self.Ysorted[hi]     # (n, P)

    def loss(self, X):
        X = X.to(DT)
        n = X.shape[0]
        xs = torch.sort(X @ self.proj.T, dim=0)[0]
        return ((xs - self._target_quantiles(n)) ** 2).mean()

    def cost_per_call(self, n):
        return n * self.P * (self.d + math.ceil(math.log2(max(n, 2))))


# --------------------------------------------------------------------------
class PopulationGMMMMD(_Base):
    """Exact MMD^2 between the empirical X and the POPULATION GMM target.

    MMD^2(P_X, G) = mean_ij k(x_i, x_j) - 2 mean_i mu_G(x_i) + E_{y,y'~G} k(y,y')
    with the multi-bandwidth kernel mean embedding mu_G in closed form
    (``tfg.gmm_mmd.kernel_mean_embedding_multibandwidth``) and the constant
    computed once.  Per call O(n^2 d + n K d^2) for K components -- with the
    paper's K=2 this is O(n) in the target.  The objective differs from the
    empirical one by the sampling error of the m targets (this is a change
    of target, not an approximation of the same number).
    """
    name = "population_gmm"

    def __init__(self, means, covs, weights, bw, n_kernels=5, mul_factor=2.0):
        self.means, self.covs, self.weights = means, covs, weights
        self.bw, self.nk, self.mf = float(bw), n_kernels, mul_factor
        mq, cq, wq = gmm_mmd._prep(means, covs, weights)
        self.K, self.d = mq.shape
        with torch.no_grad():
            self.yy = sum(gmm_mmd._inner(mq, cq, wq, mq, cq, wq, s)
                          for s in gmm_mmd.bandwidth_sigmas(self.bw, n_kernels, mul_factor))

    def loss(self, X):
        X = X.to(DT)
        xx = multi_kernel(X, X, self.bw, self.nk, self.mf).mean()
        mu = gmm_mmd.kernel_mean_embedding_multibandwidth(
            X, self.means, self.covs, self.weights, bandwidth=self.bw,
            n_kernels=self.nk, mul_factor=self.mf)
        return xx - 2 * mu.mean() + self.yy

    def cost_per_call(self, n):
        return n * n * (self.d + self.nk) + n * self.K * self.nk * self.d ** 2


# --------------------------------------------------------------------------
class TabulatedKME1D(_Base):
    """d = 1 only.  The cross term mean_i mu_S(x_i), mu_S(x) = mean_j k(x, y_j),
    is a 1-D function of x; tabulate it ONCE on a grid of G points (cost
    G m, or G log G by FFT convolution of a binned histogram -- pointless at
    m = 250, see THEORY.md) and evaluate by linear interpolation per call,
    cost O(n).  Gradient = interpolated finite-difference slope (O(h^2) bias).
    XX term exact (n^2)."""
    name = "tab_kme_1d"

    def __init__(self, Y, bw, G=2048, pad_sigmas=4.0, n_kernels=5, mul_factor=2.0):
        self.Y = Y.detach().to(DT).reshape(-1, 1)
        assert self.Y.shape[1] == 1
        self.bw, self.nk, self.mf, self.G = float(bw), n_kernels, mul_factor, int(G)
        self.m, self.d = self.Y.shape
        smax = float(sigmas_from_bandwidth(self.bw, n_kernels, mul_factor).max())
        lo, hi = float(self.Y.min()) - pad_sigmas * smax, float(self.Y.max()) + pad_sigmas * smax
        self.grid = torch.linspace(lo, hi, self.G, dtype=DT)
        self.h = float(self.grid[1] - self.grid[0])
        with torch.no_grad():
            self.table = multi_kernel(self.grid[:, None], self.Y, self.bw,
                                      n_kernels, mul_factor).mean(1)           # (G,)
            self.slope = torch.diff(self.table) / self.h                       # (G-1,)
            self.yy = multi_kernel(self.Y, self.Y, self.bw, n_kernels, mul_factor).mean()

    def _mu(self, x):
        """Linear interpolation with a correct autograd slope via a custom
        straight-through: value is interpolated, gradient is the cell slope."""
        xd = x.detach().reshape(-1)
        pos = ((xd - self.grid[0]) / self.h).clamp(0, self.G - 1 - 1e-12)
        i = pos.floor().long()
        w = pos - i.to(DT)
        val = (1 - w) * self.table[i] + w * self.table[i + 1]
        sl = self.slope[i]
        # value + slope * (x - x.detach()) gives d/dx = slope exactly
        return val + sl * (x.reshape(-1) - xd)

    def loss(self, X):
        X = X.to(DT)
        xx = multi_kernel(X, X, self.bw, self.nk, self.mf).mean()
        return xx - 2 * self._mu(X).mean() + self.yy

    def cost_per_call(self, n):
        return n * n * (1 + self.nk) + 8 * n


# --------------------------------------------------------------------------
def make(name, Y, bw, **kw):
    """Factory used by diagnostics / tests."""
    table = {
        "reference": Reference, "exact_cachedYY": ReferenceCachedYY,
        "rff": RFFMMD, "orf": lambda Y, bw, **k: RFFMMD(Y, bw, orthogonal=True, **k),
        "nystrom": NystromMMD, "subsample": SubsampledTargetMMD,
        "sliced_w2": SlicedW2, "tab_kme_1d": TabulatedKME1D,
    }
    return table[name](Y, bw, **kw)
