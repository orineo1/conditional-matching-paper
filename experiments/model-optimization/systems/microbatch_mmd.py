"""Part C -- exact micro-batched (streaming) gradient of the biased MMD^2 wrt X.

Setting: fixed target Y (m x d), samples X (n x d) produced by a differentiable
sampler X = S(theta); FIXED kernel bandwidth(s) (the repo freezes the bandwidth
to the S_G rule, _common.fixed_bandwidth, so the bandwidth does not depend on X).

    L(X) = (1/n^2) sum_{i,j} k(x_i, x_j) - (2/(n m)) sum_{i,l} k(x_i, y_l) + c_YY

Derivation.  For symmetric k, d/dx_a of the XX term is
    (1/n^2) [ sum_j d1 k(x_a, x_j) + sum_i d2 k(x_i, x_a) ] = (2/n^2) sum_j d1 k(x_a, x_j),
i.e. the gradient wrt row a only needs row a as the *differentiated* argument
and ALL rows as a *detached* second argument.  Hence with any partition of the
rows into chunks I_1..I_C,

    grad_X L = sum_c grad_{X_Ic} [ (2/n^2) sum_{i in Ic} sum_j k(x_i, sg(x_j))
                                 - (2/(n m)) sum_{i in Ic} sum_l k(x_i, y_l) ]

exactly (the diagonal i=j contributes d1 k(x,x) which equals half of
d/dx k(x,x) for symmetric k -- for the RBF it is 0 -- so the identity is exact
in general).  Each chunk materialises a (|Ic| x n) + (|Ic| x m) kernel block
instead of the (n+m)^2 stacked block the repo builds (LossFunctions.py:38-44),
so peak memory is O(c (n+m)) instead of O((n+m)^2).  The rows' gradients are
then pushed through the sampler with ONE VJP: autograd.grad(X, theta, g_X)
(or X.backward(g_X)).  The target-target block c_YY is a constant: it has no
gradient and (for a fixed bandwidth) need never be recomputed.

This file implements the chunked gradient and checks it in float64 against the
full-batch gradient of the repo's MMDLoss (LossFunctions.py) to round-off.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _setup import SIM  # noqa: E402,F401  (puts simulations/src on sys.path)

from LossFunctions import MMDLoss, RBF  # noqa: E402


def rbf_multi(D2, bandwidth, n_kernels=5, mul_factor=2.0):
    """Repo kernel (LossFunctions.RBF.forward) on a squared-distance block."""
    mult = (mul_factor ** (torch.arange(n_kernels, dtype=D2.dtype, device=D2.device) - n_kernels // 2))
    return torch.exp(-D2.unsqueeze(0) / (bandwidth * mult.view(-1, 1, 1))).sum(0)


def sqdist(A, B):
    """Exact broadcast squared distances (no cdist/mm rounding path)."""
    return ((A.unsqueeze(1) - B.unsqueeze(0)) ** 2).sum(-1)


def mmd2_grad_chunked(X, Y, bandwidth, chunk):
    """Exact dL/dX for the repo's biased MMD^2 with fixed bandwidth, computed
    chunk-by-chunk over the rows of X.  X may carry history (e.g. be the output
    of a sampler): the returned tensor is the cotangent to feed into one VJP.
    Peak kernel memory: chunk x (n + m) x n_kernels."""
    n, m = X.shape[0], Y.shape[0]
    Xd = X.detach()
    g = torch.zeros_like(Xd)
    for s in range(0, n, chunk):
        xc = Xd[s:s + chunk].clone().requires_grad_(True)
        lc = ((2.0 / n ** 2) * rbf_multi(sqdist(xc, Xd), bandwidth).sum()
              - (2.0 / (n * m)) * rbf_multi(sqdist(xc, Y), bandwidth).sum())
        g[s:s + chunk], = torch.autograd.grad(lc, xc)
    return g


def mmd2_value_chunked(X, Y, bandwidth, chunk, c_yy=None):
    """Value (no graph), chunked the same way; c_yy may be cached across calls."""
    n, m = X.shape[0], Y.shape[0]
    with torch.no_grad():
        if c_yy is None:
            c_yy = rbf_multi(sqdist(Y, Y), bandwidth).mean()
        xx = xy = 0.0
        for s in range(0, n, chunk):
            xc = X[s:s + chunk]
            xx = xx + rbf_multi(sqdist(xc, X), bandwidth).sum()
            xy = xy + rbf_multi(sqdist(xc, Y), bandwidth).sum()
        return xx / n ** 2 - 2 * xy / (n * m) + c_yy, c_yy


def test(dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    d, m = 2, 250
    Y = torch.randn(m, d, dtype=dtype) * 2 + 1
    bw = float((sqdist(Y, Y).sum() / (m * m - m)))
    mmd = MMDLoss(kernel=RBF(bandwidth=bw))
    rows = []
    for n in (8, 32, 256, 1024):
        # a toy differentiable "sampler": X = theta * Z + b(theta)
        theta = torch.randn(d, dtype=dtype).requires_grad_(True)
        Z = torch.randn(n, d, dtype=dtype)
        X = Z * theta + theta.sum()
        # full-batch reference (repo MMDLoss, 258..1274-row stacked kernel)
        L = mmd(X, Y)
        g_full_X, = torch.autograd.grad(L, X, retain_graph=True)
        g_full_theta, = torch.autograd.grad(L, theta)
        # exact broadcast full-batch (no torch.cdist mm-path rounding)
        Xe = X.detach().clone().requires_grad_(True)
        Ze = torch.cat([Xe, Y]); Ke = rbf_multi(sqdist(Ze, Ze), bw)
        Le = Ke[:n, :n].mean() - 2 * Ke[:n, n:].mean() + Ke[n:, n:].mean()
        g_exact_X, = torch.autograd.grad(Le, Xe)
        for chunk in (1, 4, 64, n):
            X2 = Z * theta + theta.sum()
            gX = mmd2_grad_chunked(X2, Y, bw, chunk)
            g_theta, = torch.autograd.grad(X2, theta, grad_outputs=gX)   # one VJP through the sampler
            val, _ = mmd2_value_chunked(X2.detach(), Y, bw, chunk)
            rows.append((n, chunk, float((gX - g_full_X).abs().max()),
                         float((g_theta - g_full_theta).abs().max()), float(abs(val - L.detach())),
                         float(g_full_X.abs().max()), float((gX - g_exact_X).abs().max())))
    return rows


if __name__ == "__main__":
    print("columns: vs repo MMDLoss (torch.cdist mm path) | vs exact broadcast kernel")
    print(f"{'n':>5} {'chunk':>5} {'|dgX|repo':>12} {'|dg_theta|':>12} {'|dL|':>12} {'|gX|max':>10} {'|dgX|exact':>12}")
    worst = worst_e = 0.0
    for n, c, a, b, v, s, e in test():
        worst = max(worst, a / s, b, v); worst_e = max(worst_e, e / s)
        print(f"{n:>5} {c:>5} {a:>12.3e} {b:>12.3e} {v:>12.3e} {s:>10.3e} {e:>12.3e}")
    print("float64 worst relative discrepancy vs repo MMDLoss:", f"{worst:.3e}",
          "(this is torch.cdist's own mm-path rounding, see BENCH.md)")
    print("float64 worst relative discrepancy vs exact broadcast kernel:", f"{worst_e:.3e}")
    r32 = test(torch.float32)
    print("float32 worst |dgX|/|gX| vs repo:", f"{max(a / s for _, _, a, _, _, s, _ in r32):.3e}")
