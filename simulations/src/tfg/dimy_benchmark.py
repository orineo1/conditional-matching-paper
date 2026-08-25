"""Dimension-configurable Gaussian benchmark: dim(X) = 1, dim(Y) = d.

Used only by Experiment 5. The paper's own settings cannot serve this purpose --
see WHY THIS AXIS below.

WHY THIS AXIS
-------------
The CDM paper's higher-dimensional settings scale the CONDITIONING variable and
always match a one-dimensional target:

    2D : dim(X)=1, dim(Y)=1        5D : dim(X)=4, dim(Y)=1
    10D: dim(X)=9, dim(Y)=1

(verified from the paper text and from its own parameter files
``simulations/params/{2D,5D,10D}_cond_1D_gmm_params.pt``, whose ``mog_means``
is ``(2,1,1)`` in every case). MMD sample complexity lives in the dimension of
the distribution being MATCHED, i.e. dim(Y), so the paper's sweep cannot test
it. This benchmark therefore holds X scalar and scales dim(Y) = d.

CONSTRUCTION
------------
Base: the validated 2-D benchmark -- 11 components, uniform weights, shared
covariance [[0.5, 0.195], [0.195, 0.2]], ground-truth optimum x* = -5.

To reach dim(Y) = d we keep:
  * the scalar design variable X and its optimum x* = -5 (X-means untouched);
  * the uniform mixture weights;
  * the per-coordinate signal scale (the same 11 Y-locations are used in every
    coordinate);
  * the per-coordinate CONDITIONAL variance, exactly 0.12395 for every d.

The Y block is Sigma_yy = 0.12395 * I + c * 11^T with c = SIGMA_XY^2 / SIGMA_XX
= 0.07605. The Schur complement is then

    Sigma_yy - Sigma_yx Sigma_xx^{-1} Sigma_xy = 0.12395 * I

for EVERY d, so the conditional covariance is exactly the 2-D benchmark's
0.12395 per coordinate, with no cross-coordinate conditional correlation, and
the joint is positive definite at every d without any diagonal loading. At d = 1
the Y block is 0.12395 + 0.07605 = 0.2, reproducing the original benchmark.

Coordinates are NOT duplicated. For coordinate j we apply a fixed permutation
pi_j of the 11 component indices to the Y-locations, so component k sits at
``y[pi_j(k)]`` in coordinate j. Every coordinate therefore has the same marginal
geometry while the joint mixture is genuinely d-dimensional: two components
that coincide in one coordinate generally differ in another. pi_0 is the
identity, so d = 1 reproduces the original 2-D benchmark exactly.

The permutations are generated from a fixed seed and hashed into the run record.
WHY A SECOND CONSTRUCTION IS NEEDED (added 2026-08-24)
-----------------------------------------------------
The construction above deliberately holds the per-coordinate conditional
variance fixed and reuses the SAME 11 Y-locations in every coordinate. That
isolates dim(Y) from every confound, but it has a consequence that invalidates
it for the question it was built for: extra coordinates are **repeated
measurements of one scalar signal**, so the MMD estimate gets MORE stable as d
grows, not less. Measured relative gradient noise on this construction, with the
kernel bandwidth frozen so the median heuristic cannot be the cause:

    d        1     2     4     8*    16    32    64
    c(n=8)  0.63  0.13  0.85  4.20  1.02  0.25  0.63
    (* x = -2 is the d = 8 plateau, so the mean gradient is ~0 and the ratio
       is inflated; it is not a real noise level.)

No dimensional trend. This is the OPPOSITE of the curse of dimensionality that
motivates the question -- where probability mass spreads out and n samples cover
the space exponentially worse. So ``as_params`` structurally CANNOT test whether
guidance benefits from momentum at higher dim(Y): it will return nulls at every
d, and that is a property of the benchmark, not of the guidance rule.

``build_nuisance`` is the construction that can. It appends coordinates that
carry NO signal about X -- pure noise, conditionally independent of the design
variable -- so the informative subspace stays 1-D while the matched distribution
lives in R^(1+m). The n samples must then cover a space whose volume grows with
m while the signal does not, which is what makes an MMD estimate harder in high
dimension. Reported apart from the primary construction.
"""

import hashlib

import torch

BASE_Y = [5.0, -5.0, 3.0, -1.0, -3.0, 4.0, -3.0, 2.0, 1.0, 5.0, -5.0]
# Component X-locations. These MUST equal the X column of the canonical file
# params/2D_cond_1D_gmm_params.pt -- the module's central claim is that d = 1
# reproduces that benchmark exactly. Component 8 read -7.0 here against -8.0 in
# the canonical file until 2026-08-23; tests/test_dimy_benchmark_d1.py now pins
# the whole vector against the file so it cannot drift again.
BASE_X = [-5.0, -5.0, 5.0, 5.0, 0.0, -2.0, -2.0, 1.0, -8.0, 7.0, 0.0]
SIGMA_XX, SIGMA_XY, SIGMA_YY = 0.5, 0.195, 0.2
COND_VAR = SIGMA_YY - SIGMA_XY ** 2 / SIGMA_XX      # 0.12395, the 2-D value
X_STAR = -5.0
FILTER_THRESHOLD = 0.01          # as in the paper's 2-D setting


def _permutations(d, K, seed=20260821):
    """Deterministic per-coordinate permutations; coordinate 0 is the identity."""
    g = torch.Generator().manual_seed(int(seed))
    perms = [torch.arange(K)]
    for _ in range(1, d):
        perms.append(torch.randperm(K, generator=g))
    return perms


def as_params(d, seed=20260821, dtype=torch.float64, threshold=FILTER_THRESHOLD):
    """Package the benchmark in the same dict shape as ``tfg.oracle.load_params``.

    Lets Experiment 5 reuse the exact evaluation path (``oracle``,
    ``gmm_l2``) and the same guided loop as Experiments 2-4.
    """
    b = build(d, seed=seed, dtype=dtype)
    K = b["means"].shape[0]
    cm, cc, w = conditional(b, b["x_star"])
    keep = w >= threshold
    return {
        "mu_list": [b["means"][k] for k in range(K)],
        "Sigma_list": [b["cov"] for _ in range(K)],
        "alpha": b["alpha"],
        "target_means": cm[keep].detach().reshape(-1, d),
        "target_variances": [cc.detach() for _ in range(int(keep.sum()))],
        "target_weights": (w[keep] / w[keep].sum()).detach(),
        "x_star": b["x_star"],
        "source": f"dimy_benchmark(d={d}, seed={seed})",
        "dim_y": d, "perm_sha256": b["perm_sha256"],
    }


def build(d, seed=20260821, dtype=torch.float64):
    """Return the joint GMM over (X, Y) with dim(Y) = d."""
    K = len(BASE_Y)
    perms = _permutations(d, K, seed)
    means = torch.zeros(K, 1 + d, dtype=dtype)
    for k in range(K):
        means[k, 0] = BASE_X[k]
        for j in range(d):
            means[k, 1 + j] = BASE_Y[int(perms[j][k])]

    c = SIGMA_XY ** 2 / SIGMA_XX                 # 0.07605
    cov = torch.zeros(1 + d, 1 + d, dtype=dtype)
    cov[0, 0] = SIGMA_XX
    cov[0, 1:] = SIGMA_XY
    cov[1:, 0] = SIGMA_XY
    cov[1:, 1:] = c + COND_VAR * torch.eye(d, dtype=dtype)
    alpha = torch.full((K,), 1.0 / K, dtype=dtype)
    return {"means": means, "cov": cov, "alpha": alpha, "d": d,
            "x_star": torch.tensor([X_STAR], dtype=dtype),
            "perm_sha256": hashlib.sha256(
                torch.stack(perms).numpy().tobytes()).hexdigest(),
            "seed": seed}


def build_nuisance(m, seed=20260821, dtype=torch.float64, nuisance_var=None):
    """dim(Y) = 1 + m, where only coordinate 0 carries signal about X.

    Coordinate 0 is exactly the validated 2-D benchmark's Y. The m appended
    coordinates are conditionally independent of X with the same marginal scale,
    so they add volume without adding information: the MMD must be estimated over
    a (1 + m)-dimensional space from the same n samples, while the part that
    determines the optimum stays one-dimensional. ``m = 0`` reproduces
    ``build(1)`` exactly.

    Contrast with ``build``, where every coordinate carries the same signal and
    extra dimensions therefore REDUCE estimator noise. See the module docstring.
    """
    K = len(BASE_Y)
    d = 1 + int(m)
    var = float(SIGMA_YY if nuisance_var is None else nuisance_var)
    means = torch.zeros(K, 1 + d, dtype=dtype)
    for k in range(K):
        means[k, 0] = BASE_X[k]
        means[k, 1] = BASE_Y[k]          # the one informative coordinate
        # coordinates 2.. are left at 0 for every component: no component
        # structure, hence no information about which component (or which x).

    cov = torch.zeros(1 + d, 1 + d, dtype=dtype)
    cov[0, 0] = SIGMA_XX
    cov[0, 1] = SIGMA_XY                 # X couples ONLY to coordinate 0 of Y
    cov[1, 0] = SIGMA_XY
    cov[1, 1] = SIGMA_YY
    for j in range(2, 1 + d):
        cov[j, j] = var                  # independent of X and of each other
    alpha = torch.full((K,), 1.0 / K, dtype=dtype)
    return {"means": means, "cov": cov, "alpha": alpha, "d": d,
            "x_star": torch.tensor([X_STAR], dtype=dtype),
            "perm_sha256": "nuisance", "n_nuisance": int(m), "seed": seed}


def as_params_nuisance(m, seed=20260821, dtype=torch.float64,
                       threshold=FILTER_THRESHOLD, nuisance_var=None):
    """``as_params`` for the nuisance construction."""
    b = build_nuisance(m, seed=seed, dtype=dtype, nuisance_var=nuisance_var)
    K, d = b["means"].shape[0], b["d"]
    cm, cc, w = conditional(b, b["x_star"])
    keep = w >= threshold
    return {
        "mu_list": [b["means"][k] for k in range(K)],
        "Sigma_list": [b["cov"] for _ in range(K)],
        "alpha": b["alpha"],
        "target_means": cm[keep].detach().reshape(-1, d),
        "target_variances": [cc.detach() for _ in range(int(keep.sum()))],
        "target_weights": (w[keep] / w[keep].sum()).detach(),
        "x_star": b["x_star"],
        "source": f"dimy_benchmark.nuisance(m={m}, seed={seed})",
        "dim_y": d, "n_nuisance": int(m), "perm_sha256": "nuisance",
    }


def conditional(bench, x):
    """Closed-form P(Y | X = x) as a Gaussian mixture. Differentiable in x."""
    m, cov, alpha = bench["means"], bench["cov"], bench["alpha"]
    d = bench["d"]
    x = x.reshape(-1)[:1].to(m.dtype)

    s_xx = cov[0, 0]
    s_yx = cov[1:, 0].reshape(d, 1)
    s_yy = cov[1:, 1:]

    mu_x = m[:, 0].reshape(-1, 1)
    mu_y = m[:, 1:]
    dx = (x.reshape(1, 1) - mu_x)                       # (K,1)
    cond_mean = mu_y + (s_yx.reshape(1, d) / s_xx) * dx  # (K,d)
    cond_cov = s_yy - (s_yx @ s_yx.T) / s_xx             # (d,d), shared

    # responsibilities from the 1-D X-marginal
    logw = (torch.log(alpha) - 0.5 * torch.log(2 * torch.pi * s_xx)
            - 0.5 * dx.reshape(-1) ** 2 / s_xx)
    w = torch.softmax(logw, dim=0)
    return cond_mean, cond_cov, w


def target(bench, threshold=FILTER_THRESHOLD):
    """G(Y) = P(Y | X = x*), components below ``threshold`` filtered out."""
    cm, cc, w = conditional(bench, bench["x_star"])
    keep = w >= threshold
    return cm[keep].detach(), cc.detach(), (w[keep] / w[keep].sum()).detach()
