"""Analytic oracle for the 2-D Gaussian experiment.

For a jointly Gaussian mixture P(X, Y), the conditional P(Y | X = x) is itself
a Gaussian mixture in closed form, and the squared L2 distance between two
Gaussian mixtures is closed form too (``tfg/gmm_l2.py``).  Composing the two
gives the **population objective** and its exact gradient with no sampling and
no trained model at all:

    L(x) = || P(Y | X = x) - G ||_2^2

This is the ground truth against which sampled gradient estimators are measured
in the estimator study.  Two properties make it the right reference:

  * it is exact, not a Monte-Carlo estimate, so it has no variance;
  * it is differentiable in ``x`` through both the conditional means/covariances
    AND the conditional mixing weights.

That second point matters.  ``dist_utils.generate_mog_samples`` uses a
Gumbel-softmax at ``tau=7.5`` over the mixing weights followed by
``torch.multinomial``; the component choice is not reparameterised, so the
sampled path carries essentially no gradient through the weights.  The
oracle here does, because ``compute_alpha`` is differentiated directly.  The
sampled Gumbel-softmax path must therefore NOT be used as ground truth --
see ``tests/test_oracle.py::test_gumbel_softmax_path_is_biased``.

Conditioning convention, verified against the repository: ``split_mean_cov``
(dist_utils.py:322) sets ``cond_indices = range(0, len(x_cond))``, so the
conditioning variable X occupies the FIRST coordinates and Y the rest.  This
is consistent with ``Optimization.py:118`` slicing ``[:, condition_on:]``, and
is confirmed numerically: at x* = -5 the analytic conditional variance is
0.2 - 0.195^2/0.5 = 0.12395, exactly the ``target_variances`` recorded in
``SimulationParameters/mog_2d_full.txt``.
"""

import os
import sys

import torch

# parents[1] of this file is simulations/src, where dist_utils.py lives
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import dist_utils  # noqa: E402

from tfg.gmm_l2 import gmm_l2_squared, gmm_pdf  # noqa: E402

PARAMS_2D = os.path.join(os.path.dirname(_ROOT), "params",
                         "2D_cond_1D_gmm_params.pt")


def load_params(path=PARAMS_2D, dtype=torch.float64):
    """Load one of the paper's canonical GMM parameter files.

    ``simulations/params/{2D,5D,10D}_cond_1D_gmm_params.pt`` are the authoritative
    joint-distribution definitions referenced by the paper. Each holds
    ``mu_list, Sigma_list, alpha`` (the joint P(X,Y)) and
    ``mog_means, mog_variances, weights`` (the target G(Y) = P(Y | X = x*)),
    plus ``x_star``.

    Note dim(Y) = 1 in ALL of them, including 5D and 10D: those settings scale
    dim(X), not the dimension of the distribution being matched.
    """
    path = str(path)
    if path.endswith(".txt"):
        # Legacy text format (exec'd, as oz/generate_data.py does). Retained so
        # the two parameter files can be compared through identical code.
        local = {}
        with open(path) as fh:
            exec(fh.read(), {"torch": torch}, local)
        return {
            "mu_list": [m.to(dtype) for m in local["mu_list"]],
            "Sigma_list": [s_.to(dtype) for s_ in local["Sigma_list"]],
            "alpha": local["alpha"].to(dtype),
            "target_means": local["target_means"].to(dtype),
            "target_variances": [v.to(dtype) for v in local["target_variances"]],
            "target_weights": local["target_weights"].to(dtype),
            "x_star": torch.tensor([-5.0], dtype=dtype),
            "source": path,
        }
    blob = torch.load(path, map_location="cpu", weights_only=False)
    tv = blob["mog_variances"]
    return {
        "mu_list": [m.to(dtype) for m in blob["mu_list"]],
        "Sigma_list": [s.to(dtype) for s in blob["Sigma_list"]],
        "alpha": blob["alpha"].to(dtype),
        "target_means": blob["mog_means"].to(dtype),
        "target_variances": [v.to(dtype) for v in tv],
        "target_weights": blob["weights"].to(dtype),
        "x_star": blob["x_star"].to(dtype),
        "source": str(path),
    }


def conditional_params(x, mu_list, Sigma_list, alpha):
    """Closed-form ``P(Y | X = x)`` as a Gaussian mixture.

    Parameters
    ----------
    x:
        Conditioning value, shape ``(d_x,)``. Differentiable.

    Returns
    -------
    ``(means, covs, weights)`` with shapes ``(K, d_y)``, ``(K, d_y, d_y)``,
    ``(K,)``. All differentiable in ``x``.
    """
    x = x.reshape(-1)
    means, covs = dist_utils.compute_conditionals(mu_list, Sigma_list, x)
    weights = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x)
    # compute_conditionals returns (K, d_y, 1); flatten the trailing axis.
    means = means.reshape(means.shape[0], -1)
    return means, covs, weights


def population_l2_squared(x, params):
    """Exact ``|| P(Y|X=x) - G ||_2^2``. Differentiable in ``x``, no sampling."""
    means, covs, weights = conditional_params(
        x, params["mu_list"], params["Sigma_list"], params["alpha"])
    tgt_cov = torch.stack([v for v in params["target_variances"]])
    return gmm_l2_squared(means, covs, weights,
                          params["target_means"], tgt_cov, params["target_weights"])


def population_grad(x, params):
    """Exact gradient of :func:`population_l2_squared` via autograd."""
    x = x.detach().clone().requires_grad_(True)
    val = population_l2_squared(x, params)
    grad, = torch.autograd.grad(val, x)
    return val.detach(), grad


def finite_difference_grad(fn, x, h=1e-5):
    """Central finite differences of a scalar function of ``x``."""
    x = x.detach().reshape(-1)
    out = torch.zeros_like(x)
    for i in range(x.numel()):
        xp, xm = x.clone(), x.clone()
        xp[i] += h
        xm[i] -= h
        out[i] = (fn(xp) - fn(xm)) / (2.0 * h)
    return out


# ---------------------------------------------------------------------------
# Independent validation paths (used by tests; never by the hot path)
# ---------------------------------------------------------------------------

def joint_pdf(points, mu_list, Sigma_list, alpha):
    """Joint ``P(X, Y)`` density, built directly from the component list."""
    means = torch.stack([m.reshape(-1) for m in mu_list])
    covs = torch.stack([s for s in Sigma_list])
    return gmm_pdf(points, means, covs, alpha)


def brute_force_conditional_pdf(y_grid, x_value, mu_list, Sigma_list, alpha):
    """``P(Y|X=x)`` by normalising the joint on a grid.

    Completely independent of ``compute_conditionals``/``compute_alpha``: it
    only evaluates the joint mixture density and divides by its integral. Used
    to check the analytic conditional rather than assuming it.
    """
    y_grid = y_grid.reshape(-1)
    pts = torch.stack([torch.full_like(y_grid, float(x_value)), y_grid], dim=1)
    joint = joint_pdf(pts, mu_list, Sigma_list, alpha)
    norm = torch.trapezoid(joint, y_grid)
    return joint / norm


def l2_squared_by_quadrature(x_value, params, lo=-12.0, hi=12.0, n=400_001):
    """``|| P(Y|X=x) - G ||_2^2`` by quadrature on the brute-force conditional.

    Shares no code with :func:`population_l2_squared` beyond the Gaussian
    density evaluation, so agreement between the two is meaningful.
    """
    y = torch.linspace(lo, hi, n, dtype=torch.float64)
    p = brute_force_conditional_pdf(y, x_value, params["mu_list"],
                                    params["Sigma_list"], params["alpha"])
    tgt_cov = torch.stack([v for v in params["target_variances"]])
    q = gmm_pdf(y.reshape(-1, 1), params["target_means"], tgt_cov,
                params["target_weights"])
    return torch.trapezoid((p - q) ** 2, y)
