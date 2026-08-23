"""Validation of the closed-form population MMD^2 reference.

Two independent checks, neither of which assumes the closed form is right:

  * value, against Monte-Carlo evaluation using the REPOSITORY's own
    ``LossFunctions.MMDLoss``/``RBF`` at a fixed bandwidth -- so the reference
    is tied to the actual kernel the paper uses, not to a re-typed copy;
  * gradient, against central finite differences of the closed-form value.

Fixed bandwidth throughout: a population quantity cannot depend on a sample,
so the repository's data-adaptive bandwidth has no population counterpart.
That difference is itself a finding, recorded in the module docstring of
``tfg/gmm_mmd.py``.
"""

import pytest
import torch

import dist_utils
from tfg import oracle
from tfg.gmm_mmd import (
    bandwidth_sigmas,
    population_mmd2,
    population_mmd2_multibandwidth,
)


@pytest.fixture(scope="module")
def params():
    return oracle.load_params()


def _popmmd(x, params, bw):
    m, c, w = oracle.conditional_params(
        x, params["mu_list"], params["Sigma_list"], params["alpha"])
    tc = torch.stack([v for v in params["target_variances"]])
    return population_mmd2_multibandwidth(
        m, c, w, params["target_means"], tc, params["target_weights"],
        bandwidth=bw)


def test_bandwidth_mapping_matches_the_repository_kernel():
    """sigma_k^2 = bw * mul^(k - n//2) / 2, from LossFunctions.py:19-23."""
    sig = bandwidth_sigmas(1.0)
    assert len(sig) == 5
    expected = [(2.0 ** k / 2.0) ** 0.5 for k in (-2, -1, 0, 1, 2)]
    for a, b in zip(sig, expected):
        assert a == pytest.approx(b, rel=1e-12)


def test_multibandwidth_is_the_sum_of_single_bandwidths(params):
    x = torch.tensor([-3.0], dtype=torch.float64)
    m, c, w = oracle.conditional_params(
        x, params["mu_list"], params["Sigma_list"], params["alpha"])
    tc = torch.stack([v for v in params["target_variances"]])
    total = sum(float(population_mmd2(m, c, w, params["target_means"], tc,
                                      params["target_weights"], sigma=s))
                for s in bandwidth_sigmas(1.0))
    combined = float(_popmmd(x, params, 1.0))
    assert combined == pytest.approx(total, rel=1e-12)


@pytest.mark.parametrize("x_value", [-5.0, -3.0, 1.0])
def test_closed_form_matches_repo_mmdloss_monte_carlo(x_value, params):
    """The decisive check: against LossFunctions.MMDLoss itself."""
    from LossFunctions import MMDLoss, RBF

    bw = 1.0
    x = torch.tensor([x_value], dtype=torch.float64)
    exact = float(_popmmd(x, params, bw))

    m, c, w = oracle.conditional_params(
        x, params["mu_list"], params["Sigma_list"], params["alpha"])
    tc = torch.stack([v for v in params["target_variances"]])
    mmd = MMDLoss(kernel=RBF(bandwidth=bw, device="cpu"), device="cpu")

    # Use the UNBIASED U-statistic for the comparison, built from the
    # repository's own RBF kernel matrix. The repository's MMDLoss is the
    # BIASED V-statistic, whose expectation exceeds the population MMD^2 by
    # ~2 * n_kernels / n (the kernel diagonal, k(y,y) = n_kernels). At a probe
    # point where MMD^2 is small that offset dominates the signal, so
    # comparing the V-statistic to a population value would fail for a reason
    # that has nothing to do with the closed form. The offset itself is pinned
    # in test_v_statistic_offset_matches_theory below.
    kernel = mmd.kernel
    torch.manual_seed(0)
    vals = []
    for _ in range(24):
        n_s = 3000
        a = dist_utils.generate_mog_samples_not_differentiable(
            n_s, m.unsqueeze(-1), c, w)
        b = dist_utils.generate_mog_samples_not_differentiable(
            n_s, params["target_means"], tc, params["target_weights"])
        K = kernel(torch.vstack([a, b]))
        Kxx, Kyy, Kxy = K[:n_s, :n_s], K[n_s:, n_s:], K[:n_s, n_s:]
        xx = (Kxx.sum() - Kxx.diagonal().sum()) / (n_s * (n_s - 1))
        yy = (Kyy.sum() - Kyy.diagonal().sum()) / (n_s * (n_s - 1))
        vals.append(float(xx - 2 * Kxy.mean() + yy))

    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    se = (var / len(vals)) ** 0.5
    z = abs(mean - exact) / se
    assert z < 4.0, (
        f"x={x_value}: closed form {exact:.8f} vs repo-MMDLoss MC "
        f"{mean:.8f} +- {se:.8f} (z={z:.2f})"
    )


@pytest.mark.parametrize("bw", [0.5, 1.0, 4.0])
@pytest.mark.parametrize("x_value", [-7.0, -5.0, -3.5, 0.0, 3.0])
def test_reference_gradient_matches_finite_differences(bw, x_value, params):
    x = torch.tensor([x_value], dtype=torch.float64, requires_grad=True)
    val = _popmmd(x, params, bw)
    g, = torch.autograd.grad(val, x)

    h = 1e-6
    up = float(_popmmd(torch.tensor([x_value + h], dtype=torch.float64), params, bw))
    dn = float(_popmmd(torch.tensor([x_value - h], dtype=torch.float64), params, bw))
    fd = (up - dn) / (2 * h)

    diff = abs(float(g[0]) - fd)
    rel = diff / max(1e-12, abs(fd))
    # Near a stationary point the gradient is ~1e-5 and central differences with
    # h=1e-6 cannot resolve it to 1e-6 RELATIVE accuracy -- the FD truncation
    # error alone is larger. Accept either a tight relative match or an
    # absolute match at the level FD can actually deliver.
    assert rel < 1e-6 or diff < 1e-8, (
        f"x={x_value}, bw={bw}: autograd={float(g[0]):.8e} fd={fd:.8e} "
        f"rel={rel:.2e} abs={diff:.2e}"
    )


def test_mmd_reference_is_not_the_l2_objective(params):
    """Guard against conflating the guidance objective with the quality metric.

    Sample MMD is what guidance optimises; exact GMM L2 is only the final
    solution-quality metric. Their gradients are different functions and must
    never be compared as if one were ground truth for the other.
    """
    x = torch.tensor([-3.0], dtype=torch.float64)
    _, g_l2 = oracle.population_grad(x, params)
    xg = x.clone().requires_grad_(True)
    g_mmd, = torch.autograd.grad(_popmmd(xg, params, 1.0), xg)
    assert abs(float(g_l2[0]) - float(g_mmd[0])) > 1e-3, (
        "the L2 and MMD population gradients happen to coincide here; pick a "
        "different probe point for this guard"
    )


def test_v_statistic_offset_matches_theory(params):
    """Pin the exact V-statistic offset against theory.

    Writing U for the unbiased blocks,

        XX_V = (1 - 1/n) U_xx + k(y,y)/n ,   likewise YY_V,

    since k(y, y) = n_kernels for every y (each summed RBF contributes
    exp(0) = 1). The cross block is unaffected. Hence

        E[V] - MMD^2 = ( 2*n_kernels - E_pp[k] - E_qq[k] ) / n

    NOT 2*n_kernels/n -- the (1 - 1/n) rescaling of the off-diagonal mass
    cancels part of the diagonal contribution. Both E_pp[k] and E_qq[k] come
    from the same closed form under test, so this is a joint check on the
    kernel expectations as well as on the offset.

    This is a VALUE bias only: it is constant in x, so it does not bias the
    gradient additively. But it is large enough to dominate near the optimum,
    where MMD^2 itself is ~5e-4.
    """
    from LossFunctions import MMDLoss, RBF

    bw, n_s, n_kernels = 1.0, 500, 5
    x = torch.tensor([-5.0], dtype=torch.float64)
    exact = float(_popmmd(x, params, bw))
    m, c, w = oracle.conditional_params(
        x, params["mu_list"], params["Sigma_list"], params["alpha"])
    tc = torch.stack([v for v in params["target_variances"]])
    mmd = MMDLoss(kernel=RBF(bandwidth=bw, device="cpu"), device="cpu")

    torch.manual_seed(1)
    vals = []
    for _ in range(24):
        a = dist_utils.generate_mog_samples_not_differentiable(
            n_s, m.unsqueeze(-1), c, w)
        b = dist_utils.generate_mog_samples_not_differentiable(
            n_s, params["target_means"], tc, params["target_weights"])
        vals.append(float(mmd(a, b)))
    mean = sum(vals) / len(vals)

    # E_pp[k] and E_qq[k] from the closed form, summed over the bandwidths.
    from tfg.gmm_mmd import _inner, _prep, bandwidth_sigmas
    mp, cp, wp = _prep(m, c, w)
    mq, cq, wq = _prep(params["target_means"], tc, params["target_weights"])
    e_pp = sum(float(_inner(mp, cp, wp, mp, cp, wp, sg))
               for sg in bandwidth_sigmas(bw))
    e_qq = sum(float(_inner(mq, cq, wq, mq, cq, wq, sg))
               for sg in bandwidth_sigmas(bw))

    predicted_offset = (2.0 * n_kernels - e_pp - e_qq) / n_s
    observed_offset = mean - exact
    assert observed_offset == pytest.approx(predicted_offset, rel=0.15), (
        f"observed V-statistic offset {observed_offset:.6f} vs predicted "
        f"{predicted_offset:.6f} (population MMD^2 = {exact:.6f})"
    )
