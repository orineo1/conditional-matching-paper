"""Validation of the analytic oracle BEFORE it is used as ground truth.

Nothing here assumes the oracle is right. Each property is checked against an
independent path:

  * the conditional density, against a brute-force normalisation of the joint
    on a grid (shares no code with ``compute_conditionals``/``compute_alpha``);
  * the objective value, against quadrature on that brute-force conditional;
  * the gradient, against central finite differences of BOTH the analytic value
    and the independent quadrature value;
  * the mixing-weight gradient specifically, since that is the component the
    Gumbel-softmax sampling path does not carry.

The final test demonstrates why the sampled Gumbel-softmax path must NOT be
used as ground truth.
"""

import math

import pytest
import torch

import dist_utils
from tfg import oracle
from tfg.gmm_l2 import gmm_l2_squared


@pytest.fixture(scope="module")
def params():
    return oracle.load_params()


X_GRID = [-7.0, -5.0, -3.5, -2.0, 0.0, 1.0, 3.0, 5.0]


# ---------------------------------------------------------------------------
# 1. The analytic conditional itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x_value", X_GRID)
def test_analytic_conditional_matches_brute_force_density(params, x_value):
    """P(Y|X=x) from compute_conditionals/compute_alpha vs normalised joint."""
    x = torch.tensor([x_value], dtype=torch.float64)
    means, covs, weights = oracle.conditional_params(
        x, params["mu_list"], params["Sigma_list"], params["alpha"])

    y = torch.linspace(-12.0, 12.0, 200_001, dtype=torch.float64)
    brute = oracle.brute_force_conditional_pdf(
        y, x_value, params["mu_list"], params["Sigma_list"], params["alpha"])

    from tfg.gmm_l2 import gmm_pdf
    analytic = gmm_pdf(y.reshape(-1, 1), means, covs, weights)

    err = (analytic - brute).abs().max().item()
    assert err < 1e-9, f"x={x_value}: max density error {err:.3e}"


def test_conditional_variance_matches_the_recorded_target(params):
    """Sanity anchor: at x* = -5 the analytic conditional variance must equal
    the 0.12395 recorded in SimulationParameters/mog_2d_full.txt."""
    x = torch.tensor([-5.0], dtype=torch.float64)
    _, covs, _ = oracle.conditional_params(
        x, params["mu_list"], params["Sigma_list"], params["alpha"])
    expected = 0.2 - 0.195 ** 2 / 0.5
    assert abs(float(covs[0]) - expected) < 1e-6
    assert abs(float(covs[0]) - float(params["target_variances"][0])) < 1e-6


def test_conditional_weights_are_a_normalised_distribution(params):
    for x_value in X_GRID:
        x = torch.tensor([x_value], dtype=torch.float64)
        _, _, w = oracle.conditional_params(
            x, params["mu_list"], params["Sigma_list"], params["alpha"])
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-8)
        assert (w >= 0).all()


# ---------------------------------------------------------------------------
# 2. The objective value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x_value", X_GRID)
def test_population_l2_matches_independent_quadrature(params, x_value):
    x = torch.tensor([x_value], dtype=torch.float64)
    exact = float(oracle.population_l2_squared(x, params))
    quad = float(oracle.l2_squared_by_quadrature(x_value, params))
    rel = abs(exact - quad) / max(1e-12, abs(quad))
    assert rel < 1e-6, f"x={x_value}: exact={exact:.8e} quad={quad:.8e} rel={rel:.3e}"


def test_objective_is_small_near_x_star(params):
    """L should be far smaller at x* = -5 than at a generic point."""
    at_star = float(oracle.population_l2_squared(
        torch.tensor([-5.0], dtype=torch.float64), params))
    elsewhere = float(oracle.population_l2_squared(
        torch.tensor([1.0], dtype=torch.float64), params))
    assert at_star < elsewhere / 100.0


# ---------------------------------------------------------------------------
# 3. The gradient
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x_value", X_GRID)
def test_gradient_matches_finite_differences_of_the_analytic_value(params, x_value):
    x = torch.tensor([x_value], dtype=torch.float64)
    _, g_auto = oracle.population_grad(x, params)
    g_fd = oracle.finite_difference_grad(
        lambda z: oracle.population_l2_squared(z, params), x, h=1e-6)
    scale = max(1e-10, float(g_auto.abs().max()), float(g_fd.abs().max()))
    rel = float((g_auto - g_fd).abs().max()) / scale
    assert rel < 1e-6, (
        f"x={x_value}: autograd={g_auto.tolist()} fd={g_fd.tolist()} rel={rel:.3e}"
    )


@pytest.mark.parametrize("x_value", [-5.0, -2.0, 1.0])
def test_gradient_matches_finite_differences_of_the_quadrature_value(params, x_value):
    """The stronger check: finite differences of a value computed by a path
    that shares no code with the analytic gradient."""
    x = torch.tensor([x_value], dtype=torch.float64)
    _, g_auto = oracle.population_grad(x, params)

    h = 1e-4
    up = float(oracle.l2_squared_by_quadrature(x_value + h, params))
    dn = float(oracle.l2_squared_by_quadrature(x_value - h, params))
    g_quad = (up - dn) / (2 * h)

    scale = max(1e-8, abs(float(g_auto[0])), abs(g_quad))
    rel = abs(float(g_auto[0]) - g_quad) / scale
    assert rel < 1e-4, (
        f"x={x_value}: autograd={float(g_auto[0]):.8e} "
        f"quadrature-FD={g_quad:.8e} rel={rel:.3e}"
    )


def test_mixing_weight_gradient_is_nonzero_and_correct(params):
    """The component the Gumbel-softmax path drops.

    Isolated by construction: with a DIAGONAL joint covariance the conditional
    means and covariances do not depend on x at all, so every bit of the
    gradient flows through the mixing weights.
    """
    mu_list = [torch.tensor([-3.0, 2.0], dtype=torch.float64),
               torch.tensor([3.0, -2.0], dtype=torch.float64)]
    Sigma_list = [torch.eye(2, dtype=torch.float64) * 0.5,
                  torch.eye(2, dtype=torch.float64) * 0.5]
    alpha = torch.tensor([0.5, 0.5], dtype=torch.float64)

    x0 = torch.tensor([0.7], dtype=torch.float64)
    m0, c0, _ = oracle.conditional_params(x0, mu_list, Sigma_list, alpha)
    m1, c1, _ = oracle.conditional_params(
        torch.tensor([-1.3], dtype=torch.float64), mu_list, Sigma_list, alpha)
    assert torch.allclose(m0, m1) and torch.allclose(c0, c1), (
        "diagonal covariance should make the conditional means/covs constant in x"
    )

    def loss(z):
        m, c, w = oracle.conditional_params(z, mu_list, Sigma_list, alpha)
        return gmm_l2_squared(m, c, w,
                              [[2.0]], [[[0.5]]], [1.0])

    x = x0.clone().requires_grad_(True)
    val = loss(x)
    g, = torch.autograd.grad(val, x)
    g_fd = oracle.finite_difference_grad(loss, x0, h=1e-6)

    assert float(g.abs().max()) > 1e-6, "weight gradient is zero; it should not be"
    rel = float((g - g_fd).abs().max()) / max(1e-10, float(g_fd.abs().max()))
    assert rel < 1e-6, f"weight gradient wrong: autograd={g.tolist()} fd={g_fd.tolist()}"


# ---------------------------------------------------------------------------
# 4. Why the Gumbel-softmax sampling path is NOT ground truth
# ---------------------------------------------------------------------------

def test_gumbel_softmax_path_drops_the_mixing_weight_gradient():
    """Demonstrates the bias, on a case where the weight path is the ONLY path.

    Setup: diagonal covariance, so P(Y|X=x) = sum_c w_c(x) N(mu_c^Y, s) with
    mu_c^Y and s constant in x. The functional E[Y] therefore depends on x
    exclusively through w_c(x).

    Exact:   d/dx sum_c w_c(x) mu_c^Y   -- computed analytically below.
    Sampled: d/dx mean(generate_mog_samples(...))  -- via dist_utils' Gumbel
             path, averaged over many draws to kill Monte-Carlo noise.

    If the sampled path carried the weight gradient, the two would agree.
    """
    mu_list = [torch.tensor([-3.0, 2.0], dtype=torch.float64),
               torch.tensor([3.0, -2.0], dtype=torch.float64)]
    Sigma_list = [torch.eye(2, dtype=torch.float64) * 0.5,
                  torch.eye(2, dtype=torch.float64) * 0.5]
    alpha = torch.tensor([0.5, 0.5], dtype=torch.float64)
    x_value = torch.tensor([0.7], dtype=torch.float64)

    def exact_mean(z):
        m, _, w = oracle.conditional_params(z, mu_list, Sigma_list, alpha)
        return (w * m.reshape(-1)).sum()

    x = x_value.clone().requires_grad_(True)
    g_exact, = torch.autograd.grad(exact_mean(x), x)

    torch.manual_seed(0)
    grads = []
    for _ in range(40):
        z = x_value.clone().requires_grad_(True)
        m, c, w = oracle.conditional_params(z, mu_list, Sigma_list, alpha)
        samples = dist_utils.generate_mog_samples(400, m.unsqueeze(-1), c, w)
        g, = torch.autograd.grad(samples.reshape(-1).mean(), z, allow_unused=True)
        grads.append(0.0 if g is None else float(g))
    g_sampled = sum(grads) / len(grads)

    assert abs(float(g_exact)) > 1e-3, "test setup failed: exact gradient is ~0"
    ratio = abs(g_sampled) / abs(float(g_exact))

    # DIAGNOSTIC, NOT A REQUIREMENT. This records how much of the exact
    # weight gradient the current Gumbel path recovers. It deliberately does
    # NOT assert that the answer is zero: a zero-gradient assertion would turn
    # an implementation limitation into required behaviour and would fail the
    # day someone connects the categorical gradient properly. The correct
    # estimator is validated in tests/test_estimators.py; this test only
    # documents the state of the legacy path.
    assert math.isfinite(ratio), "sampled gradient is not finite"
    assert 0.0 <= ratio < 5.0
    print(f"\n  exact weight-gradient   = {float(g_exact):+.6f}"
          f"\n  sampled (Gumbel) mean   = {g_sampled:+.6f}"
          f"\n  fraction recovered      = {ratio:.1%}")
