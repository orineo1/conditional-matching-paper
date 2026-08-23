"""Validation of the exact closed-form GMM L2 evaluator.

The closed form is checked three independent ways:
  1. against high-resolution numerical quadrature in 1-D,
  2. against a coarse grid quadrature in 2-D,
  3. against algebraic identities that hold regardless of the formula
     (self-distance is zero, symmetry, scaling of a pure Gaussian).
"""

import math

import pytest
import torch

from tfg.gmm_l2 import (
    gmm_l2,
    gmm_l2_squared,
    gmm_l2_squared_quadrature,
    gmm_pdf,
)


# -- 1-D: closed form vs quadrature -----------------------------------------

CASES_1D = [
    # (means_p, vars_p, w_p, means_q, vars_q, w_q, label)
    ([[0.0]], [[[1.0]]], [1.0], [[0.0]], [[[1.0]]], [1.0], "identical unit gaussians"),
    ([[0.0]], [[[1.0]]], [1.0], [[1.0]], [[[1.0]]], [1.0], "shifted"),
    ([[0.0]], [[[1.0]]], [1.0], [[0.0]], [[[4.0]]], [1.0], "different variance"),
    ([[5.0], [-5.0]], [[[0.12395]], [[0.12395]]], [0.5, 0.5],
     [[5.0], [-5.0]], [[[0.12395]], [[0.12395]]], [0.5, 0.5], "the 2D-experiment target vs itself"),
    ([[5.0], [-5.0]], [[[0.12395]], [[0.12395]]], [0.5, 0.5],
     [[4.5], [-5.2]], [[[0.15]], [[0.11]]], [0.55, 0.45], "target vs perturbed target"),
    ([[0.0], [3.0], [-3.0]], [[[1.0]], [[0.5]], [[2.0]]], [0.2, 0.5, 0.3],
     [[0.5], [2.0]], [[[1.5]], [[0.25]]], [0.7, 0.3], "3 vs 2 components"),
]


@pytest.mark.parametrize("mp,cp,wp,mq,cq,wq,label", CASES_1D)
def test_closed_form_matches_quadrature_1d(mp, cp, wp, mq, cq, wq, label):
    exact = gmm_l2_squared(mp, cp, wp, mq, cq, wq)
    quad = gmm_l2_squared_quadrature(mp, cp, wp, mq, cq, wq, n_points=400_001)
    scale = max(1.0, float(exact))
    err = abs(float(exact) - float(quad)) / scale
    assert err < 1e-6, f"{label}: exact={float(exact):.12e} quad={float(quad):.12e} rel_err={err:.3e}"


def test_quadrature_converges_to_closed_form():
    """The quadrature error must shrink as the grid refines.

    This is the check that distinguishes 'the closed form is right' from 'both
    are wrong in the same way' -- a systematically wrong closed form would not
    be approached monotonically by a refining independent integrator.
    """
    mp, cp, wp = [[0.0], [3.0]], [[[1.0]], [[0.5]]], [0.4, 0.6]
    mq, cq, wq = [[1.0]], [[[2.0]]], [1.0]
    exact = float(gmm_l2_squared(mp, cp, wp, mq, cq, wq))

    # Grids must start coarse enough to have measurable error: the trapezoid
    # rule on a smooth decaying integrand converges super-algebraically, so a
    # few thousand points already sits at the float64 noise floor and a
    # "does the error shrink" assertion would be vacuous.
    errs = []
    for n in (9, 17, 33, 65, 129):
        quad = float(gmm_l2_squared_quadrature(mp, cp, wp, mq, cq, wq, n_points=n))
        errs.append(abs(quad - exact))
    assert errs[0] > 1e-9, (
        f"coarsest grid is already exact ({errs[0]:.3e}); this test cannot "
        "distinguish convergence and must be made coarser"
    )
    assert errs[-1] < errs[0], f"quadrature did not converge: {errs}"
    assert errs[-1] < 1e-9, f"final quadrature error too large: {errs}"


# -- 2-D -------------------------------------------------------------------

def _quad_2d(mp, cp, wp, mq, cq, wq, lim=8.0, n=1400):
    ax = torch.linspace(-lim, lim, n, dtype=torch.float64)
    gx, gy = torch.meshgrid(ax, ax, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    p = gmm_pdf(pts, mp, cp, wp).reshape(n, n)
    q = gmm_pdf(pts, mq, cq, wq).reshape(n, n)
    f = (p - q) ** 2
    return torch.trapezoid(torch.trapezoid(f, ax, dim=1), ax)


def test_closed_form_matches_quadrature_2d():
    cov_a = [[[1.0, 0.3], [0.3, 0.8]]]
    cov_b = [[[0.6, -0.2], [-0.2, 1.4]], [[1.0, 0.0], [0.0, 1.0]]]
    mp, wp = [[0.0, 0.0]], [1.0]
    mq, wq = [[1.0, -0.5], [-2.0, 1.0]], [0.6, 0.4]

    exact = float(gmm_l2_squared(mp, cov_a, wp, mq, cov_b, wq))
    quad = float(_quad_2d(mp, cov_a, wp, mq, cov_b, wq))
    rel = abs(exact - quad) / max(1.0, exact)
    assert rel < 1e-5, f"exact={exact:.12e} quad={quad:.12e} rel={rel:.3e}"


# -- algebraic identities ---------------------------------------------------

def test_self_distance_is_zero():
    m, c, w = [[5.0], [-5.0]], [[[0.12395]], [[0.12395]]], [0.5, 0.5]
    assert float(gmm_l2_squared(m, c, w, m, c, w)) < 1e-24
    assert float(gmm_l2(m, c, w, m, c, w)) < 1e-12


def test_symmetry():
    mp, cp, wp = [[0.0], [2.0]], [[[1.0]], [[0.4]]], [0.3, 0.7]
    mq, cq, wq = [[1.0]], [[[2.0]]], [1.0]
    a = float(gmm_l2_squared(mp, cp, wp, mq, cq, wq))
    b = float(gmm_l2_squared(mq, cq, wq, mp, cp, wp))
    assert abs(a - b) < 1e-15


def test_single_gaussian_against_analytic():
    """For two 1-D Gaussians the L2^2 has a short independent closed form:

        1/(2 sqrt(pi s1)) + 1/(2 sqrt(pi s2))
        - 2 * N(m1 - m2; 0, s1 + s2)
    """
    m1, s1, m2, s2 = 0.7, 1.3, -0.4, 0.9
    expected = (1.0 / (2.0 * math.sqrt(math.pi * s1))
                + 1.0 / (2.0 * math.sqrt(math.pi * s2))
                - 2.0 * math.exp(-((m1 - m2) ** 2) / (2.0 * (s1 + s2)))
                / math.sqrt(2.0 * math.pi * (s1 + s2)))
    got = float(gmm_l2_squared([[m1]], [[[s1]]], [1.0], [[m2]], [[[s2]]], [1.0]))
    assert abs(got - expected) < 1e-14, f"got={got!r} expected={expected!r}"


def test_weights_are_normalised():
    m, c = [[0.0], [3.0]], [[[1.0]], [[1.0]]]
    a = float(gmm_l2_squared(m, c, [1.0, 1.0], [[1.0]], [[[1.0]]], [1.0]))
    b = float(gmm_l2_squared(m, c, [0.5, 0.5], [[1.0]], [[[1.0]]], [2.0]))
    assert abs(a - b) < 1e-15


def test_separated_mixtures_do_not_underflow():
    """Far-apart components make individual log terms very negative."""
    mp, cp, wp = [[-500.0], [500.0]], [[[1.0]], [[1.0]]], [0.5, 0.5]
    mq, cq, wq = [[-500.0], [500.0]], [[[1.0]], [[1.0]]], [0.4, 0.6]
    val = float(gmm_l2_squared(mp, cp, wp, mq, cq, wq))
    # Cross terms between the two far modes vanish, so this reduces to two
    # independent single-Gaussian comparisons with weight gaps 0.1.
    expected = 2.0 * (0.01) * (1.0 / (2.0 * math.sqrt(math.pi)))
    assert math.isfinite(val)
    assert abs(val - expected) < 1e-12, f"got={val!r} expected={expected!r}"


def test_shape_conventions_accepted():
    """The repository stores target means as (K, d, 1); that must work."""
    a = gmm_l2_squared([[[5.0]], [[-5.0]]], [[[0.12395]], [[0.12395]]], [0.5, 0.5],
                       [[0.0]], [[[1.0]]], [1.0])
    b = gmm_l2_squared([[5.0], [-5.0]], [[[0.12395]], [[0.12395]]], [0.5, 0.5],
                       [[0.0]], [[[1.0]]], [1.0])
    assert abs(float(a) - float(b)) < 1e-18


def test_differentiable_in_means():
    m = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    val = gmm_l2_squared(m, [[[1.0]]], [1.0], [[0.0]], [[[1.0]]], [1.0])
    val.backward()
    assert m.grad is not None and torch.isfinite(m.grad).all()
    assert float(m.grad) > 0, "moving the mean further away must increase L2^2"
