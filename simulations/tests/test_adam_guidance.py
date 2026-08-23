"""AdamDPS transcription tests (arXiv:2603.16797v2, Eqs. 10-11, Algorithm 1)."""
import pytest
import torch

from tfg.adam_guidance import AdamGuidance


def test_first_step_matches_hand_computation():
    """k=1: bias correction makes the update exactly sign(g) * rho."""
    a = AdamGuidance(beta1=0.9, beta2=0.999, delta=0.0, rho=1.0)
    g = torch.tensor([2.0, -5.0], dtype=torch.float64)
    out = a.step(g)
    # m1 = 0.1*g, mhat = m1/(1-0.9) = g ; v1 = 0.001*g^2, vhat = v1/(1-0.999) = g^2
    # ghat = g / |g| = sign(g)
    assert torch.allclose(out, torch.tensor([1.0, -1.0], dtype=torch.float64), atol=1e-12)


def test_second_step_matches_hand_computation():
    a = AdamGuidance(beta1=0.9, beta2=0.999, delta=0.0, rho=1.0)
    g1 = torch.tensor([1.0], dtype=torch.float64)
    g2 = torch.tensor([3.0], dtype=torch.float64)
    a.step(g1)
    out = a.step(g2)
    b1, b2 = 0.9, 0.999
    m = b1 * ((1 - b1) * 1.0) + (1 - b1) * 3.0
    v = b2 * ((1 - b2) * 1.0) + (1 - b2) * 9.0
    m_hat = m / (1 - b1 ** 2)
    v_hat = v / (1 - b2 ** 2)
    assert float(out) == pytest.approx(m_hat / (v_hat ** 0.5), rel=1e-12)


def test_bias_correction_uses_the_adam_step_counter():
    a = AdamGuidance()
    for i in range(1, 6):
        a.step(torch.ones(1, dtype=torch.float64))
        assert a.k == i


def test_rho_scales_the_output_linearly():
    g = torch.tensor([0.7, -0.2], dtype=torch.float64)
    a1 = AdamGuidance(rho=1.0); a2 = AdamGuidance(rho=2.5)
    o1 = a1.step(g.clone()); o2 = a2.step(g.clone())
    assert torch.allclose(2.5 * o1, o2, atol=1e-12)


def test_elementwise_normalisation_is_per_coordinate():
    """Coordinates with very different scales are normalised independently."""
    a = AdamGuidance(delta=0.0)
    out = a.step(torch.tensor([1e-6, 1e6], dtype=torch.float64))
    assert torch.allclose(out.abs(), torch.ones(2, dtype=torch.float64), atol=1e-12)


def test_constant_gradient_converges_to_unit_magnitude():
    a = AdamGuidance(delta=0.0)
    g = torch.tensor([4.0], dtype=torch.float64)
    for _ in range(500):
        out = a.step(g)
    assert float(out) == pytest.approx(1.0, abs=1e-6)


def test_zero_beta1_gives_plain_rmsprop_like_step():
    a = AdamGuidance(beta1=0.0, beta2=0.999, delta=0.0)
    g = torch.tensor([2.0], dtype=torch.float64)
    out = a.step(g)
    assert float(out) == pytest.approx(1.0, rel=1e-12)


def test_sign_convention_is_descent():
    """step() returns what the loop SUBTRACTS, so it must share the sign of dL/dx."""
    a = AdamGuidance()
    out = a.step(torch.tensor([3.0, -3.0], dtype=torch.float64))
    assert out[0] > 0 and out[1] < 0


def test_reset_clears_state():
    a = AdamGuidance()
    a.step(torch.ones(2, dtype=torch.float64))
    a.reset()
    assert a.k == 0 and a.m is None and a.v is None


def test_delta_prevents_division_by_zero():
    a = AdamGuidance(delta=1e-8)
    out = a.step(torch.zeros(3, dtype=torch.float64))
    assert torch.isfinite(out).all() and float(out.abs().sum()) == 0.0
