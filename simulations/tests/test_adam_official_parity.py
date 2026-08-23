"""Parity of our AdamGuidance against the OFFICIAL adam-guidance implementation.

Upstream: https://github.com/christianbelardi/adam-guidance
commit 21f878a08279ac8399cb58c36a599c511d087fb0 (2026-08-12), clean.
File: methods/adam_dps.py, sha256 ab49ae19b5d9ccf670eed5799b503dd902f61c3ed052338477fec23455e25ffd

The upstream class is used WITHOUT modification. We bypass its heavy constructor
with ``__new__`` and set only the attributes ``adaptive_moment_estimate`` reads,
so the method under test is the upstream source verbatim.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from tfg.adam_guidance import AdamGuidance

UPSTREAM = Path("/Users/stolk/github/conditional-matching-paper/literature/"
                "external-code/adam-guidance/methods/adam_dps.py")


def _load_official():
    """Import the upstream module without its package deps."""
    if not UPSTREAM.exists():
        pytest.skip(f"official clone not present at {UPSTREAM}")
    src = UPSTREAM.read_text().replace("from .base import BaseGuidance",
                                       "BaseGuidance = object")
    spec = importlib.util.spec_from_loader("official_adam_dps", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = str(UPSTREAM)
    exec(compile(src, str(UPSTREAM), "exec"), mod.__dict__)
    return mod


def official_estimator(beta1=0.9, beta2=0.995):
    mod = _load_official()
    obj = mod.AdamDPSGuidance.__new__(mod.AdamDPSGuidance)
    obj.b1, obj.b2 = beta1, beta2
    obj.one_minus_b1, obj.one_minus_b2 = 1 - beta1, 1 - beta2
    obj.m, obj.v, obj.s = None, None, 1
    return obj


# -- the transform itself ---------------------------------------------------

@pytest.mark.parametrize("beta1,beta2", [(0.9, 0.995), (0.9, 0.999), (0.0, 0.9), (0.95, 0.99)])
def test_moment_transform_matches_official_step_by_step(beta1, beta2):
    off = official_estimator(beta1, beta2)
    ours = AdamGuidance(beta1=beta1, beta2=beta2, delta=1e-8, rho=1.0)
    torch.manual_seed(0)
    for k in range(25):
        g = torch.randn(5, dtype=torch.float64) * (10.0 ** (k % 5 - 2))
        # Official consumes the ASCENT direction (-dL/dx); ours consumes dL/dx
        # and returns the quantity to SUBTRACT. The transform is odd, so
        # ours(g) must equal -official(-g).
        o = off.adaptive_moment_estimate(-g.clone())
        u = ours.step(g.clone())
        assert torch.allclose(u, -o, rtol=0, atol=1e-12), (k, u, -o)


def test_bias_correction_counter_matches():
    off = official_estimator()
    ours = AdamGuidance(beta1=0.9, beta2=0.995, delta=1e-8)
    g = torch.ones(3, dtype=torch.float64)
    for _ in range(7):
        off.adaptive_moment_estimate(-g)
        ours.step(g)
    assert ours.k == off.s - 1, (ours.k, off.s)


@pytest.mark.parametrize("scale", [1e-8, 1e-3, 1.0, 1e3, 1e8])
def test_extreme_gradient_scales(scale):
    off = official_estimator()
    ours = AdamGuidance(beta1=0.9, beta2=0.995, delta=1e-8)
    torch.manual_seed(1)
    for _ in range(10):
        g = torch.randn(4, dtype=torch.float64) * scale
        o = off.adaptive_moment_estimate(-g.clone())
        u = ours.step(g.clone())
        assert torch.allclose(u, -o, rtol=0, atol=1e-12)


def test_scalar_and_multidimensional_shapes():
    for shape in [(1,), (3,), (2, 4), (2, 3, 5)]:
        off = official_estimator()
        ours = AdamGuidance(beta1=0.9, beta2=0.995, delta=1e-8)
        torch.manual_seed(2)
        for _ in range(6):
            g = torch.randn(*shape, dtype=torch.float64)
            assert torch.allclose(ours.step(g.clone()),
                                  -off.adaptive_moment_estimate(-g.clone()),
                                  rtol=0, atol=1e-12)


def test_official_defaults_are_what_we_use():
    """utils/configs.py: beta1=0.9, beta2=0.995, guidance_strength=1.0."""
    cfg = Path("/Users/stolk/github/conditional-matching-paper/literature/"
               "external-code/adam-guidance/utils/configs.py")
    if not cfg.exists():
        pytest.skip("official clone not present")
    text = cfg.read_text()
    assert "beta1: float = field(default=0.9)" in text
    assert "beta2: float = field(default=0.995)" in text
    assert AdamGuidance().beta1 == 0.9
    assert AdamGuidance().beta2 == 0.995, (
        "our default beta2 must match the official default 0.995")


def test_delta_matches_official_hardcoded_value():
    src = UPSTREAM.read_text() if UPSTREAM.exists() else ""
    if not src:
        pytest.skip("official clone not present")
    assert "torch.sqrt(v_hat) + 1e-8" in src
    assert AdamGuidance().delta == 1e-8


def test_official_applies_inverse_sqrt_alpha_scaling():
    """Upstream divides the guidance by sqrt(alpha_t) -- Algorithm 1 omits this."""
    src = UPSTREAM.read_text() if UPSTREAM.exists() else ""
    if not src:
        pytest.skip("official clone not present")
    assert "x_prev += guidance / alpha_t ** 0.5" in src
    assert "alpha_t = alpha_prod_t / alpha_prod_t_prev" in src
