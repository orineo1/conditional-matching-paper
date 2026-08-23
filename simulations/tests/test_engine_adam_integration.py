"""Integrated-Adam equivalence.

Three claims, each asserted end-to-end:
  I1  engine(temporal="adam")  ==  engine(temporal="none") with the standalone
      AdamGuidance applied externally to the same rho-branch gradients.
  I2  the moments inside the engine match the OFFICIAL upstream
      adaptive_moment_estimate on the same gradient sequence.
  I3  temporal="none" is unchanged by the integration (no regression).
"""
import importlib.util
from pathlib import Path

import pytest
import torch

from conftest import ToyDenoiser, make_quadratic_log_f
from tfg.adam_guidance import AdamGuidance
from tfg.config import TFGConfig
from tfg.engine import GeneralizedTFG
from tfg.noise_tape import NoiseTape
from tfg.schedule import DiffusionSchedule
from tfg.trace import Tracer, compare_traces

UPSTREAM = Path("/Users/stolk/github/conditional-matching-paper/literature/"
                "external-code/adam-guidance/methods/adam_dps.py")
SHAPE, T = (1, 2), 8


def _setup(T=T, seed=17):
    sch = DiffusionSchedule(T=T)
    den = ToyDenoiser(d=SHAPE[1], T=T, seed=0, schedule=sch)
    return sch, (lambda x, t: den(x, t)), make_quadratic_log_f([1.5, -0.75], 0.6), seed


def _cfg(mode, T=T, **kw):
    c = TFGConfig(T=T, N_recur=1, N_iter=0, gamma_bar=0.0,
                  rho_scalar=0.5, mu_scalar=0.0, n_mc=1)
    c.temporal.mode = mode
    for k, v in kw.items():
        setattr(c.temporal, k, v)
    return c


def _official():
    if not UPSTREAM.exists():
        pytest.skip("official clone absent")
    src = UPSTREAM.read_text().replace("from .base import BaseGuidance",
                                       "BaseGuidance = object")
    spec = importlib.util.spec_from_loader("off_adam", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, str(UPSTREAM), "exec"), mod.__dict__)
    obj = mod.AdamDPSGuidance.__new__(mod.AdamDPSGuidance)
    obj.b1, obj.b2 = 0.9, 0.995
    obj.one_minus_b1, obj.one_minus_b2 = 0.1, 0.005
    obj.m, obj.v, obj.s = None, None, 1
    return obj


# -- I3: no regression on the plain path ------------------------------------

def test_temporal_none_is_unchanged():
    sch, eps, lf, seed = _setup()
    tr_a, tr_b = Tracer(), Tracer()
    GeneralizedTFG(eps, lf, sch, NoiseTape(seed=seed), _cfg("none")).run(SHAPE, trace=tr_a)
    GeneralizedTFG(eps, lf, sch, NoiseTape(seed=seed), TFGConfig(
        T=T, N_recur=1, N_iter=0, gamma_bar=0.0, rho_scalar=0.5,
        mu_scalar=0.0, n_mc=1)).run(SHAPE, trace=tr_b)
    ok, rep = compare_traces(tr_a, tr_b, atol=0.0)
    # grad_rho_used is new; compare only the pre-existing keys
    shared = {k for k in tr_a.records if k[0] != "grad_rho_used"}
    for k in shared:
        assert torch.equal(tr_a.records[k], tr_b.records[k]), k


# -- I1: integrated == standalone -------------------------------------------

def test_integrated_adam_equals_standalone():
    sch, eps, lf, seed = _setup()
    tr = Tracer()
    x_int = GeneralizedTFG(eps, lf, sch, NoiseTape(seed=seed),
                           _cfg("adam", adam_rho=0.7)).run(SHAPE, trace=tr)

    # replay the traced raw gradients through the standalone implementation
    standalone = AdamGuidance(beta1=0.9, beta2=0.995, delta=1e-8, rho=0.7,
                              inv_sqrt_alpha=False)
    raw = [tr.records[("grad_rho_raw", t, 1, None)] for t in range(T, 0, -1)]
    used = [tr.records[("grad_rho_used", t, 1, None)] for t in range(T, 0, -1)]
    for g_raw, g_used in zip(raw, used):
        assert torch.allclose(standalone.step(g_raw), g_used, rtol=0, atol=1e-14)
    assert torch.isfinite(x_int).all()


def test_integrated_adam_changes_the_trajectory():
    """Anti-vacuity: the Adam path must differ from the plain path."""
    sch, eps, lf, seed = _setup()
    a = GeneralizedTFG(eps, lf, sch, NoiseTape(seed=seed), _cfg("none")).run(SHAPE)
    b = GeneralizedTFG(eps, lf, sch, NoiseTape(seed=seed),
                       _cfg("adam", adam_rho=0.7)).run(SHAPE)
    assert (a - b).abs().max().item() > 1e-6


# -- I2: integrated moments == official upstream ----------------------------

def test_integrated_moments_match_official():
    sch, eps, lf, seed = _setup()
    tr = Tracer()
    GeneralizedTFG(eps, lf, sch, NoiseTape(seed=seed),
                   _cfg("adam", adam_rho=1.0)).run(SHAPE, trace=tr)
    off = _official()
    for t in range(T, 0, -1):
        g_raw = tr.records[("grad_rho_raw", t, 1, None)]
        g_used = tr.records[("grad_rho_used", t, 1, None)]
        # engine feeds the ASCENT direction, as upstream does
        assert torch.allclose(off.adaptive_moment_estimate(g_raw.clone()),
                              g_used, rtol=0, atol=1e-12), t


@pytest.mark.parametrize("N_recur,N_iter", [(1, 0), (2, 1)])
def test_integration_holds_across_configurations(N_recur, N_iter):
    sch, eps, lf, seed = _setup()
    c = _cfg("adam", adam_rho=0.4)
    c.N_recur, c.N_iter, c.mu_scalar = N_recur, N_iter, 0.2
    tr = Tracer()
    GeneralizedTFG(eps, lf, sch, NoiseTape(seed=seed), c).run(SHAPE, trace=tr)
    standalone = AdamGuidance(beta1=0.9, beta2=0.995, delta=1e-8, rho=0.4,
                              inv_sqrt_alpha=False)
    keys = [k for k in tr.order if k[0] == "grad_rho_used"]
    for k in keys:
        raw = tr.records[("grad_rho_raw",) + k[1:]]
        assert torch.allclose(standalone.step(raw), tr.records[k], rtol=0, atol=1e-14)


def test_unknown_temporal_mode_rejected():
    with pytest.raises(ValueError, match="unknown temporal mode"):
        _cfg("momentum").validate()
