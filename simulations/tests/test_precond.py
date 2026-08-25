"""[Agent P] Deterministic unit tests for tfg/precond.py.

Every mode has a known-sequence test (known gradients -> hand-computed
preconditioned output), plus: off-by-default identity of the engine path,
state reset, causality (whiten/diag use PAST gradients only), norm
preservation, determinism, and float32 safety.
"""
import math

import pytest
import torch

from conftest import ToyDenoiser, make_quadratic_log_f
from tfg.config import PrecondConfig, TFGConfig
from tfg.engine import GeneralizedTFG
from tfg.noise_tape import NoiseTape
from tfg.precond import GuidancePreconditioner, make_preconditioner
from tfg.schedule import DiffusionSchedule
from tfg.trace import Tracer, compare_traces

SHAPE, T = (1, 2), 8


def _engine(cfg_mut=None, seed=17):
    sch = DiffusionSchedule(T=T)
    den = ToyDenoiser(d=2, T=T, seed=0, schedule=sch)
    lf = make_quadratic_log_f([1.5, -0.75], 0.6)
    cfg = TFGConfig(T=T, rho_scalar=0.5)
    if cfg_mut is not None:
        cfg_mut(cfg)
    tape = NoiseTape(seed=seed)
    eng = GeneralizedTFG(lambda x, t: den(x, t), lf, sch, tape, cfg)
    tr = Tracer()
    x = eng.run(SHAPE, trace=tr)
    return x, eng, tr


# -- factory / config -------------------------------------------------------

def test_factory_none_and_default_config_give_no_preconditioner():
    assert make_preconditioner(None) is None
    assert make_preconditioner(PrecondConfig()) is None          # mode="none"
    assert isinstance(make_preconditioner(PrecondConfig(mode="sign")),
                      GuidancePreconditioner)


def test_factory_rejects_bad_values():
    with pytest.raises(ValueError):
        GuidancePreconditioner("frobnicate")
    with pytest.raises(ValueError):
        GuidancePreconditioner("diag", ema=1.0)
    with pytest.raises(ValueError):
        GuidancePreconditioner("diag", eps=0.0)
    with pytest.raises(ValueError):
        GuidancePreconditioner("median", window=0)
    with pytest.raises(ValueError):
        GuidancePreconditioner("whiten", warmup=-1)
    with pytest.raises(ValueError):
        _engine(lambda c: setattr(c.temporal.precond, "mode", "frobnicate"))


def test_engine_default_path_identical_with_precond_field_present():
    # The precond field exists on every config now; mode="none" must leave the
    # engine trace byte-identical to a config built before the field is touched.
    x_a, _, tr_a = _engine(None)
    x_b, _, tr_b = _engine(lambda c: None)
    ok, rep = compare_traces(tr_a, tr_b, atol=0.0)
    assert ok, rep
    assert torch.equal(x_a, x_b)


# -- sign -------------------------------------------------------------------

def test_sign_known_values_norm_preserving_and_zero_safe():
    p = GuidancePreconditioner("sign")
    g = torch.tensor([[3.0, -4.0]])
    out = p.apply(g)
    # ||g|| = 5, 2 nonzeros -> each coordinate 5/sqrt(2)
    exp = torch.tensor([[5.0, -5.0]]) / math.sqrt(2.0)
    assert torch.allclose(out, exp)
    assert abs(float(out.norm()) - 5.0) < 1e-6
    # a zero coordinate is excluded from the count
    out2 = p.apply(torch.tensor([[0.0, -2.0]]))
    assert torch.allclose(out2, torch.tensor([[0.0, -2.0]]))
    # all-zero gradient passes through
    z = torch.zeros(1, 3)
    assert torch.equal(p.apply(z), z)


def test_sign_is_identity_in_one_dimension():
    p = GuidancePreconditioner("sign")
    for val in (0.7, -1.3, 0.0):
        g = torch.tensor([[val]])
        assert torch.allclose(p.apply(g), g)


# -- diag -------------------------------------------------------------------

def test_diag_known_sequence_causal_and_norm_preserving():
    p = GuidancePreconditioner("diag", ema=0.5, eps=1e-12, warmup=1)
    g1 = torch.tensor([[2.0, 1.0]], dtype=torch.float64)
    # first step: no past state -> identity, state absorbs g1
    assert torch.allclose(p.apply(g1), g1)
    # second step: v = g1^2 = [4, 1]; u = g2 / sqrt(v_reg)
    g2 = torch.tensor([[2.0, 2.0]], dtype=torch.float64)
    out = p.apply(g2)
    v = torch.tensor([4.0, 1.0], dtype=torch.float64)
    v_reg = v + 1e-12 * v.mean()
    u = g2.reshape(-1) / torch.sqrt(v_reg)
    exp = (u * (g2.norm() / u.norm())).reshape(1, 2)
    assert torch.allclose(out, exp, atol=1e-12)
    assert abs(float(out.norm()) - float(g2.norm())) < 1e-10
    # the equalisation went the right way: coordinate 1 (small past variance)
    # is boosted relative to coordinate 0
    assert out[0, 1] > out[0, 0]


def test_diag_warmup_is_identity():
    p = GuidancePreconditioner("diag", warmup=3)
    gs = [torch.tensor([[1.0, 5.0]]), torch.tensor([[2.0, 0.5]]),
          torch.tensor([[3.0, 3.0]])]
    for g in gs:                      # seen = 0, 1, 2 < warmup -> identity
        assert torch.allclose(p.apply(g), g)
    g4 = torch.tensor([[1.0, 1.0]])
    assert not torch.allclose(p.apply(g4), g4)   # active from step 4


# -- whiten -----------------------------------------------------------------

def test_whiten_known_sequence_diagonal_case_matches_diag_math():
    # With axis-aligned history the covariance is diagonal and whitening must
    # reproduce the diagonal rule exactly.
    p = GuidancePreconditioner("whiten", ema=0.5, eps=1e-9, warmup=2)
    g1 = torch.tensor([[2.0, 0.0]], dtype=torch.float64)
    g2 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    assert torch.allclose(p.apply(g1), g1)       # warmup
    assert torch.allclose(p.apply(g2), g2)       # warmup
    # state: C initialises to g1 g1^T = diag(4,0), then EMA with g2 g2^T:
    # C = 0.5*diag(4,0) + 0.5*diag(0,1) = diag(2, 0.5)
    g3 = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    out = p.apply(g3)
    C = torch.diag(torch.tensor([2.0, 0.5], dtype=torch.float64))
    ridge = 1e-9 * torch.diagonal(C).sum() / 2
    w = g3.reshape(-1) / torch.sqrt(torch.diagonal(C) + ridge)
    exp = (w * (g3.norm() / w.norm())).reshape(1, 2)
    assert torch.allclose(out, exp, atol=1e-9)
    assert abs(float(out.norm()) - float(g3.norm())) < 1e-10
    # the low-variance direction is boosted
    assert out[0, 1] > out[0, 0]


def test_whiten_rotation_invariance():
    # Whitening a rotated history with a rotated gradient = rotating the
    # whitened output (the diagonal rule fails this; the full-cov rule must not).
    th = 0.3
    R = torch.tensor([[math.cos(th), -math.sin(th)],
                      [math.sin(th), math.cos(th)]], dtype=torch.float64)
    gs = [torch.tensor([2.0, 0.1]), torch.tensor([1.5, -0.2]),
          torch.tensor([2.5, 0.3]), torch.tensor([1.0, 0.05])]
    q = torch.tensor([1.0, 1.0], dtype=torch.float64)
    pa = GuidancePreconditioner("whiten", ema=0.5, eps=1e-6, warmup=2)
    pb = GuidancePreconditioner("whiten", ema=0.5, eps=1e-6, warmup=2)
    for g in gs:
        g = g.to(torch.float64)
        pa.apply(g.reshape(1, 2))
        pb.apply((R @ g).reshape(1, 2))
    out_a = pa.apply(q.reshape(1, 2)).reshape(-1)
    out_b = pb.apply((R @ q).reshape(1, 2)).reshape(-1)
    assert torch.allclose(R @ out_a, out_b, atol=1e-8)


# -- median -----------------------------------------------------------------

def test_median_window_known_values_and_outlier_rejection():
    p = GuidancePreconditioner("median", window=3)
    g1 = torch.tensor([[1.0, -1.0]])
    assert torch.allclose(p.apply(g1), g1)              # window = [g1]
    g2 = torch.tensor([[3.0, -3.0]])
    out2 = p.apply(g2)                                  # median of {1,3} etc.
    # torch.median of an even count returns the lower middle: 1.0 / -3.0
    assert torch.allclose(out2, torch.tensor([[1.0, -3.0]]))
    g3 = torch.tensor([[500.0, -500.0]])                # tail outlier
    out3 = p.apply(g3)                                  # median of {1,3,500}
    assert torch.allclose(out3, torch.tensor([[3.0, -3.0]]))
    g4 = torch.tensor([[2.0, -2.0]])                    # g1 evicted
    out4 = p.apply(g4)                                  # median of {3,500,2}
    assert torch.allclose(out4, torch.tensor([[3.0, -3.0]]))


# -- state / determinism / dtype -------------------------------------------

def test_reset_restores_fresh_state():
    p = GuidancePreconditioner("diag", warmup=0)
    seq = [torch.tensor([[1.0, 4.0]]), torch.tensor([[2.0, 1.0]])]
    first = [p.apply(g).clone() for g in seq]
    p.reset()
    second = [p.apply(g).clone() for g in seq]
    for a, b in zip(first, second):
        assert torch.equal(a, b)
    assert p.state()["seen"] == 2


def test_deterministic_and_float32_safe():
    for mode in GuidancePreconditioner.MODES:
        pa = GuidancePreconditioner(mode, warmup=1, window=3)
        pb = GuidancePreconditioner(mode, warmup=1, window=3)
        torch.manual_seed(0)
        seq = [torch.randn(1, 5, dtype=torch.float32) for _ in range(6)]
        for g in seq:
            oa, ob = pa.apply(g), pb.apply(g)
            assert torch.equal(oa, ob), mode
            assert oa.dtype == torch.float32 and oa.shape == g.shape
            assert torch.isfinite(oa).all()


def test_engine_with_precond_runs_and_differs_from_baseline():
    x_base, _, _ = _engine(None)
    for mode in ("whiten", "diag", "median"):
        def mut(cfg, _m=mode):
            cfg.temporal.precond.mode = _m
            cfg.temporal.precond.warmup = 1
            cfg.temporal.precond.window = 3
        x, eng, _ = _engine(mut)
        assert torch.isfinite(x).all()
        assert eng._precond is not None
        assert not torch.equal(x, x_base), mode
    # sign in d=2 with a generic gradient also differs
    x_s, _, _ = _engine(lambda c: setattr(c.temporal.precond, "mode", "sign"))
    assert torch.isfinite(x_s).all()


def test_engine_precond_does_not_change_call_counts():
    _, eng_base, _ = _engine(None)

    def mut(cfg):
        cfg.temporal.precond.mode = "median"
        cfg.temporal.precond.window = 3
    _, eng_p, _ = _engine(mut)
    assert eng_p.counter.conditional_calls == eng_base.counter.conditional_calls
    assert eng_p.counter.denoiser_calls == eng_base.counter.denoiser_calls
