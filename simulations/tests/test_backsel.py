"""[Agent B] Importance-selected backpropagation (tfg/backsel.py): bit-identical
subset regeneration, k=n exactness (all rules, 1e-12), statistical
unbiasedness of the uniform and importance rules, mass conservation of the
k-center rule, off-by-default identity, engine/runner integration and the
two cost currencies.

Run: cd simulations && python -m pytest tests/test_backsel.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

SIM = Path(__file__).resolve().parents[1]
for p in (SIM / "experiments", SIM.parents[0] / "experiments" / "model-optimization" / "estimator"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tfg.backsel import (output_gradients, select_importance, select_kcenter,   # noqa: E402
                         select_uniform, wrap_log_f)
from tfg.config import BackselConfig, TFGConfig                                # noqa: E402
from tfg.distributional import DistributionalLoss                              # noqa: E402
from tfg.noise_tape import NoiseTape                                           # noqa: E402

CKPT = SIM / "artifacts" / "checkpoints"
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="no checkpoints")


# ---------------------------------------------------------------------------
# helpers: a tape-keyed, nonlinear, differentiable fake sampler + fixed loss
# ---------------------------------------------------------------------------

class ToySampler:
    """``y_i = tanh(A_i x + 3 z_i) + B_i x`` with ``z_i`` keyed on the eta key
    (like ``CMSampler``): per-sample Jacobians differ, subsets regenerate
    exactly, heavy-ish spread of the outputs."""
    cache_on = False

    def __init__(self, tape, d_x=2, d_y=2, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.A = torch.randn(d_y, d_x, generator=g, dtype=torch.float64)
        self.B = 0.3 * torch.randn(d_y, d_x, generator=g, dtype=torch.float64)
        self.tape, self.d_y = tape, d_y
        self.cm_samples = 0

    def __call__(self, x, keys):
        Z = torch.stack([self.tape.randn(k, (self.d_y,), dtype=torch.float64) for k in keys])
        self.cm_samples += len(keys)
        lin = x.reshape(1, -1) @ self.A.T
        return torch.tanh(lin + 3.0 * Z) + x.reshape(1, -1) @ self.B.T


def make_setup(seed=0, m=60, backend="reference"):
    tape = NoiseTape(seed=seed)
    g = torch.Generator().manual_seed(seed + 100)
    S = torch.randn(m, 2, generator=g, dtype=torch.float64) * 1.3 + 0.4
    loss = DistributionalLoss(S, bandwidth="fixed", bandwidth_value=1.7, backend=backend)
    return tape, ToySampler(tape), loss


def keys_for(t, n, j=0):
    return [("eta", t, j, i) for i in range(n)]


def full_grad(sampler, loss, x, keys):
    xl = x.detach().clone().requires_grad_(True)
    v = -loss(sampler(xl, keys))
    (g,) = torch.autograd.grad(v, xl)
    return v.detach(), g


# ---------------------------------------------------------------------------
# bit-identical subset regeneration
# ---------------------------------------------------------------------------

def test_toy_subset_regeneration_bit_identical():
    tape, sampler, _ = make_setup()
    x = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    keys = keys_for(7, 8)
    with torch.no_grad():
        Y = sampler(x, keys)
    for idx in ([1, 5], [0], [2, 3, 7], list(range(8))):
        assert torch.equal(sampler(x, [keys[i] for i in idx]), Y[idx])


@needs_ckpt
def test_cmsampler_subset_regeneration_bit_identical(f32):
    """The real conditional sampler: a SUBSET of the per-sample eta keys
    replays bit-identical NOISE (the tape), and the regenerated rows equal
    the no-grad full-batch rows to float32 round-off (measured max rel
    1.1e-6 at n=32; 1e-14 in float64).  They are NOT bit-identical in
    general: CPU BLAS blocking depends on the batch dimension, so a batched
    row's rounding differs between a batch of n and a batch of k -- a
    property of batched inference, not of the keying.  The surrogate is
    linear in the regenerated rows with the output gradient fixed, so this
    round-off enters the estimator only through the evaluation point of J_i."""
    from _models import PAPER_TS
    from engine_runner import build_models
    from tfg.distributional import CMSampler
    _, S_G, bw, mc, mu = build_models("2D")
    tape = NoiseTape(seed=3, dtype=torch.float32)
    sampler = CMSampler(mc, PAPER_TS, tape, source="tape", dtype=torch.float32)
    x = torch.tensor([[0.7]], dtype=torch.float32, requires_grad=True)
    for n in (8, 32):
        keys = keys_for(50, n)
        Z_full = sampler._noise(keys, n, mc.nfeatures)
        with torch.no_grad():
            Y = sampler(x, keys)
        assert not Y.requires_grad
        for idx in ([0], [3, 5], [n - 1, 1, 4, 6], sorted(range(0, n, 3))):
            sub = [keys[i] for i in idx]
            assert torch.equal(sampler._noise(sub, len(sub), mc.nfeatures), Z_full[:, idx])
            y_sub = sampler(x, sub)
            assert y_sub.requires_grad
            assert torch.allclose(y_sub.detach(), Y[idx], rtol=1e-5, atol=1e-5), (n, idx)
        # the full key set regenerates bit-identically (same batch dimension)
        assert torch.equal(sampler(x, keys).detach(), Y)


# ---------------------------------------------------------------------------
# k = n: exact full gradient, all rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["reference", "fast"])
@pytest.mark.parametrize("rule", ["uniform", "importance", "kcenter"])
def test_k_equals_n_reproduces_full_gradient(rule, backend):
    tape, sampler, loss = make_setup(backend=backend)
    x = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    n = 8
    keys = keys_for(9, n)
    v_ref, g_ref = full_grad(sampler, loss, x, keys)
    for k in (n, n + 3):
        lf = wrap_log_f(sampler, loss, tape, BackselConfig(enabled=True, rule=rule, k=k))
        xl = x.clone().requires_grad_(True)
        v = lf(xl, n_t=n, eta_keys=keys)
        (g,) = torch.autograd.grad(v, xl)
        assert abs(float(v.detach()) - float(v_ref)) < 1e-12
        assert float((g - g_ref).abs().max()) < 1e-12, rule
        assert lf.stats["diff_samples"] % n == 0


def test_value_is_full_batch_and_grad_is_subset_only():
    tape, sampler, loss = make_setup()
    x = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    n, k = 8, 3
    keys = keys_for(9, n)
    v_ref, _ = full_grad(sampler, loss, x, keys)
    lf = wrap_log_f(sampler, loss, tape, BackselConfig(enabled=True, rule="uniform", k=k))
    sampler.cm_samples = 0
    xl = x.clone().requires_grad_(True)
    v = lf(xl, n_t=n, eta_keys=keys)
    torch.autograd.grad(v, xl)
    assert abs(float(v.detach()) - float(v_ref)) < 1e-12 # value: full batch
    assert sampler.cm_samples == n + k                     # forwards: n + k
    assert lf.stats["diff_samples"] == k                   # graphs: k
    assert lf.stats["forward_samples"] == n + k


# ---------------------------------------------------------------------------
# unbiasedness (expectation over selection draws == full gradient)
# ---------------------------------------------------------------------------

def _selection_mean(rule, n=8, k=3, R=3000, floor=0.25):
    tape, sampler, loss = make_setup()
    x = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    keys = keys_for(11, n)
    _, g_ref = full_grad(sampler, loss, x, keys)
    with torch.no_grad():
        Y = sampler(x, keys)
    g_out, _ = output_gradients(loss, Y)
    # per-sample h_i = J_i^T g_i (n exact terms) -> every estimator is a
    # weighted sum of these, so we can draw selections cheaply
    H = []
    for i in range(n):
        xl = x.clone().requires_grad_(True)
        yi = sampler(xl, [keys[i]])
        (h,) = torch.autograd.grad((yi * g_out[i]).sum(), xl)
        H.append(h.reshape(-1))
    H = torch.stack(H)                                   # (n, d_x)
    assert torch.allclose(-H.sum(0), g_ref.reshape(-1), atol=1e-10)
    ests = []
    for r in range(R):
        key = ("backsel", r, 0)
        if rule == "uniform":
            idx, g_eff = select_uniform(g_out, k, tape, key)
        else:
            idx, g_eff = select_importance(g_out, k, tape, key, floor=floor)
        # weight of row i = ||g_eff_i|| / ||g_i|| (g_eff is w_i * g_i)
        w = torch.tensor([float(g_eff[a].norm() / g_out[i].norm()) for a, i in enumerate(idx)],
                         dtype=torch.float64)
        ests.append((w.unsqueeze(1) * H[idx]).sum(0))
        assert len(idx) <= k
    E = torch.stack(ests)
    return E, H.sum(0)


@pytest.mark.parametrize("rule", ["uniform", "importance"])
def test_selection_is_unbiased(rule):
    E, G = _selection_mean(rule)
    mean = E.mean(0)
    se = E.std(0) / E.shape[0] ** 0.5
    z = ((mean - G).abs() / se).max()
    assert float(z) < 4.0, (rule, mean, G, se)
    assert float((mean - G).norm() / G.norm()) < 0.05


def test_importance_beats_uniform_on_heavy_tail_and_weights_are_ip():
    tape, sampler, loss = make_setup()
    n, k = 8, 3
    # a synthetic heavy-tailed output gradient: one row carries 90% of the norm
    g = torch.full((n, 2), 0.05, dtype=torch.float64)
    g[2] = torch.tensor([4.0, -3.0])
    norms = g.norm(dim=-1)
    p = 0.75 * norms / norms.sum() + 0.25 / n
    for r in range(50):
        idx, g_eff = select_importance(g, k, tape, ("t", r, 0))
        assert 1 <= len(idx) <= k
        w = g_eff.norm(dim=-1) / norms[idx]
        counts = torch.round(w * k * p[idx])
        assert torch.allclose(w, counts / (k * p[idx]))       # w_i = c_i / (k p_i)
        assert int(counts.sum()) == k
    # variance of the (scalar) estimator sum_i w_i ||g_i|| over 2000 draws
    tot = float(norms.sum())
    est_u, est_i = [], []
    for r in range(2000):
        iu, gu = select_uniform(g, k, tape, ("u", r, 0))
        ii, gi = select_importance(g, k, tape, ("i", r, 0))
        est_u.append(float(gu.norm(dim=-1).sum()))
        est_i.append(float(gi.norm(dim=-1).sum()))
    eu, ei = torch.tensor(est_u), torch.tensor(est_i)
    assert abs(float(eu.mean()) - tot) < 0.05 * tot and abs(float(ei.mean()) - tot) < 0.05 * tot
    assert float(ei.var()) < 0.5 * float(eu.var())


def test_kcenter_conserves_gradient_mass_and_is_deterministic():
    tape, sampler, loss = make_setup()
    x = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    n = 8
    with torch.no_grad():
        Y = sampler(x, keys_for(5, n))
    g = torch.randn(n, 2, dtype=torch.float64)
    for k in (1, 2, 4, 8):
        idx, g_eff = select_kcenter(Y, g, k, tape, ("kc", 5, 0))
        assert len(idx) == min(k, n) and len(set(idx)) == len(idx)
        assert torch.allclose(g_eff.sum(0), g.sum(0), atol=1e-12)
        idx2, g_eff2 = select_kcenter(Y, g, k, tape, ("kc", 5, 0))
        assert idx == idx2 and torch.equal(g_eff, g_eff2)


# ---------------------------------------------------------------------------
# off by default / config / engine integration
# ---------------------------------------------------------------------------

def test_disabled_wrapper_is_plain_path_and_config_validates():
    tape, sampler, loss = make_setup()
    x = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    keys = keys_for(4, 6)
    v_ref, g_ref = full_grad(sampler, loss, x, keys)
    lf = wrap_log_f(sampler, loss, tape, BackselConfig(enabled=False, rule="kcenter", k=2))
    xl = x.clone().requires_grad_(True)
    v = lf(xl, n_t=6, eta_keys=keys)
    (g,) = torch.autograd.grad(v, xl)
    assert float(v) == float(v_ref) and torch.equal(g, g_ref)
    cfg = TFGConfig()
    assert cfg.backsel.enabled is False and cfg.all_extensions_disabled()
    cfg.validate()
    with pytest.raises(ValueError):
        BackselConfig(rule="random").validate()
    with pytest.raises(ValueError):
        BackselConfig(k=0).validate()
    with pytest.raises(ValueError):
        BackselConfig(floor=0.0).validate()
    with pytest.raises(ValueError):
        wrap_log_f(sampler, loss, tape, BackselConfig(enabled=True))(x, n_t=6, eta_keys=None)


@pytest.fixture
def f32():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(prev)


@needs_ckpt
def test_engine_backsel_off_is_identical_and_on_counts_currencies(f32):
    from engine_runner import build_models, run_engine
    _, S_G, bw, mc, mu = build_models("2D")
    xa, ia = run_engine(mc, mu, S_G, bw, 8, "no_lgd", "none", 0)

    def off(cfg):
        cfg.backsel.rule, cfg.backsel.k = "importance", 2      # enabled stays False
    xb, ib = run_engine(mc, mu, S_G, bw, 8, "no_lgd", "none", 0, cfg_mutator=off)
    assert torch.equal(xa, xb) and ia["cm_samples"] == ib["cm_samples"]
    assert ia["diff_samples"] == ia["cm_samples"]
    T = ia["steps"]
    for cand, k in (("backsel_uni_k2", 2), ("backsel_is_k4_trust", 4), ("backsel_clust_k2", 2)):
        x, i = run_engine(mc, mu, S_G, bw, 8, "no_lgd", "none", 0, candidate=cand)
        assert torch.isfinite(x).all()
        if cand.startswith("backsel_is"):                # iid draws de-duplicated: <= k
            assert 8 * T < i["cm_samples"] <= (8 + k) * T and 0 < i["diff_samples"] <= k * T, cand
            assert i["cm_samples"] - 8 * T == i["diff_samples"]
        else:
            assert i["cm_samples"] == (8 + k) * T, cand  # forwards: n + k
            assert i["diff_samples"] == k * T, cand      # graphs: k
    # k = n through the engine: identical trajectory to the baseline
    x8, i8 = run_engine(mc, mu, S_G, bw, 8, "no_lgd", "none", 0, candidate="backsel_uni_k8")
    assert torch.allclose(x8, xa, atol=1e-5) and i8["diff_samples"] == 8 * T


@needs_ckpt
def test_engine_backsel_composes_with_cohort_replay(f32):
    from engine_runner import build_models, run_engine
    _, S_G, bw, mc, mu = build_models("2D")
    x, i = run_engine(mc, mu, S_G, bw, 8, "no_lgd", "none", 1,
                      candidate="backsel_is_k2_cohort16_trust")
    T = i["steps"]
    assert torch.isfinite(x).all()
    assert 8 * T < i["cm_samples"] <= (8 + 2) * T and 0 < i["diff_samples"] <= 2 * T
    assert i["cm_samples"] - 8 * T == i["diff_samples"]   # fresh forwards = n + diff


# ---------------------------------------------------------------------------
# [B-R7] stratified selection and soft assignment
# ---------------------------------------------------------------------------

def test_soft_limits_and_mass_conservation():
    from tfg.backsel import select_kcenter, select_stratified, soft_aggregate
    tape, sampler, loss = make_setup()
    x = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    n, k = 16, 4
    with torch.no_grad():
        Y = sampler(x, keys_for(5, n))
    g = torch.randn(n, 2, dtype=torch.float64)
    idx, hard = select_kcenter(Y, g, k, tape, ("kc", 5, 0))
    # tau -> 0 with the centers as S recovers hard kcenter aggregation
    assert torch.allclose(soft_aggregate(Y, g, idx, 1e-9), hard, atol=1e-10)
    # tau -> inf: every non-selected g split equally over S
    mask = torch.ones(n, dtype=torch.bool); mask[idx] = False
    expect = g[idx] + g[mask].sum(0) / k
    assert torch.allclose(soft_aggregate(Y, g, idx, 1e9), expect, atol=1e-10)
    # mass conservation at every tau; stratified reps are distinct members of distinct clusters
    for tau in (1e-3, 0.5, 2.0, 50.0):
        assert torch.allclose(soft_aggregate(Y, g, idx, tau).sum(0), g.sum(0), atol=1e-10)
    sidx, sg = select_stratified(Y, g, k, tape, ("st", 5, 0))
    assert len(set(sidx)) == k and torch.allclose(sg.sum(0), g.sum(0), atol=1e-10)


@pytest.mark.parametrize("rule,weighting", [("stratified", "hard"), ("uniform", "soft"),
                                            ("stratified", "soft"), ("kcenter", "soft")])
def test_soft_and_stratified_k_equals_n_exact(rule, weighting):
    tape, sampler, loss = make_setup(backend="fast")
    x = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    n = 8
    keys = keys_for(9, n)
    v_ref, g_ref = full_grad(sampler, loss, x, keys)
    lf = wrap_log_f(sampler, loss, tape, BackselConfig(enabled=True, rule=rule, weighting=weighting, k=n))
    xl = x.clone().requires_grad_(True)
    v = lf(xl, n_t=n, eta_keys=keys)
    (g,) = torch.autograd.grad(v, xl)
    assert abs(float(v.detach()) - float(v_ref)) < 1e-12
    assert float((g - g_ref).abs().max()) < 1e-12
    # k < n runs, uses tau = tau_mult x bandwidth, and conserves mass in the surrogate
    lf2 = wrap_log_f(sampler, loss, tape, BackselConfig(enabled=True, rule=rule, weighting=weighting, k=3, tau_mult=0.25))
    xl = x.clone().requires_grad_(True)
    v2 = lf2(xl, n_t=n, eta_keys=keys)
    torch.autograd.grad(v2, xl)
    assert lf2.stats["diff_samples"] == 3
    with pytest.raises(ValueError):
        BackselConfig(weighting="fuzzy").validate()


# ---------------------------------------------------------------------------
# [Agent S] shared primitives: balanced stratified rule, soft tau modes, trust
# ---------------------------------------------------------------------------

def test_stratified_balanced_unbiased_and_bounded():
    from tfg.backsel import select_stratified_balanced
    from tfg.noise_tape import NoiseTape
    torch.manual_seed(0)
    n, k = 32, 8
    Y = torch.randn(n, 3, dtype=torch.float64); Y[:20] *= 0.01     # one tight blob + spread
    g = torch.randn(n, 3, dtype=torch.float64)
    acc = torch.zeros(n, 3, dtype=torch.float64)
    R = 3000
    for r in range(R):
        idx, ge, sizes = select_stratified_balanced(Y, g, k, NoiseTape(r), ("b", 1, 0))
        assert len(idx) == k and max(sizes) <= 4 and sum(sizes) == n
        acc[idx] += ge
    # E[sum_i G_i e_i] = sum_i g_i e_i  <=> per-row expectation of the weighted g equals g
    assert float((acc / R - g).abs().max()) < 0.15
    idx, ge, sizes = select_stratified_balanced(Y, g, n, NoiseTape(0), ("b", 1, 0))
    assert list(idx) == list(range(n)) and torch.equal(ge, g)


def test_soft_tau_modes_and_config():
    from tfg.backsel import soft_tau, soft_aggregate
    from tfg.config import BackselConfig
    torch.manual_seed(1)
    centers = torch.randn(4, 6, dtype=torch.float64) * 10
    Y = centers.repeat_interleave(8, dim=0) + 0.1 * torch.randn(32, 6, dtype=torch.float64)
    idx = [0, 8, 16, 24, 3, 11]
    assert soft_tau(Y, idx, "local") < soft_tau(Y, idx, "bandwidth")
    assert soft_tau(Y, idx, "bandwidth", bandwidth=2.5, scale=2.0) == 5.0
    g = torch.randn(32, 6, dtype=torch.float64)
    ge, mass = soft_aggregate(Y, g, idx, soft_tau(Y, idx, "local"), return_mass=True)
    assert abs(float(mass.sum()) - 32) < 1e-9 and torch.allclose(ge.sum(0), g.sum(0), atol=1e-10)
    BackselConfig(rule="stratified_balanced", weighting="soft", tau_mode="local").validate()
    with pytest.raises(ValueError):
        BackselConfig(tau_mode="global").validate()


def test_trust_clip_step_and_engine_mode():
    from tfg.trust import clip_step, noise_cap
    from tfg.config import TFGConfig, TemporalConfig
    d = torch.ones(10, dtype=torch.float64)
    assert abs(float(clip_step(d, 1.0).norm()) - 1.0) < 1e-12
    assert torch.equal(clip_step(d, 100.0), d)
    assert noise_cap(1.0, 0.5, numel=16) == 2.0 and noise_cap(1.0, 0.0, numel=1, min_noise=0.03) == 0.03
    TFGConfig(temporal=TemporalConfig(step_clip="noise_prev_rms")).validate()
    with pytest.raises(ValueError):
        TFGConfig(temporal=TemporalConfig(step_clip="noise_rms")).validate()


def test_engine_run_accepts_x_init():
    from tfg.config import TFGConfig
    from tfg.engine import GeneralizedTFG
    from tfg.noise_tape import NoiseTape
    from tfg.schedule import DiffusionSchedule
    sch = DiffusionSchedule(T=5)
    eng = GeneralizedTFG(lambda x, t: torch.zeros_like(x), lambda x: (x ** 2).sum(), sch,
                         NoiseTape(0), TFGConfig(T=5))
    x0 = torch.full((1, 2), 3.0, dtype=torch.float64)
    out = eng.run((1, 2), x_init=x0)
    # eps=0 and rho=0: pure DDIM with zero noise prediction rescales x_T by sqrt(ab_0/ab_T)
    assert torch.allclose(out, x0 / sch.sqrt_ab(5), atol=1e-8)
