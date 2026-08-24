"""[A4] Tiny deterministic tests for the campaign's opt-in engine mechanisms.

Each candidate of experiments/model-optimization/hypotheses/agent4.yaml has at
least one test here:
  1 normalisation-only / clipping / unit-norm        (temporal.grad_norm, beta1=0)
  2 adaptive n_t (agreement / improvement, budget)    (n_schedule.type="adaptive")
  3 adaptive recurrence v1 (early stopping)           (adaptive_recurrence)
  4 variance reduction: frozen keys / antithetic      (eta_keying, CMSampler)
  5 small-n bandwidth pathologies + robust policies   (DistributionalLoss)
  6 stale-gradient reuse (approximate)                (temporal_cache "stale")
plus: the default path is untouched (all_extensions_disabled, trace-identical).
"""
import math

import pytest
import torch

from conftest import ToyDenoiser, make_quadratic_log_f
from tfg.adaptive import adaptive_n, gradient_agreement, recurrence_should_stop
from tfg.config import TFGConfig
from tfg.distributional import DistributionalLoss
from tfg.engine import GeneralizedTFG
from tfg.n_schedule import conditional_seed_keys, n_at
from tfg.noise_tape import NoiseTape
from tfg.reference import run_reference_tfg
from tfg.schedule import DiffusionSchedule, constant_vector
from tfg.trace import Tracer, compare_traces, format_report

SHAPE, T = (1, 2), 8


def _setup(seed=17):
    sch = DiffusionSchedule(T=T)
    den = ToyDenoiser(d=2, T=T, seed=0, schedule=sch)
    return sch, (lambda x, t: den(x, t)), seed


class NoisyLogF:
    """Distributional-style predictor: -||mean of n noisy draws around x - c||^2,
    with the draws keyed by eta keys (order independent)."""

    def __init__(self, tape, center=(1.5, -0.75), sigma=1.0):
        self.tape, self.c, self.sigma = tape, torch.tensor(center, dtype=torch.float64), sigma
        self.calls, self.samples = 0, 0
        self.seen_keys = []

    def __call__(self, x, n_t=None, eta_keys=None):
        self.calls += 1
        self.samples += len(eta_keys)
        self.seen_keys.append(tuple(eta_keys))
        Z = torch.stack([self.tape.randn(k, x.shape, dtype=x.dtype) for k in eta_keys])
        y = (x + self.sigma * Z).mean(0)
        return -((y - self.c) ** 2).sum()


def _engine(cfg_mut, seed=17, log_f=None, rho=0.5):
    sch, eps_theta, seed = _setup(seed)
    tape = NoiseTape(seed=seed)
    lf = log_f(tape) if callable(log_f) else make_quadratic_log_f([1.5, -0.75], 0.6)
    cfg = TFGConfig(T=T, rho_scalar=rho)
    cfg_mut(cfg)
    eng = GeneralizedTFG(eps_theta, lf, sch, tape, cfg)
    tr = Tracer()
    x = eng.run(SHAPE, trace=tr)
    return x, eng, tr, lf


# -- 0. default path untouched ----------------------------------------------

def test_defaults_are_still_the_reference():
    sch, eps_theta, seed = _setup()
    lf = make_quadratic_log_f([1.5, -0.75], 0.6)
    tr_ref = Tracer()
    run_reference_tfg(eps_theta, lf, sch, NoiseTape(seed=seed), SHAPE, N_recur=2,
                      gamma_bar=0.1, n_mc=2, rho=constant_vector(0.5, T), trace=tr_ref)
    cfg = TFGConfig(T=T, N_recur=2, gamma_bar=0.1, n_mc=2, rho_scalar=0.5)
    assert cfg.all_extensions_disabled()
    tr = Tracer()
    GeneralizedTFG(eps_theta, lf, sch, NoiseTape(seed=seed), cfg).run(SHAPE, trace=tr)
    ok, rep = compare_traces(tr_ref, tr, atol=0.0)
    assert ok, format_report(rep)


def test_new_switches_flip_all_extensions_disabled():
    for mut in (lambda c: setattr(c, "init", "zeros"),
                lambda c: setattr(c, "guidance_scaling", "raw"),
                lambda c: setattr(c, "smoothing", "lgd_beta"),
                lambda c: setattr(c.temporal, "grad_norm", "unit")):
        c = TFGConfig(T=T)
        mut(c)
        assert not c.validate().all_extensions_disabled()


def test_gated_surfaces_still_refused_but_v1_accepted():
    c = TFGConfig(T=T)
    c.adaptive_recurrence.enabled = True
    with pytest.raises(NotImplementedError):
        c.validate()
    c.adaptive_recurrence.implementation = "v1"
    c.adaptive_recurrence.max_recurrences = 3
    c.validate()
    c.N_recur = 2
    with pytest.raises(ValueError):
        c.validate()
    c = TFGConfig(T=T)
    c.temporal_cache.enabled = True
    with pytest.raises(NotImplementedError):
        c.validate()
    c.temporal_cache.implementation = "stale"
    c.temporal_cache.refresh_every = 2
    c.validate()


# -- legacy switches --------------------------------------------------------

def test_init_zeros_and_raw_scaling():
    x_def, eng, tr, _ = _engine(lambda c: None)
    x_z, _, tr_z, _ = _engine(lambda c: setattr(c, "init", "zeros"))
    assert torch.equal(tr_z.records[("x_T", T, None, None)], torch.zeros(SHAPE, dtype=torch.float64))
    assert not torch.equal(x_def, x_z)
    # raw scaling: x_prev - x_ddim == Delta_t exactly (no /sqrt(alpha_t))
    _, _, tr_r, _ = _engine(lambda c: setattr(c, "guidance_scaling", "raw"))
    for t in range(T, 0, -1):
        d = tr_r.records[("x_prev", t, 1, None)] - tr_r.records[("x_ddim", t, 1, None)]
        assert torch.allclose(d, tr_r.records[("Delta_t", t, 1, None)], atol=1e-15, rtol=0)
    sch = DiffusionSchedule(T=T)
    for t in range(T, 0, -1):          # default divides by sqrt(alpha_t)
        d = tr.records[("x_prev", t, 1, None)] - tr.records[("x_ddim", t, 1, None)]
        assert torch.allclose(d, tr.records[("Delta_t", t, 1, None)] / torch.sqrt(sch.alpha(t)),
                              atol=1e-15, rtol=0)


def test_lgd_beta_smoothing_scale():
    """x + r_t*delta with r_t = beta_t/sqrt(1+beta_t^2): the perturbed input
    that reaches log_f differs from x0 by exactly r_t * delta."""
    seen = []
    sch, eps_theta, seed = _setup()

    def lf(x):
        seen.append(x.detach().clone())
        return -((x - 1.0) ** 2).sum()
    cfg = TFGConfig(T=T, rho_scalar=0.1, smoothing="lgd_beta", n_mc=2)
    tape = NoiseTape(seed=seed)
    tr = Tracer()
    GeneralizedTFG(eps_theta, lf, sch, tape, cfg).run(SHAPE, trace=tr)
    i = 0
    for t in range(T, 0, -1):
        x0 = tr.records[("x0_pred", t, 1, None)]
        beta = sch.betas[t - 1]
        r_t = beta / torch.sqrt(1 + beta ** 2)
        for j in range(2):
            delta = tape.randn(("delta", t, j), SHAPE)
            assert torch.allclose(seen[i], x0 + r_t * delta, atol=1e-15, rtol=0)
            i += 1


# -- 1. normalisation / clipping --------------------------------------------

def test_grad_norm_clip_and_unit():
    _, _, tr_c, _ = _engine(lambda c: (setattr(c.temporal, "grad_norm", "clip"),
                                       setattr(c.temporal, "grad_clip", 1e-3)))
    for t in range(T, 0, -1):
        g = tr_c.records[("Delta_t", t, 1, None)] / 0.5
        assert float(g.norm()) <= 1e-3 * (1 + 1e-9)
    _, _, tr_u, _ = _engine(lambda c: setattr(c.temporal, "grad_norm", "unit"))
    for t in range(T, 0, -1):
        g = tr_u.records[("Delta_t", t, 1, None)] / 0.5
        assert abs(float(g.norm()) - 1.0) < 1e-9
    # clip with a huge threshold is the identity
    x_def, _, _, _ = _engine(lambda c: None)
    x_big, _, _, _ = _engine(lambda c: (setattr(c.temporal, "grad_norm", "clip"),
                                        setattr(c.temporal, "grad_clip", 1e9)))
    assert torch.allclose(x_def, x_big, atol=1e-12, rtol=0)


def test_norm_only_is_adam_with_beta1_zero_and_differs_from_adam():
    x_a, _, _, _ = _engine(lambda c: setattr(c.temporal, "mode", "adam"))
    x_n, _, tr, _ = _engine(lambda c: (setattr(c.temporal, "mode", "adam"),
                                       setattr(c.temporal, "beta1", 0.0)))
    assert not torch.equal(x_a, x_n)
    # with beta1=0 the used gradient is g/(|g|+delta) per coordinate (bias-corrected
    # v); at k=1 that is sign(g) up to delta.
    g = tr.records[("grad_rho_raw", T, 1, None)]
    u = tr.records[("grad_rho_used", T, 1, None)]
    assert torch.allclose(u, g / (g.abs() + 1e-8), atol=1e-12, rtol=0)


# -- 2. adaptive n_t ----------------------------------------------------------

def test_adaptive_n_policy_rules_and_budget():
    st = {"n_prev": None, "n_start": 8, "n_min": 2, "grow": 2, "policy": "agreement"}
    assert adaptive_n(32, st) == 8
    st.update(n_prev=8, agreement=0.1, agreement_threshold=0.5)
    assert adaptive_n(32, st) == 16
    st.update(n_prev=8, agreement=0.9)
    assert adaptive_n(32, st) == 4
    st.update(n_prev=2, agreement=0.9)
    assert adaptive_n(32, st) == 2                  # n_min
    st.update(n_prev=32, agreement=0.0)
    assert adaptive_n(32, st) == 32                 # n_max
    # budget: never plan below n_min for the remaining steps, spend the rest last
    st.update(n_prev=8, agreement=0.0, budget_remaining=10, steps_left=3)
    assert adaptive_n(32, st) == 10 - 2 * 2
    st.update(budget_remaining=7, steps_left=1)
    assert adaptive_n(32, st) == 7
    # surplus that later steps cannot absorb at n_max is spread, not dumped
    st.update(n_prev=8, agreement=0.9, budget_remaining=100, steps_left=3)
    assert adaptive_n(32, st) == 100 - 2 * 32
    # improvement policy
    st2 = {"n_prev": 8, "n_min": 1, "grow": 2, "policy": "improvement",
           "improved": -0.1, "improvement_threshold": 0.0}
    assert adaptive_n(32, st2) == 16
    st2["improved"] = 0.3
    assert adaptive_n(32, st2) == 4
    # n_at hook: state-less adaptive == constant; other kinds ignore state
    sch = DiffusionSchedule(T=T)
    assert n_at(5, sch, 16, 1.0, "adaptive") == 16
    assert n_at(5, sch, 16, 1.0, "time", state={"n_prev": 1}) == n_at(5, sch, 16, 1.0, "time")


def test_gradient_agreement_cosine():
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    assert abs(gradient_agreement(a, b)) < 1e-12            # orthogonal -> 0
    assert abs(gradient_agreement(a, a) - 1.0) < 1e-12      # equal -> 1
    assert abs(gradient_agreement(a, -a) + 1.0) < 1e-12     # opposite -> -1
    assert abs(gradient_agreement(torch.tensor([1.0]), torch.tensor([0.5])) - 2 / 3) < 1e-12  # graded in 1-D
    assert 0.0 < gradient_agreement(a, 3 * a) < 1.0         # unequal norms: less than 1


def test_engine_adaptive_n_agreement_spends_exactly_the_budget():
    n, budget = 4, 4 * T

    def mut(c):
        c.n_schedule.enabled = True
        c.n_schedule.type = "adaptive"
        c.n_schedule.policy = "agreement"
        c.n_schedule.n_min, c.n_schedule.n_max, c.n_schedule.n_start = 2, 16, n
        c.n_schedule.agreement_threshold = 0.5
        c.n_schedule.budget_total = budget
    x, eng, tr, lf = _engine(mut, log_f=lambda tape: NoisyLogF(tape, sigma=2.0))
    hist = eng.counter.n_t_history
    assert sum(hist) == budget
    assert len(eng.counter.agreement_history) == T
    assert all(-1.0 <= a <= 1.0 for a in eng.counter.agreement_history)
    assert min(hist) >= 2 and len(set(hist)) > 1, hist      # it actually adapted
    # the half-batch keys are the two halves of the step's keys
    keys_T = [k for k in lf.seen_keys if k and k[0][1] == T]
    full, a, b = keys_T[0], keys_T[1], keys_T[2]
    assert a + b == full and len(a) == len(b) == n // 2
    # constant-n baseline at the same budget makes the same number of samples
    x_c, eng_c, _, lf_c = _engine(lambda c: (setattr(c.n_schedule, "enabled", True),
                                             setattr(c.n_schedule, "n_max", n)),
                                  log_f=lambda tape: NoisyLogF(tape, sigma=2.0))
    assert sum(eng_c.counter.n_t_history) == budget == lf_c.samples
    assert not torch.equal(x, x_c)


def test_engine_adaptive_n_improvement_policy_runs_and_keeps_budget():
    def mut(c):
        c.n_schedule.enabled = True
        c.n_schedule.type = "adaptive"
        c.n_schedule.policy = "improvement"
        c.n_schedule.n_min, c.n_schedule.n_max, c.n_schedule.n_start = 1, 16, 4
        c.n_schedule.budget_total = 4 * T
    x, eng, tr, lf = _engine(mut, log_f=lambda tape: NoisyLogF(tape, sigma=2.0))
    assert sum(eng.counter.n_t_history) == 4 * T == lf.samples
    assert eng.counter.agreement_history == []


# -- 3. adaptive recurrence -------------------------------------------------

def test_recurrence_stop_rules():
    prev = {"log_f": -1.0, "x_prev": torch.tensor([1.0, 0.0]), "grad": torch.tensor([1.0, 0.0])}
    cur_same = {"log_f": -1.0 + 1e-6, "x_prev": torch.tensor([1.0, 1e-6]), "grad": torch.tensor([1.0, 1e-6])}
    cur_diff = {"log_f": -2.0, "x_prev": torch.tensor([3.0, 0.0]), "grad": torch.tensor([0.0, 1.0])}
    assert not recurrence_should_stop("clean_proxy", 1e-2, None, cur_same)
    for m in ("clean_proxy", "next_state_tweedie", "grad_stability"):
        assert recurrence_should_stop(m, 1e-2, prev, cur_same)
        assert not recurrence_should_stop(m, 1e-2, prev, cur_diff)


def test_engine_adaptive_recurrence_v1_stops_early_and_equals_fixed_when_threshold_zero():
    def mut_v1(thr, R=3, metric="next_state_tweedie"):
        def m(c):
            c.adaptive_recurrence.enabled = True
            c.adaptive_recurrence.implementation = "v1"
            c.adaptive_recurrence.max_recurrences = R
            c.adaptive_recurrence.metric = metric
            c.adaptive_recurrence.threshold = thr
        return m
    # threshold 0 never stops -> identical to N_recur = R
    x_fixed, eng_f, tr_f, _ = _engine(lambda c: setattr(c, "N_recur", 3))
    x_v1, eng_v, tr_v, _ = _engine(mut_v1(0.0))
    assert torch.equal(x_fixed, x_v1)
    assert eng_v.counter.recurrence_history == [3] * T
    ok, rep = compare_traces(tr_f, tr_v, atol=0.0)
    assert ok, format_report(rep)
    # a large threshold stops after the second recurrence everywhere
    x_s, eng_s, _, _ = _engine(mut_v1(10.0))
    assert eng_s.counter.recurrence_history == [2] * T
    assert eng_s.counter.denoiser_calls == 2 * T
    # grad_stability metric also runs
    _, eng_g, _, _ = _engine(mut_v1(0.5, metric="grad_stability"))
    assert all(1 <= r <= 3 for r in eng_g.counter.recurrence_history)


# -- 4. variance reduction ----------------------------------------------------

def test_eta_keys_per_perturbation_and_frozen():
    assert conditional_seed_keys(7, 2) == [("eta", 7, 0), ("eta", 7, 1)]
    assert conditional_seed_keys(7, 2, j=1) == [("eta", 7, 1, 0), ("eta", 7, 1, 1)]
    assert conditional_seed_keys(7, 2, frozen=True) == [("eta", "frozen", 0), ("eta", "frozen", 1)]
    assert conditional_seed_keys(7, 1, j=2, frozen=True) == [("eta", "frozen", 2, 0)]

    def base(c):
        c.n_schedule.enabled = True
        c.n_schedule.n_max = 3
        c.n_mc = 2
    _, _, _, lf = _engine(base, log_f=NoisyLogF)
    assert all(len(k[0]) == 3 for k in lf.seen_keys)
    assert lf.seen_keys[0] == lf.seen_keys[1]                     # shared across j (C5)
    _, _, _, lf_p = _engine(lambda c: (base(c), setattr(c.n_schedule, "eta_per_perturbation", True)),
                            log_f=NoisyLogF)
    assert lf_p.seen_keys[0] != lf_p.seen_keys[1] and lf_p.seen_keys[0][0][2] == 0
    _, _, _, lf_f = _engine(lambda c: (base(c), setattr(c.n_schedule, "eta_keying", "frozen")),
                            log_f=NoisyLogF)
    assert len(set(lf_f.seen_keys)) == 1                          # same keys every step


# -- 5. small-n bandwidth pathologies -----------------------------------------

def _grad_norm(loss, Y):
    Y = Y.clone().requires_grad_(True)
    g, = torch.autograd.grad(loss(Y), Y)
    return float(g.norm())


def test_pooled_bandwidth_collapses_for_tiny_batches_and_fixed_policies_do_not():
    """Pathology: the repository's adaptive bandwidth is the mean squared
    distance of the POOLED (X;Y) set. With a tiny, tightly clustered target
    and a collapsed batch it shrinks toward 0 and the kernel gradient blows
    up; it also makes the bandwidth a function of the batch, so the
    gradient flows through it. Fixed / target / floored policies are immune."""
    torch.manual_seed(0)
    S_small = torch.tensor([[0.0], [1e-3]], dtype=torch.float64)          # tiny, tight target
    Y = torch.tensor([[0.5e-3], [0.4e-3]], dtype=torch.float64)
    pooled = DistributionalLoss(S_small, bandwidth="pooled")
    fixed = DistributionalLoss(S_small, bandwidth="fixed", bandwidth_value=1.0)
    floor = DistributionalLoss(S_small, bandwidth="pooled_floor", floor_frac=1.0)
    target = DistributionalLoss(S_small, bandwidth="target")
    g_pooled = _grad_norm(pooled, Y)
    pooled(Y)
    assert pooled.last_bandwidth < 1e-5
    assert g_pooled > 100 * _grad_norm(fixed, Y)
    floor(Y)
    assert floor.last_bandwidth >= target.target_bw * (1 - 1e-12)
    assert _grad_norm(floor, Y) < g_pooled / 2
    # n = 1 with the pooled rule: bandwidth driven entirely by the target, and
    # XX = k(0) is a constant -- the statistic still has a gradient through XY.
    g1 = _grad_norm(pooled, Y[:1])
    assert math.isfinite(g1)
    # gradient flows through the pooled bandwidth: detaching it changes the gradient
    Yr = Y.clone().requires_grad_(True)
    gp, = torch.autograd.grad(pooled(Yr), Yr)
    Z = torch.vstack([Yr.detach(), S_small])
    from tfg.distributional import pooled_bandwidth
    bw = float(pooled_bandwidth(Z))
    fixed_bw = DistributionalLoss(S_small, bandwidth="fixed", bandwidth_value=bw)
    gf, = torch.autograd.grad(fixed_bw(Yr), Yr)
    assert not torch.allclose(gp, gf, rtol=1e-3, atol=0)


def test_sqrt_transforms_gradient_bounds():
    """sqrt(|MMD^2| + eps) (SD code) has gradient ~ 1/(2 sqrt eps) at MMD^2 = 0;
    sqrt_floor bounds it by 1/(2 sqrt c) with c the V-statistic floor; mmd2 is
    smooth. All three agree asymptotically for large MMD^2."""
    torch.manual_seed(1)
    S = torch.randn(50, 1, dtype=torch.float64)
    Y_far = S + 5.0
    Y_near = S + 1e-3 * torch.randn(50, 1, dtype=torch.float64)   # MMD^2 ~ 1e-6
    for tr in ("mmd2", "sqrt_abs_eps", "sqrt_floor"):
        L = DistributionalLoss(S, bandwidth="target", transform=tr)
        assert math.isfinite(float(L(Y_far))) and float(L(Y_far)) > 0
    sd = DistributionalLoss(S, bandwidth="target", transform="sqrt_abs_eps", eps=1e-8)
    fl = DistributionalLoss(S, bandwidth="target", transform="sqrt_floor", floor_frac=0.1)
    raw = DistributionalLoss(S, bandwidth="target", transform="mmd2")
    # near the optimum the SD transform's gradient is much larger than the floored one
    assert _grad_norm(sd, Y_near) > 10 * _grad_norm(fl, Y_near)
    # floored transform -> same asymptote: sqrt(m2 + c) - sqrt(c) ~ sqrt(m2)
    m2 = float(raw(Y_far))
    assert abs(float(fl(Y_far)) - math.sqrt(m2)) / math.sqrt(m2) < 0.2
    assert abs(float(sd(Y_far)) - math.sqrt(m2)) < 1e-6


# -- 6. stale gradient ----------------------------------------------------------

def test_stale_gradient_reuse_skips_predictor_and_refresh_one_is_identity():
    x_def, eng_d, tr_d, lf_d = _engine(lambda c: (setattr(c.n_schedule, "enabled", True),
                                                  setattr(c.n_schedule, "n_max", 2)),
                                       log_f=NoisyLogF)

    def stale(k):
        def m(c):
            c.n_schedule.enabled = True
            c.n_schedule.n_max = 2
            c.temporal_cache.enabled = True
            c.temporal_cache.implementation = "stale"
            c.temporal_cache.refresh_every = k
        return m
    x1, eng1, tr1, lf1 = _engine(stale(1), log_f=NoisyLogF)
    assert torch.equal(x_def, x1) and lf1.samples == lf_d.samples
    x2, eng2, tr2, lf2 = _engine(stale(2), log_f=NoisyLogF)
    assert lf2.samples == 2 * T / 2 and eng2.counter.stale_steps == T // 2
    assert not torch.equal(x_def, x2)
    # on a stale step the applied gradient equals the previous step's
    g = tr2.records
    assert torch.equal(g[("grad_rho_raw", T - 1, 1, None)], g[("grad_rho_raw", T, 1, None)])
    assert not torch.equal(g[("grad_rho_raw", T - 2, 1, None)], g[("grad_rho_raw", T - 1, 1, None)])


# -- round 2: scale-free clipping rules -----------------------------------------

def _raw_norms(tr):
    return [float(tr.records[("grad_rho_raw", t, 1, None)].norm()) for t in range(T, 0, -1)]


def _used_norms(tr, rho=0.5):
    return [float(tr.records[("Delta_t", t, 1, None)].norm()) / rho for t in range(T, 0, -1)]


def test_relative_clip_median_and_ema_are_causal_and_scale_free():
    import statistics
    for ref in ("median", "ema"):
        _, eng, tr, _ = _engine(lambda c: (setattr(c.temporal, "grad_norm", "clip_rel"),
                                           setattr(c.temporal, "clip_ref", ref),
                                           setattr(c.temporal, "grad_clip", 0.5)))
        raw, used = _raw_norms(tr), _used_norms(tr)
        # raw history is the pre-clip norm; first step is never clipped
        assert eng._raw_norms == pytest.approx(raw, rel=1e-9)
        assert used[0] == pytest.approx(raw[0], rel=1e-9)
        ema = None
        for i in range(1, T):
            hist = raw[:i]
            if ref == "median":
                thr = 0.5 * sorted(hist)[len(hist) // 2]
            else:
                ema = hist[0] if ema is None else 0.9 * ema + 0.1 * hist[-1] if i > 1 else hist[0]
                # recompute ema explicitly
                e = hist[0]
                for h in hist[1:]:
                    e = 0.9 * e + 0.1 * h
                thr = 0.5 * e
            assert used[i] == pytest.approx(min(raw[i], thr), rel=1e-6)
    # scale-free: scaling log_f by 10 scales every raw gradient by 10 and the
    # clipping decision (clip or not) is identical
    sch, eps_theta, seed = _setup()
    decisions = []
    for scale in (1.0, 10.0):
        lf = make_quadratic_log_f([1.5, -0.75], 0.6 * scale)
        cfg = TFGConfig(T=T, rho_scalar=0.05 / scale)
        cfg.temporal.grad_norm, cfg.temporal.grad_clip = "clip_rel", 1.0
        tr = Tracer()
        GeneralizedTFG(eps_theta, lf, sch, NoiseTape(seed=seed), cfg).run(SHAPE, trace=tr)
        raw, used = _raw_norms(tr), _used_norms(tr, rho=0.05 / scale)
        decisions.append([u < r * (1 - 1e-9) for r, u in zip(raw, used)])
    assert decisions[0] == decisions[1] and any(decisions[0])


def test_quantile_clip():
    _, eng, tr, _ = _engine(lambda c: (setattr(c.temporal, "grad_norm", "clip_quantile"),
                                       setattr(c.temporal, "grad_clip", 0.5)))
    raw, used = _raw_norms(tr), _used_norms(tr)
    assert used[0] == pytest.approx(raw[0], rel=1e-9)
    for i in range(1, T):
        srt = sorted(raw[:i])
        thr = srt[min(len(srt) - 1, int(0.5 * len(srt)))]
        assert used[i] == pytest.approx(min(raw[i], thr), rel=1e-6)
    with pytest.raises(ValueError):
        c = TFGConfig(T=T); c.temporal.grad_norm = "clip_quantile"; c.temporal.grad_clip = 1.5
        c.validate()


def test_step_trust_region_noise_and_ddim():
    sch = DiffusionSchedule(T=T)
    for kind in ("noise", "ddim"):
        _, eng, tr, _ = _engine(lambda c: (setattr(c.temporal, "step_clip", kind),
                                           setattr(c.temporal, "step_tau", 0.05)), rho=5.0)
        for t in range(T, 0, -1):
            x_prev = tr.records[("x_prev", t, 1, None)]
            x_ddim = tr.records[("x_ddim", t, 1, None)]
            x_in = tr.records[("x_t_in", t, 1, None)]
            applied = (x_prev - x_ddim) * torch.sqrt(sch.alpha(t))     # undo line 9's 1/sqrt(alpha_t)
            ref = (0.05 * float(sch.sqrt_one_minus_ab(t)) if kind == "noise"
                   else 0.05 * float((x_ddim - x_in).norm()))
            assert float(applied.norm()) <= ref * (1 + 1e-6)
            raw_step = tr.records[("Delta_t", t, 1, None)]
            assert float(applied.norm()) == pytest.approx(min(float(raw_step.norm()), ref), rel=1e-6)
    # a huge tau is the identity
    x_def, _, _, _ = _engine(lambda c: None)
    x_big, _, _, _ = _engine(lambda c: (setattr(c.temporal, "step_clip", "noise"),
                                        setattr(c.temporal, "step_tau", 1e9)))
    assert torch.allclose(x_def, x_big, atol=1e-12, rtol=0)
    assert not TFGConfig(T=T, temporal=TFGConfig().temporal.__class__(step_clip="noise")).all_extensions_disabled()
