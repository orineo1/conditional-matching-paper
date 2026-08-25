"""[Agent M] Sample-replay MMD (tfg/replay.py): counts, buffer, determinism,
fresh-only gradient, weighted-vs-subsampled agreement, engine integration.

Run: cd simulations && python -m pytest tests/test_replay.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

SIM = Path(__file__).resolve().parents[1]
for p in (SIM / "experiments", SIM.parents[0] / "experiments" / "model-optimization" / "estimator"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tfg.config import ReplayConfig, TFGConfig                     # noqa: E402
from tfg.distributional import DistributionalLoss                  # noqa: E402
from tfg.noise_tape import NoiseTape                               # noqa: E402
from tfg.replay import (ReplayBuffer, replay_counts, subsample_rows,  # noqa: E402
                        weighted_mmd2, wrap_log_f)

CKPT = SIM / "artifacts" / "checkpoints"
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="no checkpoints")


# ---------------------------------------------------------------------------
# helpers: a differentiable fake sampler + fixed-bandwidth loss
# ---------------------------------------------------------------------------

def make_setup(f=6, d=2, m=60, seed=0, backend="reference"):
    g = torch.Generator().manual_seed(seed)
    S = torch.randn(m, d, generator=g, dtype=torch.float64) * 1.3 + 0.4
    A = torch.randn(f, d, generator=g, dtype=torch.float64)
    B_off = torch.randn(f, d, generator=g, dtype=torch.float64)

    def sampler(x, keys):
        assert len(keys) == f
        return x.reshape(1, -1) * A + B_off        # differentiable in x

    loss = DistributionalLoss(S, bandwidth="fixed", bandwidth_value=1.7,
                              backend=backend)
    return sampler, loss, S


def keys_for(t, f, j=0):
    return [("eta", t, j, i) for i in range(f)]


# ---------------------------------------------------------------------------
# counts and buffer
# ---------------------------------------------------------------------------

def test_replay_counts_sum_and_ori_split():
    for B, lam, D in [(8, 0.5, 3), (32, 0.7, 5), (4, 0.3, 3), (250, 3 / 7, 1)]:
        c = replay_counts(B, lam, D)
        assert sum(c) == B and all(x >= 0 for x in c) and c[0] >= 1
        assert c[0] == max(c)
    # Ori's reuse_frac=0.3 at nsamples in {8, 250}: n_reuse = round(0.3 n)
    assert replay_counts(8, 0.3 / 0.7, 1) == [6, 2]
    assert replay_counts(250, 0.3 / 0.7, 1) == [175, 75]
    # decay 0 = replay exactly off
    assert replay_counts(8, 0.0, 4) == [8, 0, 0, 0, 0]


def test_buffer_depth_eviction_and_replacement():
    buf = ReplayBuffer(depth=2)
    r = {t: torch.full((3, 2), float(t)) for t in (9, 8, 7, 6)}
    for t in (9, 8, 7):
        buf.push(t, 0, r[t])
    ents = buf.entries(6, 0)                      # steps 7 and 8 (k=1,2), 9 evicted
    assert [k for k, _ in ents] == [1, 2]
    assert float(ents[0][1][0, 0]) == 7.0 and float(ents[1][1][0, 0]) == 8.0
    buf.push(7, 0, r[6])                          # recurrence: replaces, no dup
    assert float(buf.entries(6, 0)[0][1][0, 0]) == 6.0
    assert buf.entries(6, 1) == []                # per-j isolation
    # stored rows are detached
    leaf = torch.randn(3, 2, requires_grad=True)
    buf.push(5, 0, leaf * 2)
    assert not buf.entries(4, 0)[0][1].requires_grad


# ---------------------------------------------------------------------------
# off / decay=0 identity, determinism
# ---------------------------------------------------------------------------

def run_sequence(rcfg, seed=11, steps=(9, 8, 7, 6, 5), f=6, x0=0.3):
    sampler, loss, _ = make_setup(f=f)
    tape = NoiseTape(seed=seed)
    lf = wrap_log_f(sampler, loss, tape, rcfg)
    outs, grads = [], []
    for t in steps:
        x = torch.full((1, 2), x0 + 0.01 * t, dtype=torch.float64,
                       requires_grad=True)
        v = lf(x, n_t=f, eta_keys=keys_for(t, f))
        (g,) = torch.autograd.grad(v, x)
        outs.append(float(v))
        grads.append(g)
    return outs, grads


def test_disabled_and_decay0_identical_to_plain():
    base_o, base_g = run_sequence(ReplayConfig(enabled=False))
    off_o, off_g = run_sequence(ReplayConfig(enabled=True, decay=0.0, depth=3,
                                             batch_total=12))
    for a, b in zip(base_o, off_o):
        assert a == b
    for a, b in zip(base_g, off_g):
        assert torch.equal(a, b)


def test_determinism_given_tape_seed():
    r = ReplayConfig(enabled=True, mode="subsample", decay=0.5, depth=2)
    o1, g1 = run_sequence(r, seed=11)
    o2, g2 = run_sequence(r, seed=11)
    o3, _ = run_sequence(r, seed=12)
    assert o1 == o2 and all(torch.equal(a, b) for a, b in zip(g1, g2))
    assert o1[0] == o3[0]                  # first step has no replay -> equal
    # (fake sampler is tape-independent; subsampling begins at the 2nd step,
    #  but caches smaller than the request are taken whole -> augment counts
    #  round(6*0.5)=3 < 6 rows: subsampled; check some later step differs)
    assert o1[2:] != o3[2:]


def test_subsample_rows_is_tape_keyed():
    tape = NoiseTape(seed=3)
    C = torch.arange(20, dtype=torch.float64).reshape(10, 2)
    a = subsample_rows(C, 4, tape, ("replay", 5, 0, 1))
    b = subsample_rows(C, 4, tape, ("replay", 5, 0, 1))
    c = subsample_rows(C, 4, tape, ("replay", 5, 0, 2))
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert subsample_rows(C, 10, tape, ("replay", 4, 0, 1)) is C
    # rows are actual members, no repetition
    rows = {tuple(r.tolist()) for r in a}
    assert len(rows) == 4 and all(tuple(r.tolist()) in
                                  {tuple(q.tolist()) for q in C} for r in a)


# ---------------------------------------------------------------------------
# gradient: only fresh rows differentiable; assembler == manual batch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["reference", "fast"])
def test_fresh_only_gradient_matches_manual_batch(backend):
    f = 5
    sampler, loss, _ = make_setup(f=f, backend=backend)
    tape = NoiseTape(seed=7)
    rcfg = ReplayConfig(enabled=True, mode="subsample", decay=0.5, depth=1,
                        batch_total=8)
    # batch_total=8, decay=0.5, depth=1 -> counts [5, 3]
    assert replay_counts(8, 0.5, 1) == [5, 3]
    lf = wrap_log_f(sampler, loss, tape, rcfg)
    x9 = torch.tensor([[0.2, -0.4]], dtype=torch.float64, requires_grad=True)
    lf(x9, n_t=f, eta_keys=keys_for(9, f))                 # fills the buffer
    cache = lf.buffer.entries(8, 0)[0][1]
    x8 = torch.tensor([[0.1, 0.5]], dtype=torch.float64, requires_grad=True)
    v = lf(x8, n_t=f, eta_keys=keys_for(8, f))
    (g,) = torch.autograd.grad(v, x8, retain_graph=False)
    # manual: same fresh rows differentiable + the SAME subsampled constants
    sel = subsample_rows(cache, 3, tape, ("replay", 8, 0, 1))
    x8b = x8.detach().clone().requires_grad_(True)
    fresh = sampler(x8b, keys_for(8, f))
    v_manual = -loss(torch.cat([fresh, sel], dim=0))
    (g_manual,) = torch.autograd.grad(v_manual, x8b)
    assert abs(float(v) - float(v_manual)) <= 1e-12
    assert float((g - g_manual).abs().max()) <= 1e-12
    # and the replayed rows really are constants: perturbing the buffer rows
    # changes the value but the gradient path exists only through x
    assert not cache.requires_grad


def test_weighted_uniform_equals_unweighted_both_backends():
    g = torch.Generator().manual_seed(5)
    X = torch.randn(9, 2, generator=g, dtype=torch.float64)
    for backend in ("reference", "fast"):
        _, loss, _ = make_setup(backend=backend)
        w = torch.full((9,), 1.0 / 9, dtype=torch.float64)
        a = float(weighted_mmd2(loss, X, w))
        b = float(loss(X))
        assert abs(a - b) <= 1e-12
    with pytest.raises(NotImplementedError):
        S = torch.randn(30, 2, generator=g, dtype=torch.float64)
        weighted_mmd2(DistributionalLoss(S, bandwidth="target"), X, w)


def test_weighted_matches_subsampled_expectation():
    """mean over tape keys of the subsampled V-stat == weighted V-stat + the
    exact within-group diagonal correction Delta (THEORY.md section 1), to
    within 5 standard errors; and the correction formula itself is exact."""
    torch.manual_seed(0)
    f, c, r, d = 6, 40, 6, 2
    F = torch.randn(f, d, dtype=torch.float64)
    C = torch.randn(c, d, dtype=torch.float64) * 1.1 + 0.2
    B = f + r
    _, loss, S = make_setup(f=f, d=d)
    # weighted with group weights matching the subsample marginals (f/B, r/B)
    w = torch.cat([torch.full((f,), 1.0 / B, dtype=torch.float64),
                   torch.full((c,), r / (B * c), dtype=torch.float64)])
    L_w = float(weighted_mmd2(loss, torch.cat([F, C]), w))
    # exact expectation of the subsampled stat differs only in the CC block:
    K = loss.kernel(C)
    S_diag = float(K.diagonal().sum())
    S_all = float(K.sum())
    S_off = S_all - S_diag
    E_cc = (r * (r - 1) / (c * (c - 1)) * S_off + (r / c) * S_diag) / B ** 2
    W_cc = (r / B) ** 2 * S_all / c ** 2
    E_sub_exact = L_w - W_cc + E_cc
    # Monte Carlo over tape keys
    tape = NoiseTape(seed=99)
    vals = []
    for i in range(400):
        sel = subsample_rows(C, r, tape, ("replay", i, 0, 1))
        vals.append(float(loss(torch.cat([F, sel]))))
    vals = torch.tensor(vals)
    se = float(vals.std() / len(vals) ** 0.5)
    assert abs(float(vals.mean()) - E_sub_exact) <= 5 * se + 1e-12
    # the diagonal correction is small but real
    assert abs(E_sub_exact - L_w) <= 0.5 * abs(L_w) + 0.05


# ---------------------------------------------------------------------------
# config surface
# ---------------------------------------------------------------------------

def test_config_defaults_and_validation():
    cfg = TFGConfig()
    assert cfg.replay.enabled is False
    assert cfg.all_extensions_disabled()          # replay is outside the engine
    ReplayConfig().validate()
    with pytest.raises(ValueError):
        ReplayConfig(mode="magic").validate()
    with pytest.raises(ValueError):
        ReplayConfig(decay=1.0).validate()
    with pytest.raises(ValueError):
        ReplayConfig(depth=0).validate()
    sampler, loss, _ = make_setup()
    lf = wrap_log_f(sampler, loss, NoiseTape(seed=1),
                    ReplayConfig(enabled=True))
    with pytest.raises(ValueError):
        lf(torch.zeros(1, 2, dtype=torch.float64))   # unkeyed protocol


# ---------------------------------------------------------------------------
# engine integration (real 2D checkpoints)
# ---------------------------------------------------------------------------

@pytest.fixture
def f32():
    """The experiments run float32 end to end (conftest's autouse fixture sets
    float64, which would rebuild the checkpointed models in float64)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(prev)


@needs_ckpt
def test_engine_replay_decay0_identical_to_baseline(f32):
    from engine_runner import build_models, run_engine

    def mut(cfg):
        cfg.replay.enabled = True
        cfg.replay.decay = 0.0
        cfg.replay.depth = 3

    _, S_G, bw, mc, mu = build_models("2D")
    xa, ia = run_engine(mc, mu, S_G, bw, 4, "no_lgd", "none", 0)
    xb, ib = run_engine(mc, mu, S_G, bw, 4, "no_lgd", "none", 0, cfg_mutator=mut)
    assert torch.equal(xa, xb)
    assert ia["cm_samples"] == ib["cm_samples"]


@needs_ckpt
def test_engine_replay30_saves_calls_and_runs(f32):
    from engine_runner import build_models, run_engine
    _, S_G, bw, mc, mu = build_models("2D")
    xb, ib = run_engine(mc, mu, S_G, bw, 8, "no_lgd", "none", 0)
    x, i = run_engine(mc, mu, S_G, bw, 8, "no_lgd", "none", 0,
                      candidate="replay30")
    T = i["steps"]
    assert i["cm_samples"] == 6 * T and ib["cm_samples"] == 8 * T
    assert torch.isfinite(x).all() and not torch.equal(x, xb)
    # augment arm: same calls as baseline
    xa, ia = run_engine(mc, mu, S_G, bw, 8, "no_lgd", "none", 0,
                        candidate="replay_geo0.5d3_aug")
    assert ia["cm_samples"] == 8 * T
    # weighted arm parses, runs, saves the same calls as its subsample twin
    xw, iw = run_engine(mc, mu, S_G, bw, 8, "no_lgd", "none", 0,
                        candidate="replay_w30")
    assert iw["cm_samples"] == 6 * T and torch.isfinite(xw).all()


# ---------------------------------------------------------------------------
# [M-9] fixed-batch top-up (ReplayConfig.fill)
# ---------------------------------------------------------------------------

def test_fill_counts_plan_and_edges():
    from tfg.replay import fill_counts
    # the registered M-9 plans (B=8, decay 0.7, depth 5)
    assert fill_counts(8, 1, 0.7, 5) == [1, 2, 2, 1, 1, 1]
    assert fill_counts(8, 2, 0.7, 5) == [2, 2, 1, 1, 1, 1]
    assert fill_counts(8, 4, 0.7, 5) == [4, 1, 1, 1, 1, 0]
    for f in (1, 2, 4):
        assert sum(fill_counts(8, f, 0.7, 5)) == 8
    # no recycling when the batch is already full, or decay 0
    assert fill_counts(8, 8, 0.7, 5) == [8, 0, 0, 0, 0, 0]
    assert fill_counts(4, 8, 0.7, 5) == [8, 0, 0, 0, 0, 0]
    assert fill_counts(8, 2, 0.0, 3) == [2, 0, 0, 0]
    with pytest.raises(ValueError):
        fill_counts(8, 0, 0.7, 5)
    # config validation
    with pytest.raises(ValueError):
        ReplayConfig(enabled=True, fill=True, batch_total=0).validate()
    with pytest.raises(ValueError):
        ReplayConfig(enabled=True, fill=True, batch_total=8,
                     mode="weighted").validate()


def test_fill_availability_clamp_f1():
    """f=1: each buffered step holds ONE row, so the realised batch ramps
    1,2,..,6 and is capped at f + depth = 6 < B = 8 (registered caveat)."""
    f = 1
    sampler, loss, _ = make_setup(f=f)
    tape = NoiseTape(seed=4)
    rcfg = ReplayConfig(enabled=True, mode="subsample", decay=0.7, depth=5,
                        batch_total=8, fill=True)
    seen = []
    real_loss = loss

    class SpyLoss:
        bandwidth, transform, _fast = "fixed", "mmd2", None
        def __call__(self, Y):
            seen.append(Y.shape[0])
            return real_loss(Y)
    lf = wrap_log_f(sampler, SpyLoss(), tape, rcfg)
    for i, t in enumerate(range(20, 10, -1)):
        x = torch.full((1, 2), 0.1 + 0.01 * t, dtype=torch.float64,
                       requires_grad=True)
        lf(x, n_t=f, eta_keys=keys_for(t, f))
    assert seen == [1, 2, 3, 4, 5, 6, 6, 6, 6, 6]


def test_fill_f2_reaches_full_batch_and_grad_is_fresh_only():
    f = 2
    sampler, loss, _ = make_setup(f=f)
    tape = NoiseTape(seed=6)
    rcfg = ReplayConfig(enabled=True, mode="subsample", decay=0.7, depth=5,
                        batch_total=8, fill=True)
    lf = wrap_log_f(sampler, loss, tape, rcfg)
    for t in range(20, 14, -1):        # warm up the depth-5 buffer
        x = torch.full((1, 2), 0.1, dtype=torch.float64, requires_grad=True)
        v = lf(x, n_t=f, eta_keys=keys_for(t, f))
    # steady state: batch 8 = 2 fresh + (2,1,1,1,1) recycled
    ents = lf.buffer.entries(14, 0)
    assert [k for k, _ in ents] == [1, 2, 3, 4, 5]
    x = torch.full((1, 2), 0.1, dtype=torch.float64, requires_grad=True)
    v = lf(x, n_t=f, eta_keys=keys_for(14, f))
    (g,) = torch.autograd.grad(v, x)
    assert torch.isfinite(g).all() and float(g.abs().sum()) > 0


def test_f1_mmd_is_well_defined_fixed_bandwidth():
    """n=1 fresh-only: XX block is the constant k(0) (no gradient); the loss
    and its gradient stay finite for both backends (registered caveat ii)."""
    for backend in ("reference", "fast"):
        _, loss, _ = make_setup(f=1, backend=backend)
        x = torch.tensor([[0.3, -0.2]], dtype=torch.float64, requires_grad=True)
        v = loss(x)
        (g,) = torch.autograd.grad(v, x)
        assert torch.isfinite(v) and torch.isfinite(g).all()
        # XX term of a 1-row batch contributes no gradient: the gradient
        # equals that of the XY(+const) part alone
        x2 = x.detach().clone().requires_grad_(True)
        S = loss.S_G.to(torch.float64)
        K = loss.kernel(torch.vstack([x2, S]))
        v2 = -2.0 * K[:1, 1:].mean()
        (g2,) = torch.autograd.grad(v2, x2)
        assert float((g - g2).abs().max()) <= 1e-12


@needs_ckpt
def test_engine_fill_candidate_calls_accounting(f32):
    from engine_runner import build_models, run_engine
    _, S_G, bw, mc, mu = build_models("2D")
    x, i = run_engine(mc, mu, S_G, bw, 2, "no_lgd", "none", 0,
                      candidate="replay_fill8_geo0.7d5_trust")
    T = i["steps"]
    assert i["cm_samples"] == 2 * T            # fresh stays f = n = 2
    assert torch.isfinite(x).all()
    x1, i1 = run_engine(mc, mu, S_G, bw, 1, "no_lgd", "none", 0,
                        candidate="replay_fill8_geo0.7d5_trust")
    assert i1["cm_samples"] == 1 * T and torch.isfinite(x1).all()


# ---------------------------------------------------------------------------
# [M-10] fifo / cohort buffer policies
# ---------------------------------------------------------------------------

def test_fifo_and_cohort_plans():
    from tfg.replay import cohort_counts, fifo_counts
    # the registered M-10 plans
    assert fifo_counts(8, 2) == [2, 2, 2, 2]
    assert fifo_counts(16, 2) == [2] + [2] * 7
    assert fifo_counts(8, 4) == [4, 4]
    assert fifo_counts(16, 4) == [4, 4, 4, 4]
    assert cohort_counts(8, 2) == [2, 2, 2, 1, 1]
    assert cohort_counts(16, 2) == [2, 2, 2, 2] + [1] * 8
    assert cohort_counts(16, 4) == [4, 4, 4, 2, 1, 1]
    # registered degeneracy: cohort8 == fifo8 at f=4
    assert cohort_counts(8, 4) == fifo_counts(8, 4) == [4, 4]
    # sums equal the batch, no recycling when B <= f
    for fn in (fifo_counts, cohort_counts):
        for B, f in [(8, 2), (16, 2), (8, 4), (16, 4)]:
            assert sum(fn(B, f)) == B
        assert fn(4, 8) == [8]
        assert fn(2, 2) == [2]
    # depth padding
    assert fifo_counts(8, 4, depth=5) == [4, 4, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        fifo_counts(8, 0)
    with pytest.raises(ValueError):
        ReplayConfig(policy="lru").validate()


def test_policy_dispatch_and_depth_guard():
    f = 2
    sampler, loss, _ = make_setup(f=f)
    # depth too small for the plan -> explicit error, not silent truncation
    rc = ReplayConfig(enabled=True, mode="subsample", batch_total=16, fill=True,
                      policy="cohort", depth=3)   # cohort16@f=2 needs depth 11
    lf = wrap_log_f(sampler, loss, NoiseTape(seed=2), rc)
    with pytest.raises(ValueError, match="depth"):
        lf(torch.zeros(1, 2, dtype=torch.float64), n_t=f,
           eta_keys=keys_for(30, f))   # fails fast: the plan is static
    # fifo: batch ramps f, 2f, .. and saturates at B; full cohorts -> no
    # subsampling, so two seeds give identical outputs (determinism trivially)
    seen = []
    real = loss

    class Spy:
        bandwidth, transform, _fast = "fixed", "mmd2", None
        def __call__(self, Y):
            seen.append(Y.shape[0])
            return real(Y)
    rc2 = ReplayConfig(enabled=True, mode="subsample", batch_total=8, fill=True,
                       policy="fifo", depth=3)
    lf2 = wrap_log_f(sampler, Spy(), NoiseTape(seed=2), rc2)
    for t in range(40, 33, -1):
        x = torch.full((1, 2), 0.05, dtype=torch.float64, requires_grad=True)
        v = lf2(x, n_t=f, eta_keys=keys_for(t, f))
        (g,) = torch.autograd.grad(v, x)
        assert torch.isfinite(g).all()
    assert seen == [2, 4, 6, 8, 8, 8, 8]
    # cohort with partial cohorts subsamples via the tape -> deterministic
    def run(seed):
        rc3 = ReplayConfig(enabled=True, mode="subsample", batch_total=8,
                           fill=True, policy="cohort", depth=4)
        lf3 = wrap_log_f(sampler, real, NoiseTape(seed=seed), rc3)
        out = []
        for t in range(40, 33, -1):
            x = torch.full((1, 2), 0.05 + 0.001 * t, dtype=torch.float64,
                           requires_grad=True)
            out.append(float(lf3(x, n_t=f, eta_keys=keys_for(t, f))))
        return out
    assert run(5) == run(5)
    assert run(5)[3:] != run(6)[3:]        # partial-cohort subsample differs


@needs_ckpt
def test_engine_fifo_cohort_candidates(f32):
    from engine_runner import build_models, run_engine
    _, S_G, bw, mc, mu = build_models("2D")
    for cand in ("replay_fifo8_trust", "replay_cohort16_trust"):
        x, i = run_engine(mc, mu, S_G, bw, 2, "no_lgd", "none", 0, candidate=cand)
        assert i["cm_samples"] == 2 * i["steps"], cand   # fresh stays f = 2
        assert torch.isfinite(x).all()
