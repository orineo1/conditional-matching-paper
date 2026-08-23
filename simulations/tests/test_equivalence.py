"""Checkpoint 1: numerical equivalence of the generalised engine and TFG.

PROVENANCE CAVEAT, restated here because it bounds what these tests prove.
No TFG implementation existed in this repository before this work (only the
LGD subspace in ``Optimization.py:88``).  ``tfg/reference_tfg.py`` is therefore
*new* code, and these tests compare two new implementations of Algorithm 1
rather than checking a new engine against established code.  They are a strong
check on internal consistency and on the conventions C1-C5, and a weak check on
whether both share a misreading of the paper.  The mitigations are that the two
modules were written to be mathematically independent (no shared update-step
helpers) and that all eleven intermediate tensors are compared per (t, r)
rather than only the final output.

Test A -- generic TFG reduction, with an ordinary target predictor.
Test B -- the one-sample distributional predictor as a TFG instance.
"""

import itertools
import math

import pytest
import torch

from conftest import ToyDenoiser, make_quadratic_log_f
from tfg.config import NScheduleConfig, TFGConfig
from tfg.engine import GeneralizedTFG
from tfg.noise_tape import NoiseTape, compare_access
from tfg.reference import run_reference_tfg
from tfg.schedule import DiffusionSchedule, constant_vector
from tfg.trace import Tracer, compare_traces, format_report

# float64 on CPU: the brief allows 1e-7, but the two implementations should
# agree far more tightly than that. We assert the tight bound and report the
# achieved value, rather than loosening to the permitted ceiling.
TOL = 1e-12

SHAPE = (1, 2)
T_SMALL = 8


def _setup(T=T_SMALL, seed=17):
    schedule = DiffusionSchedule(T=T)
    denoiser = ToyDenoiser(d=SHAPE[1], T=T, seed=0, schedule=schedule)
    eps_theta = lambda x, t: denoiser(x, t)
    log_f = make_quadratic_log_f([1.5, -0.75], scale=0.6)
    return schedule, eps_theta, log_f, seed


def _run_pair(N_recur, N_iter, gamma_bar, rho_scalar, mu_scalar,
              n_mc=1, T=T_SMALL, seed=17, engine_config_mutator=None):
    """Run reference and engine on identical tapes; return both tracers."""
    schedule, eps_theta, log_f, seed = _setup(T, seed)

    rho = constant_vector(rho_scalar, T)
    mu = constant_vector(mu_scalar, T)

    tape_ref = NoiseTape(seed=seed)
    tr_ref = Tracer()
    x_ref = run_reference_tfg(
        eps_theta, log_f, schedule, tape_ref, SHAPE,
        N_recur=N_recur, N_iter=N_iter, rho=rho, mu=mu,
        gamma_bar=gamma_bar, n_mc=n_mc, trace=tr_ref,
    )

    cfg = TFGConfig(
        T=T, N_recur=N_recur, N_iter=N_iter, gamma_bar=gamma_bar,
        rho_scalar=rho_scalar, mu_scalar=mu_scalar,
        rho_structure="constant", mu_structure="constant", n_mc=n_mc,
    )
    if engine_config_mutator is not None:
        engine_config_mutator(cfg)

    tape_eng = NoiseTape(seed=seed)
    tr_eng = Tracer()
    engine = GeneralizedTFG(eps_theta, log_f, schedule, tape_eng, cfg)
    x_eng = engine.run(SHAPE, trace=tr_eng)

    return tr_ref, tr_eng, tape_ref, tape_eng, x_ref, x_eng, engine


# ---------------------------------------------------------------------------
# Test A: generic TFG reduction
# ---------------------------------------------------------------------------

MODES = {
    "rho_only": (0.9, 0.0),
    "mu_only": (0.0, 0.35),
    "both": (0.9, 0.35),
}

CELLS = []
for N_recur, N_iter, gamma_bar, mode in itertools.product(
        (1, 2), (0, 1, 4), (0.0, 0.35), ("rho_only", "mu_only", "both")):
    if N_iter == 0 and mode == "mu_only":
        continue          # no guidance at all; asserted separately below
    if N_iter == 0 and mode == "both":
        continue          # identical to rho_only; cross-checked separately
    CELLS.append((N_recur, N_iter, gamma_bar, mode))


@pytest.mark.parametrize("N_recur,N_iter,gamma_bar,mode", CELLS)
def test_A_engine_matches_reference(N_recur, N_iter, gamma_bar, mode):
    rho_scalar, mu_scalar = MODES[mode]
    n_mc = 3 if gamma_bar != 0.0 else 1
    tr_ref, tr_eng, tape_ref, tape_eng, x_ref, x_eng, _ = _run_pair(
        N_recur, N_iter, gamma_bar, rho_scalar, mu_scalar, n_mc=n_mc)

    only_ref, only_eng = compare_access(tape_ref, tape_eng)
    assert not only_ref and not only_eng, (
        f"tape key sets differ: only_reference={only_ref[:5]} only_engine={only_eng[:5]}"
    )

    ok, report = compare_traces(tr_ref, tr_eng, atol=TOL)
    assert ok, format_report(report)
    assert torch.allclose(x_ref, x_eng, rtol=0.0, atol=TOL)


def test_A_all_intermediate_names_are_actually_compared():
    """Guard against a vacuous pass: every named quantity must be present.

    If a rename silently dropped ``grad_rho_raw`` from one side, the comparison
    would still 'pass' on the remaining keys. This asserts the full list.
    """
    tr_ref, tr_eng, _, _, _, _, _ = _run_pair(
        N_recur=2, N_iter=2, gamma_bar=0.3, rho_scalar=0.9, mu_scalar=0.35, n_mc=2)
    names = {k[0] for k in tr_ref.keys()}
    required = {
        "x_T", "x_t_in", "eps_theta", "x0_pred", "log_f_tilde_rho",
        "grad_rho_raw", "Delta_t", "grad_mu_raw", "Delta_0_iter", "Delta_0",
        "x_ddim", "x_prev", "renoise_eps", "x_t_out", "x_0",
    }
    assert required <= names, f"missing traced quantities: {sorted(required - names)}"
    assert names == {k[0] for k in tr_eng.keys()}


@pytest.mark.parametrize("N_recur", [1, 2])
@pytest.mark.parametrize("gamma_bar", [0.0, 0.35])
def test_A_no_guidance_is_plain_ddim(N_recur, gamma_bar):
    """N_iter=0 with mu-only: Delta_0 == 0 and rho == 0, so guidance vanishes.

    Both engines must then reproduce unguided DDIM exactly, which proves the
    guidance plumbing has no side effects on the sampling path.
    """
    tr_ref, tr_eng, _, _, x_ref, x_eng, _ = _run_pair(
        N_recur, 0, gamma_bar, rho_scalar=0.0, mu_scalar=0.35, n_mc=1)
    ok, report = compare_traces(tr_ref, tr_eng, atol=TOL)
    assert ok, format_report(report)

    # Compare x_ddim against an INDEPENDENT hand-written DDIM iterate. The
    # previous version only asserted x_prev == x_ddim, which is true by algebra
    # (Delta_t = 0*grad, Delta_0 = zeros) and therefore tested nothing: x_ddim
    # itself could have been arbitrarily wrong.
    schedule, eps_theta, _, _ = _setup()
    for key, value in tr_ref.records.items():
        if key[0] != "x_ddim":
            continue
        t, r = key[1], key[2]
        x_in = tr_ref.records[("x_t_in", t, r, None)]
        ab_t = schedule.alphabar[t]
        ab_prev = schedule.alphabar[t - 1]
        eps = eps_theta(x_in, t)
        x0 = (x_in - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
        expected = torch.sqrt(ab_prev) * x0 + torch.sqrt(1 - ab_prev) * eps
        assert torch.allclose(value, expected.detach(), rtol=0.0, atol=1e-14), (
            f"x_ddim at t={t}, r={r} does not match an independent DDIM iterate"
        )


@pytest.mark.parametrize("N_recur", [1, 2])
@pytest.mark.parametrize("gamma_bar", [0.0, 0.35])
def test_A_niter0_both_equals_rho_only(N_recur, gamma_bar):
    """With N_iter=0 the mu branch cannot act, so 'both' == 'rho only'."""
    # index 0 is the REFERENCE tracer; an earlier version unpacked index 1 and
    # compared engine-vs-engine, silently losing half the coverage.
    tr_both, _, _, _, x_both, _, _ = _run_pair(N_recur, 0, gamma_bar, 0.9, 0.35)
    tr_rho, _, _, _, x_rho, _, _ = _run_pair(N_recur, 0, gamma_bar, 0.9, 0.0)
    ok, report = compare_traces(tr_both, tr_rho, atol=0.0,
                                label_a="both", label_b="rho_only")
    assert ok, format_report(report)


def test_A_nrecur1_consumes_no_renoise_key():
    """Convention C2: no re-noising draw may be consumed when N_recur == 1."""
    _, _, tape_ref, tape_eng, _, _, _ = _run_pair(1, 1, 0.0, 0.9, 0.35)
    for tape, name in ((tape_ref, "reference"), (tape_eng, "engine")):
        renoise = [k for k in tape.requested_keys() if k[0] == "renoise"]
        assert renoise == [], f"{name} consumed re-noising keys with N_recur=1: {renoise}"


def test_A_delta_key_carries_no_recurrence_index():
    """Convention C1: the gamma_bar smoothing noise is drawn once per outer t."""
    _, _, tape_ref, tape_eng, _, _, _ = _run_pair(
        N_recur=3, N_iter=2, gamma_bar=0.3, rho_scalar=0.9, mu_scalar=0.35, n_mc=2)
    for tape, name in ((tape_ref, "reference"), (tape_eng, "engine")):
        deltas = {k for k in tape.requested_keys() if k[0] == "delta"}
        # (t, j) only -> exactly T * n_mc distinct keys regardless of N_recur/N_iter
        assert len(deltas) == T_SMALL * 2, (
            f"{name}: expected {T_SMALL * 2} delta keys, got {len(deltas)}; "
            "a recurrence or inner-iteration index has leaked into the key"
        )


def test_A_extra_cell_deep_recurrence():
    """N_recur=3, N_iter=2: catches r vs r+1 index mixing that N_recur=2 misses."""
    tr_ref, tr_eng, _, _, x_ref, x_eng, _ = _run_pair(
        N_recur=3, N_iter=2, gamma_bar=0.4, rho_scalar=0.9, mu_scalar=0.35, n_mc=2)
    ok, report = compare_traces(tr_ref, tr_eng, atol=TOL)
    assert ok, format_report(report)


@pytest.mark.parametrize("T", [2, 3])
def test_A_short_horizon_boundary(T):
    """t=1 -> t=0 boundary, where alphabar_0 == 1 and sqrt(1-alphabar_0) == 0."""
    tr_ref, tr_eng, _, _, x_ref, x_eng, _ = _run_pair(
        N_recur=2, N_iter=1, gamma_bar=0.3, rho_scalar=0.9, mu_scalar=0.35,
        n_mc=2, T=T)
    ok, report = compare_traces(tr_ref, tr_eng, atol=TOL)
    assert ok, format_report(report)
    assert torch.isfinite(x_ref).all() and torch.isfinite(x_eng).all()


def test_A_gamma_zero_is_independent_of_n_mc():
    """With gamma_bar=0 the smoothing collapses, so n_mc must not matter.

    This catches a missing 1/n normalisation or a logsumexp-vs-mean bug.
    """
    _, _, _, _, x1, _, _ = _run_pair(1, 1, 0.0, 0.9, 0.35, n_mc=1)
    _, _, _, _, x5, _, _ = _run_pair(1, 1, 0.0, 0.9, 0.35, n_mc=5)
    assert torch.allclose(x1, x5, rtol=0.0, atol=TOL)


def test_A_guidance_actually_moves_the_sample():
    """Anti-vacuity: if guidance were a no-op everywhere, everything above
    would pass trivially. Confirm rho and mu each change the outcome."""
    _, _, _, _, x_off, _, _ = _run_pair(1, 0, 0.0, 0.0, 0.0)
    _, _, _, _, x_rho, _, _ = _run_pair(1, 0, 0.0, 0.9, 0.0)
    _, _, _, _, x_mu, _, _ = _run_pair(1, 2, 0.0, 0.0, 0.35)
    assert (x_rho - x_off).abs().max().item() > 1e-6
    assert (x_mu - x_off).abs().max().item() > 1e-6


def test_A_disabled_extensions_flag_is_true():
    cfg = TFGConfig(T=T_SMALL)
    assert cfg.all_extensions_disabled()


def test_A_neutral_n_schedule_does_not_change_results():
    """Component 1 switched on but neutral (constant, n_max=1) must be inert."""
    def mutate(cfg):
        cfg.n_schedule = NScheduleConfig(enabled=True, type="constant", n_max=1)

    schedule, eps_theta, log_f, seed = _setup()

    class Wrapped:
        """Accepts both call signatures so one predictor serves both engines."""
        def __call__(self, x, n_t=None, eta_keys=None):
            return log_f(x)

    wrapped = Wrapped()
    rho = constant_vector(0.9, T_SMALL)
    mu = constant_vector(0.35, T_SMALL)

    tr_ref = Tracer()
    run_reference_tfg(eps_theta, log_f, schedule, NoiseTape(seed=seed), SHAPE,
                      N_recur=2, N_iter=1, rho=rho, mu=mu, gamma_bar=0.3,
                      n_mc=2, trace=tr_ref)

    cfg = TFGConfig(T=T_SMALL, N_recur=2, N_iter=1, gamma_bar=0.3,
                    rho_scalar=0.9, mu_scalar=0.35, n_mc=2)
    mutate(cfg)
    tr_eng = Tracer()
    GeneralizedTFG(eps_theta, wrapped, schedule, NoiseTape(seed=seed), cfg).run(
        SHAPE, trace=tr_eng)

    ok, report = compare_traces(tr_ref, tr_eng, atol=TOL)
    assert ok, format_report(report)


def test_A_unimplemented_extensions_are_refused():
    cfg = TFGConfig(T=T_SMALL)
    cfg.temporal_cache.enabled = True
    with pytest.raises(NotImplementedError, match="temporal_cache"):
        cfg.validate()

    cfg = TFGConfig(T=T_SMALL)
    cfg.adaptive_recurrence.enabled = True
    with pytest.raises(NotImplementedError, match="adaptive_recurrence"):
        cfg.validate()


# ---------------------------------------------------------------------------
# Test B: the one-sample distributional predictor as a TFG instance
# ---------------------------------------------------------------------------

def _mmd_loss():
    from LossFunctions import MMDLoss, RBF
    return MMDLoss(kernel=RBF(device="cpu"), device="cpu")


def test_B_mmd_is_the_squared_v_statistic():
    """Verify MMDLoss returns MMD^2 (biased V-statistic), not MMD.

    Required before writing exp(-beta * L): if L were MMD rather than MMD^2,
    the composite predictor would not be the f_eta of the brief. Checked two
    ways -- against a hand-written V-statistic, and by confirming it is NOT the
    square root of that.
    """
    from LossFunctions import RBF

    torch.manual_seed(0)
    X = torch.randn(6, 2, dtype=torch.float64)
    Y = torch.randn(9, 2, dtype=torch.float64) + 1.0

    kernel = RBF(device="cpu")
    K = kernel(torch.vstack([X, Y]))
    n = X.shape[0]

    # Biased V-statistic: full blocks including the diagonal, divided by n^2.
    v_stat = (K[:n, :n].sum() / (n * n)
              - 2.0 * K[:n, n:].sum() / (n * Y.shape[0])
              + K[n:, n:].sum() / (Y.shape[0] ** 2))

    got = _mmd_loss()(X, Y)
    assert torch.allclose(got, v_stat, rtol=0, atol=1e-12), (
        f"MMDLoss={float(got):.12e} does not equal the biased V-statistic "
        f"{float(v_stat):.12e}"
    )
    assert not torch.allclose(got, torch.sqrt(v_stat.clamp_min(0)), atol=1e-8), (
        "MMDLoss appears to return the square root; the composite predictor "
        "would then be exp(-beta*MMD), not exp(-beta*MMD^2)"
    )

    # Unbiased U-statistic would exclude the diagonal; confirm it does not.
    u_xx = (K[:n, :n].sum() - K[:n, :n].diagonal().sum()) / (n * (n - 1))
    assert not torch.allclose(K[:n, :n].mean(), u_xx, atol=1e-10), (
        "XX block appears to exclude the diagonal; estimator is not the "
        "biased V-statistic assumed by the Checkpoint 0 report"
    )


class CompositePredictor:
    """f_eta(x) = exp(-beta * MMD_hat^2({h_phi(x, eta_1)}, S_G)), n_max = 1.

    Returns ``log f_eta`` directly as ``-beta * MMD^2``; never
    ``log(exp(...))``, which underflows to -inf for large beta*MMD^2.

    Frozen-eta convention: ``eta_1`` is drawn once at construction and reused
    for every evaluation. This makes f_eta a deterministic function of x, so
    any mismatch between the two engines is unambiguously a guidance-engine
    bug rather than predictor noise. A tape-keyed eta is possible but cannot
    be used for cross-engine equivalence, because the two engines evaluate the
    predictor a different number of times whenever N_iter differs, so they
    would request different key sets by construction.
    """

    def __init__(self, S_G, beta=1.0, eta=None, d_y=1, seed=3):
        self.S_G = S_G
        self.beta = float(beta)
        self.d_y = d_y
        self.mmd = _mmd_loss()
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(2, d_y, generator=g, dtype=torch.float64)
        self.eta = (torch.randn(1, d_y, generator=g, dtype=torch.float64)
                    if eta is None else eta)
        self.n_calls = 0

    def h_phi(self, x):
        """One conditional sample given conditioning x; differentiable in x."""
        mean = torch.tanh(x @ self.W)
        return mean + 0.3 * self.eta

    def __call__(self, x, n_t=None, eta_keys=None):
        self.n_calls += 1
        samples = self.h_phi(x)
        return -self.beta * self.mmd(samples, self.S_G)


def _make_S_G(seed=1, m=24, d_y=1):
    g = torch.Generator().manual_seed(seed)
    return 0.5 * torch.randn(m, d_y, generator=g, dtype=torch.float64) + 0.4


@pytest.mark.parametrize("N_recur,N_iter,gamma_bar", [
    (1, 0, 0.0),
    (1, 1, 0.0),
    (2, 1, 0.0),
    (1, 2, 0.3),
    (2, 2, 0.3),
])
def test_B_composite_predictor_matches_across_engines(N_recur, N_iter, gamma_bar):
    schedule, eps_theta, _, seed = _setup()
    S_G = _make_S_G()
    predictor = CompositePredictor(S_G, beta=1.0)

    rho = constant_vector(0.5, T_SMALL)
    mu = constant_vector(0.2, T_SMALL)
    n_mc = 2 if gamma_bar != 0.0 else 1

    tr_ref = Tracer()
    x_ref = run_reference_tfg(
        eps_theta, predictor, schedule, NoiseTape(seed=seed), SHAPE,
        N_recur=N_recur, N_iter=N_iter, rho=rho, mu=mu,
        gamma_bar=gamma_bar, n_mc=n_mc, trace=tr_ref)

    cfg = TFGConfig(T=T_SMALL, N_recur=N_recur, N_iter=N_iter,
                    gamma_bar=gamma_bar, rho_scalar=0.5, mu_scalar=0.2,
                    n_mc=n_mc)
    cfg.n_schedule = NScheduleConfig(enabled=True, type="constant", n_max=1)

    tr_eng = Tracer()
    engine = GeneralizedTFG(eps_theta, predictor, schedule,
                            NoiseTape(seed=seed), cfg)
    x_eng = engine.run(SHAPE, trace=tr_eng)

    ok, report = compare_traces(tr_ref, tr_eng, atol=TOL)
    assert ok, format_report(report)
    assert torch.allclose(x_ref, x_eng, rtol=0.0, atol=TOL)
    # NOTE: this cell has n_schedule ENABLED (n_max=1), so it is a
    # neutral-extension cell, not an all-extensions-disabled cell. It is
    # counted separately in the Checkpoint 1 report.
    assert engine.counter.predictor_evals > 0


def test_B_predictor_is_differentiable_and_finite():
    S_G = _make_S_G()
    predictor = CompositePredictor(S_G, beta=1.0)
    x = torch.randn(1, 2, dtype=torch.float64, requires_grad=True)
    lf = predictor(x)
    assert torch.isfinite(lf), "log f_eta must be finite"
    g, = torch.autograd.grad(lf, x)
    assert torch.isfinite(g).all() and g.abs().max() > 0


def test_B_large_beta_does_not_underflow():
    """log f is computed analytically, so a large beta must stay finite."""
    S_G = _make_S_G()
    predictor = CompositePredictor(S_G, beta=1e4)
    x = torch.randn(1, 2, dtype=torch.float64, requires_grad=True)
    lf = predictor(x)
    assert torch.isfinite(lf)
    g, = torch.autograd.grad(lf, x)
    assert torch.isfinite(g).all()
