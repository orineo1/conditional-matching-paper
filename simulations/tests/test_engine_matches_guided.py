"""[A4] The engine reproduces ``experiments/_guided.py::run`` bit-for-bit.

``_guided.run`` is the standalone loop Experiments 2-7 were run through.  With
the opt-in legacy switches (``init="zeros"``, ``guidance_scaling="raw"``,
``smoothing="lgd_beta"``, ``n_schedule.eta_per_perturbation``), the
repository-formula schedule (``tfg.distributional.repository_schedule``) and
the legacy RNG replay (``LegacyTape`` + ``CMSampler(source="legacy")``),
``GeneralizedTFG`` produces the SAME final x and the SAME per-step
intermediates as ``_guided.run`` on the real checkpoints.

The pipeline is float32 end to end, exactly as the experiments ran it (the
checkpoints are float32 and ``_guided`` never sets a dtype).  The achieved
tolerance is 0.0 (bitwise); the test asserts <= 1e-6 as the brief requires and
reports the achieved value.

Needs the trained checkpoints in ``simulations/artifacts/checkpoints``
(skipped if absent; ``experiments/_models.py`` trains them on first use).
"""
import sys
from pathlib import Path

import pytest
import torch

SIM = Path(__file__).resolve().parents[1]
for p in (SIM / "experiments", SIM.parents[0] / "experiments" / "model-optimization" / "estimator"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tfg.distributional import CMSampler, repository_schedule, sample_with_noise  # noqa: E402
from tfg.noise_tape import NoiseTape                                            # noqa: E402

TOL = 1e-6
CKPT = SIM / "artifacts" / "checkpoints"
ARMS = [("no_lgd", "none"), ("no_lgd", "adam"), ("lgd", "none")]
RESTARTS = [0, 1, 2]
N = 8


@pytest.fixture(autouse=True)
def _float32_default():
    """Runs AFTER conftest's float64 autouse fixture (same scope, later
    definition) and pins the experiments' dtype for every test here."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)       # the experiments' dtype
    yield
    torch.set_default_dtype(prev)


@pytest.fixture(scope="module")
def setup():
    if not (CKPT / "cm_seed20240401_dx1dy1.pt").exists():
        pytest.skip("2D checkpoints absent")
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        from engine_runner import build_models
        return build_models("2D")
    finally:
        torch.set_default_dtype(prev)


def _guided_with_recorder(mc, mu, S_G, bw, n, spatial, temporal, restart):
    """Run _guided.run while recording its DDIM intermediates per step, by
    wrapping the model method it calls (no change to _guided.py)."""
    from _guided import run as guided_run
    rec = {"x_in": {}, "x_ddim": {}, "x0": {}}
    orig = mu.sample_ddim_step

    def wrapped(x_start, t, *a, **kw):
        x_prev, pred_x0 = orig(x_start, t, *a, **kw)
        rec["x_in"][int(t)] = x_start.detach().clone()
        rec["x_ddim"][int(t)] = x_prev.detach().clone()
        rec["x0"][int(t)] = pred_x0.detach().clone()
        return x_prev, pred_x0

    mu.sample_ddim_step = wrapped
    try:
        x, info = guided_run(mc, mu, S_G, bw, n, spatial, temporal, restart)
    finally:
        del mu.sample_ddim_step       # restore the bound method
    return x, info, rec


def _engine_with_recorder(mc, mu, S_G, bw, n, spatial, temporal, restart):
    from engine_runner import run_engine
    from tfg.engine import GeneralizedTFG
    rec = {"x_in": {}, "x_ddim": {}, "x0": {}}
    orig_run = GeneralizedTFG.run

    def run_rec(self, shape, trace=None):
        def tr(name, t, r, k, tensor):
            if name == "x_t_in":
                rec["x_in"][int(t)] = tensor.detach().clone()
            elif name == "x_ddim":
                rec["x_ddim"][int(t)] = tensor.detach().clone()
            elif name == "x0_pred":
                rec["x0"][int(t)] = tensor.detach().clone()
            if trace is not None:
                trace(name, t, r, k, tensor)
        return orig_run(self, shape, trace=tr)

    GeneralizedTFG.run = run_rec
    try:
        x, info = run_engine(mc, mu, S_G, bw, n, spatial, temporal, restart, rng="legacy")
    finally:
        GeneralizedTFG.run = orig_run
    return x, info, rec


@pytest.mark.parametrize("spatial,temporal", ARMS)
def test_engine_matches_guided_final_and_intermediates(setup, spatial, temporal):
    params, S_G, bw, mc, mu = setup
    worst = 0.0
    for r in RESTARTS:
        xg, ig, rg = _guided_with_recorder(mc, mu, S_G, bw, N, spatial, temporal, r)
        xe, ie, re = _engine_with_recorder(mc, mu, S_G, bw, N, spatial, temporal, r)
        assert ig["diverged"] == ie["diverged"]
        assert ig["conditional_calls"] == ie["cm_samples"] == ie["conditional_calls"]
        d_final = float((xg.double() - xe.double()).abs().max())
        worst = max(worst, d_final)
        assert d_final <= TOL, (spatial, temporal, r, d_final)
        assert set(rg["x_in"]) == set(re["x_in"]) == set(range(1, mu.diffusion_steps))
        for name in ("x_in", "x_ddim", "x0"):
            for t in rg[name]:
                d = float((rg[name][t].double() - re[name][t].double()).abs().max())
                worst = max(worst, d)
                assert d <= TOL, (spatial, temporal, r, name, t, d)
    print(f"\n[{spatial}/{temporal}] achieved max abs diff over finals and "
          f"per-step intermediates: {worst:.3e}")


def test_engine_matches_guided_corrected_protocol(setup):
    """Protocol correction of 2026-08-24: x_T ~ N(0,I) (per-restart generator),
    zeta = 8, noise-level trust region. Engine (--x-init randn --zeta 8
    --step-clip noise) must still equal _guided.run(x_init="randn", zeta=8,
    step_clip="noise") bit-for-bit, for none and adam."""
    import inspect
    from _guided import run as guided_run
    if "x_init" not in inspect.signature(guided_run).parameters:
        pytest.skip("_guided.run has no x_init (pre-correction checkout)")
    from engine_runner import run_engine
    params, S_G, bw, mc, mu = setup
    worst = 0.0
    for temporal, zeta in (("none", 8.0), ("adam", 0.125)):
        for r in (0, 1):
            xg, ig = guided_run(mc, mu, S_G, bw, N, "no_lgd", temporal, r, zeta=zeta,
                                step_clip="noise", step_tau=1.0, x_init="randn")
            xe, ie = run_engine(mc, mu, S_G, bw, N, "no_lgd", temporal, r, rng="legacy",
                                x_init="randn", zeta=zeta, step_clip="noise", step_tau=1.0)
            assert ig["diverged"] == ie["diverged"]
            assert ie["protocol"]["x_init"] == "randn" and ie["protocol"]["zeta"] == zeta
            d = float((xg.double() - xe.double()).abs().max())
            worst = max(worst, d)
            assert d <= TOL, (temporal, r, d)
    x0, _ = run_engine(mc, mu, S_G, bw, N, "no_lgd", "none", 0, rng="legacy", x_init="randn")
    x1, _ = run_engine(mc, mu, S_G, bw, N, "no_lgd", "none", 1, rng="legacy", x_init="randn")
    assert float((x0 - x1).abs().max()) > 1e-3       # different x_T per restart
    print(f"\n[corrected protocol] achieved max abs diff: {worst:.3e}")


def test_the_comparison_is_not_vacuous(setup):
    """Different restarts / arms must give different answers, otherwise the
    equality above would be meaningless."""
    from engine_runner import run_engine
    params, S_G, bw, mc, mu = setup
    x0, _ = run_engine(mc, mu, S_G, bw, N, "no_lgd", "none", 0, rng="legacy")
    x1, _ = run_engine(mc, mu, S_G, bw, N, "no_lgd", "none", 1, rng="legacy")
    xa, _ = run_engine(mc, mu, S_G, bw, N, "no_lgd", "adam", 0, rng="legacy")
    assert float((x0 - x1).abs().max()) > 1e-3
    assert float((x0 - xa).abs().max()) > 1e-3


def test_repository_schedule_matches_model_bitwise(setup):
    params, S_G, bw, mc, mu = setup
    sch = repository_schedule(mu, dtype=torch.float32)
    assert sch.matches_model(mu)
    assert sch.T == mu.diffusion_steps - 1
    sch64 = repository_schedule(mu, dtype=torch.float64)
    assert sch64.alphabar.dtype == torch.float64
    # a rebuild, not a cast: float64 differs from the float32 values by O(1e-7)
    d = float((sch64.alphabar - mu.baralphas.double()).abs().max())
    assert 0.0 < d < 1e-6


def test_tape_keyed_sampler_agrees_with_manual_seed_path(setup):
    """The two RNG paths agree: feeding the sampler the noise that
    ``torch.manual_seed(s)`` + ``model.sample`` would draw gives the same
    samples, and the tape path is the same arithmetic with tape noise."""
    from _models import PAPER_TS
    params, S_G, bw, mc, mu = setup
    n, d_y = 8, mc.nfeatures
    cond = torch.full((n, 1), -1.5)
    torch.manual_seed(12345)
    y_ref, _, _ = mc.sample(nsamples=n, condition_x=cond, ts=PAPER_TS)
    g = torch.Generator().manual_seed(12345)
    Z = torch.stack([torch.randn(n, d_y, generator=g) for _ in PAPER_TS])
    y = sample_with_noise(mc, cond, PAPER_TS, Z)
    assert torch.equal(y_ref, y)
    tape = NoiseTape(seed=7, dtype=torch.float32)
    smp = CMSampler(mc, PAPER_TS, tape, source="tape")
    keys = [("eta", 5, i) for i in range(n)]
    y_t = smp(torch.tensor([[-1.5]]), keys)
    Zt = torch.stack([tape.randn(k, (len(PAPER_TS), d_y), dtype=torch.float32) for k in keys], dim=1)
    assert torch.equal(y_t, sample_with_noise(mc, cond, PAPER_TS, Zt))
    assert smp.cm_samples == n
    # antithetic: second half is the negated noise of the first half
    smp2 = CMSampler(mc, PAPER_TS, tape, source="tape", antithetic=True)
    y_a = smp2(torch.tensor([[-1.5]]), keys)
    Za = torch.cat([Zt[:, :n // 2], -Zt[:, :n // 2]], dim=1)
    assert torch.equal(y_a, sample_with_noise(mc, cond, PAPER_TS, Za))
