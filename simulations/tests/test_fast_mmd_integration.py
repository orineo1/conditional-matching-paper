"""[A4 integration] tfg.fast_mmd.MMDFixedTarget == LossFunctions.MMDLoss, and the
engine runner's --loss fast path reproduces --loss reference."""
import sys
from pathlib import Path

import pytest
import torch

SIM = Path(__file__).resolve().parents[1]
for p in (SIM / "experiments", SIM.parents[0] / "experiments" / "model-optimization" / "estimator"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from LossFunctions import MMDLoss, RBF          # noqa: E402  (ot stub via conftest)
from tfg.fast_mmd import MMDFixedTarget         # noqa: E402

CKPT = SIM / "artifacts" / "checkpoints"


@pytest.mark.parametrize("n,m,d", [(1, 250, 1), (4, 250, 1), (32, 250, 3), (7, 50, 2), (250, 8, 2)])
@pytest.mark.parametrize("bw", [None, 0.7, 3.0])
def test_value_and_gradient_match_reference_1e12(n, m, d, bw):
    g = torch.Generator().manual_seed(n * 1000 + m + int(bool(bw)))
    X = torch.randn(n, d, generator=g, dtype=torch.float64)
    Y = torch.randn(m, d, generator=g, dtype=torch.float64) * 1.5 + 0.3
    ref = MMDLoss(kernel=RBF(bandwidth=bw))
    fast = MMDFixedTarget(Y, bandwidth=bw)
    Xr = X.clone().requires_grad_(True)
    Xf = X.clone().requires_grad_(True)
    vr, vf = ref(Xr, Y), fast(Xf)
    assert abs(float(vr) - float(vf)) <= 1e-12 * max(1.0, abs(float(vr)))
    gr, = torch.autograd.grad(vr, Xr)
    gf, = torch.autograd.grad(vf, Xf)
    assert float((gr - gf).abs().max()) <= 1e-12 * max(1.0, float(gr.abs().max()))


def test_distributional_loss_fast_backend_equals_reference():
    from tfg.distributional import DistributionalLoss
    g = torch.Generator().manual_seed(3)
    S = torch.randn(120, 2, generator=g, dtype=torch.float64)
    Y = torch.randn(6, 2, generator=g, dtype=torch.float64).requires_grad_(True)
    for kw in ({"bandwidth": "fixed", "bandwidth_value": 1.3}, {"bandwidth": "target"},
               {"bandwidth": "pooled"}):
        a = DistributionalLoss(S, backend="reference", **kw)(Y)
        b = DistributionalLoss(S, backend="fast", **kw)(Y)
        assert abs(float(a) - float(b)) <= 1e-12
        ga, = torch.autograd.grad(a, Y)
        gb, = torch.autograd.grad(b, Y)
        assert float((ga - gb).abs().max()) <= 1e-12
    with pytest.raises(ValueError):
        DistributionalLoss(S, backend="fast", bandwidth="pooled_floor")


@pytest.fixture
def f32():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(prev)


def test_engine_runner_fast_matches_reference_teacher_forced(f32):
    """Per-step gradients with the SAME x_t (teacher forced from the reference
    run's trajectory) agree to 1e-5 in float32, and the two full trajectories
    agree up to float32 round-off (checked loosely)."""
    if not (CKPT / "cm_seed20240401_dx1dy1.pt").exists():
        pytest.skip("2D checkpoints absent")
    from engine_runner import build_models, run_engine
    params, S_G, bw, mc, mu = build_models("2D")
    worst_g, worst_x = 0.0, 0.0
    for r in (0, 1):
        xr, ir = run_engine(mc, mu, S_G, bw, 8, "no_lgd", "none", r, trace_steps=True,
                            loss_backend="reference")
        xf, i_f = run_engine(mc, mu, S_G, bw, 8, "no_lgd", "none", r, trace_steps=True,
                             loss_backend="fast")
        assert ir["cm_samples"] == i_f["cm_samples"]
        # teacher-forced: evaluate the fast gradient at the reference x_t of step t
        from tfg.distributional import CMSampler, DistributionalLoss, repository_schedule
        from tfg.noise_tape import NoiseTape
        from _models import PAPER_TS
        sch = repository_schedule(mu, dtype=torch.float32)
        for t in (99, 60, 20, 1):
            x_in = (ir["x_prev_trace"].get((t + 1, 1)) if t < sch.T else torch.zeros(1, 1))
            grads = []
            for backend in ("reference", "fast"):
                tape = NoiseTape(seed=r, dtype=torch.float32)
                smp = CMSampler(mc, PAPER_TS, tape, source="tape", dtype=torch.float32)
                loss = DistributionalLoss(S_G, bandwidth="fixed", bandwidth_value=bw, backend=backend)
                xx = x_in.clone().requires_grad_(True)
                eps = mu(xx, torch.full([1, 1], t), None)
                x0 = (xx - sch.sqrt_one_minus_ab(t) * eps) / sch.sqrt_ab(t)
                keys = [("eta", t, 0, i) for i in range(8)]
                g, = torch.autograd.grad(loss(smp(x0, keys)), xx)
                grads.append(g)
            d = float((grads[0] - grads[1]).abs().max())
            worst_g = max(worst_g, d)
            assert d <= 1e-5 * max(1.0, float(grads[0].abs().max())), (r, t, d)
        worst_x = max(worst_x, float((xr - xf).abs().max()))
    # full trajectories: float32 round-off is amplified by the chaotic loop; document
    print(f"\nteacher-forced max grad diff {worst_g:.2e}; final-x diff {worst_x:.2e}")
    assert worst_x < 1.0
