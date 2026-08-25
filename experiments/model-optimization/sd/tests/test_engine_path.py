"""CPU tests for the opt-in ``--engine tfg`` path (src/tfg_engine_path.py):
schedule mapping, config construction, and a toy end-to-end equivalence of the
engine trajectory with a hand-rolled legacy loop (DDIM step + -zeta*grad,
optional trust cap) using mocked architect/objective."""
import copy
import math
import os
import sys
import types

import pytest
import torch

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "SD_cond_SD_controlnet", "src")
sys.path.insert(0, os.path.abspath(_SRC))
try:
    from diffusers import DDIMScheduler
except Exception:  # pragma: no cover
    DDIMScheduler = None
try:
    import torchvision  # noqa: F401
except ImportError:  # pragma: no cover
    _tv = types.ModuleType("torchvision"); _tr = types.ModuleType("torchvision.transforms")
    _tf = types.ModuleType("torchvision.transforms.functional")
    _tv.transforms = _tr; _tr.functional = _tf
    sys.modules.update({"torchvision": _tv, "torchvision.transforms": _tr,
                        "torchvision.transforms.functional": _tf})

import tfg_engine_path as ep  # noqa: E402
from profiling import StepProfiler  # noqa: E402
from trust import apply_trust, prev_alpha_bar, trust_cap  # noqa: E402

pytestmark = pytest.mark.skipif(DDIMScheduler is None, reason="diffusers not available")


def _sched(n_steps=8):
    # SDXL-base scheduler config: no sample clipping, set_alpha_to_one=False, leading spacing
    s = DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear",
                      beta_start=0.00085, beta_end=0.012, clip_sample=False,
                      set_alpha_to_one=False, steps_offset=1, timestep_spacing="leading")
    s.set_timesteps(n_steps)
    return s


def test_sd_schedule_matches_ddim_table():
    s = _sched(8)
    ts = s.timesteps[3:]
    sch = ep.SDSchedule(s, ts, "cpu")
    assert sch.T == len(ts)
    for i, t in enumerate(ts):
        assert float(sch.ab(sch.T - i)) == pytest.approx(float(s.alphas_cumprod[int(t)]), rel=1e-6)
        assert sch.diffusers_t(sch.T - i) == int(t)
    assert float(sch.ab(0)) == float(s.final_alpha_cumprod)
    # engine's abar_{t-1} == the legacy prev_alpha_bar at every guided step
    for i, t in enumerate(ts):
        assert float(sch.ab(sch.T - i - 1)) == pytest.approx(prev_alpha_bar(s, t), rel=1e-6)


def _args(**kw):
    a = types.SimpleNamespace(seed=3, num_variations=4, variation_batch_size=1, base_zeta=2.0,
                              trust_noise=0.0, backsel=0, backsel_rule="uniform",
                              backsel_weighting="ht", backsel_soft_tau_scale=1.0,
                              backsel_soft_tau_mode="local", guidance_scale=0.0,
                              arch_single_batch=True, sprinter_variation_prompt="", loss_scale=1.0)
    a.__dict__.update(kw)
    return a


def test_build_config_validates():
    cfg = ep.build_config(5, 4, _args(trust_noise=0.25, backsel=2, backsel_rule="strat"), 0.03)
    assert cfg.temporal.step_clip == "noise_prev_rms" and cfg.temporal.step_tau == 0.25
    assert cfg.backsel.rule == "stratified_balanced" and cfg.n_schedule.n_max == 4
    assert cfg.guidance_scaling == "raw" and cfg.rho_scalar == 1.0


class _Fakes:
    """Toy architect: eps = A x_t (linear); objective L(x0) = ||x0 - c||^2 + 1 (x0 = decoded 'pixels')."""

    def __init__(self, sched, shape):
        g = torch.Generator().manual_seed(0)
        self.A = 0.1 * torch.randn(shape.numel(), shape.numel(), generator=g)
        self.c = torch.randn(shape, generator=g)
        self.scheduler = sched
        self.vae = types.SimpleNamespace(
            config=types.SimpleNamespace(scaling_factor=0.13),
            dtype=torch.float32,
            decode=lambda lat: types.SimpleNamespace(sample=lat * 2.0 - 1.0))
        self.unet = None

    def eps(self, x):
        return (self.A @ x.reshape(-1)).reshape(x.shape)

    def objective(self, pixel_x0_norm, base_zeta):
        L = ((pixel_x0_norm - self.c) ** 2).sum() + 1.0
        return L, base_zeta / L.detach()


@pytest.mark.parametrize("tau", [0.0, 0.02])
def test_engine_path_matches_legacy_loop_on_toy(monkeypatch, tau):
    sched = _sched(8)
    ts = sched.timesteps[2:]
    shape = torch.Size([1, 2, 4, 4])
    fk = _Fakes(sched, shape)
    args = _args(trust_noise=tau)

    monkeypatch.setattr(ep, "predict_noise_cfg",
                        lambda unet, s, x, t, enc, add, gs, single_batch=False: fk.eps(x))

    def fake_objective(**kw):
        L, zeta = fk.objective(kw["pixel_x0_norm"], kw["base_zeta_prime"])
        return L, L.detach(), zeta, L.detach(), None, None
    monkeypatch.setattr(ep, "variation_objective", fake_objective)

    x_init = torch.randn(shape, generator=torch.Generator().manual_seed(1))
    prof = StepProfiler(enabled=False)
    x_eng, x_reg, recs = ep.run_engine_loop(
        args=args, architect=fk, sprinter=fk, clip_model=None, clip_processor=None, loss_fn=None,
        all_clip_embeddings=None, latents_init=x_init, latents_regular_init=x_init,
        cfg_encoder_states=None, added_cond_kwargs=None, timesteps_to_run=ts,
        scheduler_regular=copy.deepcopy(sched), prof=prof, device="cpu")
    x_eng = x_eng.float()

    # hand-rolled legacy loop (run_mlgd_f.py main loop with the same toy pieces)
    from generation import compute_pred_x0_direct, denoise_step
    x = x_init.clone()
    xr = x_init.clone()
    sch_g, sch_r = copy.deepcopy(sched), copy.deepcopy(sched)
    ac0 = float(sched.alphas_cumprod[0])
    for t in ts:
        xs = x.detach().requires_grad_(True)
        eps = fk.eps(xs)
        x0 = compute_pred_x0_direct(sch_g, eps, t, xs)
        px = torch.clamp((fk.vae.decode(x0 / 0.13).sample + 1.0) / 2.0, 0.0, 1.0)
        L, zeta = fk.objective(px, args.base_zeta)
        grad, = torch.autograd.grad(L, xs)
        corr = -zeta * grad
        if tau > 0:
            abar_prev = min(prev_alpha_bar(sch_g, t), ac0)
            corr, _, _ = apply_trust(corr, trust_cap(tau, abar_prev, corr.numel()))
        x = denoise_step(sch_g, eps, t, xs, correction=corr)
        with torch.no_grad():
            xr = denoise_step(sch_r, fk.eps(xr), t, xr)

    assert torch.allclose(x_eng, x, atol=1e-4, rtol=1e-4), (x_eng - x).abs().max()
    assert torch.allclose(x_reg.float(), xr, atol=1e-6)
    assert len(recs) == len(ts) and recs[0]["step"] == 1
    if tau > 0:
        assert any(r["trust_scale"] < 1.0 for r in recs) or all(
            r["correction_norm_raw"] <= r["trust_cap"] + 1e-6 for r in recs)
