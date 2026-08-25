"""
CPU unit tests for the opt-in SD flags (no diffusers pipelines, no GPU).

Run:  /Users/stolk/miniconda3/bin/python -m pytest experiments/model-optimization/sd/tests -q
"""
import math
import os
import sys

import pytest
import torch

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "SD_cond_SD_controlnet", "src")
sys.path.insert(0, os.path.abspath(_SRC))

# diffusers must be imported BEFORE the torchvision stub below (its import checks find_spec).
try:
    from diffusers import DDIMScheduler
except Exception:  # pragma: no cover
    DDIMScheduler = None

# metrics.py imports torchvision only for the eval helper; stub it if absent (CPU box).
try:
    import torchvision  # noqa: F401
except ImportError:  # pragma: no cover
    import types
    _tv = types.ModuleType("torchvision"); _tr = types.ModuleType("torchvision.transforms")
    _tf = types.ModuleType("torchvision.transforms.functional")
    _tv.transforms = _tr; _tr.functional = _tf
    sys.modules.update({"torchvision": _tv, "torchvision.transforms": _tr,
                        "torchvision.transforms.functional": _tf})

from backsel import IS_FLOOR, select_backprop_set, kcenter_greedy  # noqa: E402
from metrics import compute_mmd  # noqa: E402
from trust import apply_trust, prev_alpha_bar, trust_cap  # noqa: E402
from generation import predict_noise_cfg  # noqa: E402


# --------------------------------------------------------------------------
# back-selection
# --------------------------------------------------------------------------
def _toy(n=12, d=6, seed=0):
    """Toy differentiable 'sampler': e_i(x) = normalise(A_i x + b_i)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, d, 4, generator=g, dtype=torch.float64)
    b = torch.randn(n, d, generator=g, dtype=torch.float64)
    T = torch.randn(30, d, generator=g, dtype=torch.float64)
    T = T / T.norm(dim=1, keepdim=True)
    x = torch.randn(4, generator=g, dtype=torch.float64).requires_grad_(True)

    def E(x):
        e = torch.einsum("ndk,k->nd", A, x) + b
        return e / e.norm(dim=1, keepdim=True)
    return x, E, T


def _full_grad(x, E, T):
    loss = compute_mmd(E(x), T, bandwidth=torch.tensor(1.0, dtype=torch.float64))
    return torch.autograd.grad(loss, x)[0]


def _backsel_grad(x, E, T, k, rule, gen):
    with torch.no_grad():
        Emat = E(x)
    leaf = Emat.clone().requires_grad_(True)
    loss = compute_mmd(leaf, T, bandwidth=torch.tensor(1.0, dtype=torch.float64))
    g = torch.autograd.grad(loss, leaf)[0]
    idx, G, _ = select_backprop_set(g, Emat, k, rule, gen)
    e_sel = E(x)[idx]
    return torch.autograd.grad((G * e_sel).sum(), x)[0]


@pytest.mark.parametrize("rule", ["uniform", "is", "kcenter", "strat"])
def test_k_ge_n_is_exact(rule):
    x, E, T = _toy()
    full = _full_grad(x, E, T)
    got = _backsel_grad(x, E, T, k=12, rule=rule, gen=torch.Generator().manual_seed(1))
    assert torch.allclose(full, got, atol=1e-12)
    got = _backsel_grad(x, E, T, k=50, rule=rule, gen=torch.Generator().manual_seed(1))
    assert torch.allclose(full, got, atol=1e-12)


@pytest.mark.parametrize("rule", ["uniform", "is", "strat"])
def test_unbiased(rule):
    x, E, T = _toy()
    full = _full_grad(x, E, T)
    R = 4000
    acc = torch.zeros_like(full)
    sq = torch.zeros_like(full)
    for r in range(R):
        gr = _backsel_grad(x, E, T, k=4, rule=rule, gen=torch.Generator().manual_seed(r))
        acc += gr
        sq += gr ** 2
    mean = acc / R
    se = ((sq / R - mean ** 2).clamp_min(0) / R).sqrt()
    z = ((mean - full) / (se + 1e-15)).abs().max().item()
    assert z < 4.5, f"{rule}: max |z| = {z}"


def test_uniform_weights_and_sizes():
    g = torch.randn(10, 3)
    idx, G, info = select_backprop_set(g, g, 3, "uniform", torch.Generator().manual_seed(0))
    assert idx.numel() == 3 and len(set(idx.tolist())) == 3
    assert torch.allclose(G, g[idx] * (10 / 3))


def test_is_weights_probabilities():
    g = torch.randn(10, 3)
    g[0] *= 50  # heavy row
    idx, G, info = select_backprop_set(g, g, 4, "is", torch.Generator().manual_seed(0))
    p = torch.tensor(info["p"])
    assert abs(p.sum().item() - 1) < 1e-6
    assert p.min().item() >= IS_FLOOR / 10 - 1e-12
    assert 0 in idx.tolist()  # heavy row selected with p ~ 0.75+
    assert idx.numel() <= 4
    # weights c_i/(k p_i)
    for j, i in enumerate(idx.tolist()):
        w = info["weights"][j]
        assert torch.allclose(G[j].double(), g[i].double() * w)


def test_kcenter_aggregates_all_gradient_mass():
    g = torch.randn(20, 5)
    E = torch.randn(20, 5)
    idx, G, info = select_backprop_set(g, E, 4, "kcenter", torch.Generator().manual_seed(0))
    assert idx.numel() == 4
    assert sum(info["cluster_sizes"]) == 20
    assert torch.allclose(G.sum(0), g.sum(0), atol=1e-5)  # 100% of output-gradient mass kept
    # deterministic given the start row: the same generator seed reproduces the selection
    idx2, G2, _ = select_backprop_set(g, E, 4, "kcenter", torch.Generator().manual_seed(0))
    assert torch.equal(idx, idx2) and torch.allclose(G, G2)
    centers, assign = kcenter_greedy(E, 4, 3)
    assert len(set(centers.tolist())) == 4 and assign.max() < 4


def test_is_zero_gradient_falls_back_to_uniform_probs():
    g = torch.zeros(6, 2)
    idx, G, info = select_backprop_set(g, g, 2, "is", torch.Generator().manual_seed(0))
    assert all(abs(p - 1 / 6) < 1e-12 for p in info["p"])


# --------------------------------------------------------------------------
# trust region
# --------------------------------------------------------------------------
def test_trust_cap_formula_and_apply():
    cap = trust_cap(1.0, abar_prev=0.75, numel=16384)
    assert abs(cap - 0.5 * 128) < 1e-9
    v = torch.ones(4, 64, 64) * 3.0  # norm = 3*128 = 384 > cap 64
    out, scale, norm = apply_trust(v, cap)
    assert abs(norm - 384) < 1e-4
    assert abs(out.norm().item() - cap) < 1e-3
    assert torch.allclose(out / out.norm(), v / v.norm())
    small = torch.ones(4, 64, 64) * 0.1  # norm 12.8 < cap
    out, scale, _ = apply_trust(small, cap)
    assert scale == 1.0 and out is small
    out, scale, _ = apply_trust(v, 0.0)  # cap 0 = disabled
    assert scale == 1.0


def test_prev_alpha_bar_matches_ddim_step():
    if DDIMScheduler is None:
        pytest.skip("diffusers not available")
    sch = DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear",
                                  beta_start=0.00085, beta_end=0.012)
    sch.set_timesteps(10)
    ts = sch.timesteps
    for t in ts:
        prev_t = int(t) - 1000 // 10
        expect = float(sch.alphas_cumprod[prev_t]) if prev_t >= 0 else float(sch.final_alpha_cumprod)
        assert prev_alpha_bar(sch, t) == expect
    # the last step lands on final_alpha_cumprod = alphas_cumprod[0] ~ 1 (set_alpha_to_one=True default)
    assert trust_cap(1.0, prev_alpha_bar(sch, ts[-1]), 16384) < 1e-6
    # cap decreases monotonically along the trajectory (noise shrinks)
    caps = [trust_cap(1.0, prev_alpha_bar(sch, t), 16384) for t in ts]
    assert all(a >= b for a, b in zip(caps, caps[1:]))


# --------------------------------------------------------------------------
# single-batch CFG
# --------------------------------------------------------------------------
class _FakeUNet:
    """Per-sample deterministic map: out = lat * f(enc) + text_embeds mean."""
    def __call__(self, lmi, t, encoder_hidden_states, added_cond_kwargs, return_dict):
        scale = encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1)
        shift = added_cond_kwargs["text_embeds"].mean(dim=1).view(-1, 1, 1, 1)
        return (lmi * scale + shift + float(t),)


class _FakeSched:
    def scale_model_input(self, x, t):
        return x


def test_single_batch_equals_cfg_at_gs0():
    g = torch.Generator().manual_seed(0)
    lat = torch.randn(1, 4, 8, 8, generator=g)
    enc = torch.randn(2, 7, 5, generator=g)
    added = {"text_embeds": torch.randn(2, 6, generator=g), "time_ids": torch.randn(2, 6, generator=g)}
    unet, sch = _FakeUNet(), _FakeSched()
    a = predict_noise_cfg(unet, sch, lat, 3, enc, added, 0.0)
    b = predict_noise_cfg(unet, sch, lat, 3, enc, added, 0.0, single_batch=True)
    assert torch.equal(a, b)
    # gs != 0: flag ignored, CFG blend unchanged
    c = predict_noise_cfg(unet, sch, lat, 3, enc, added, 2.0)
    d = predict_noise_cfg(unet, sch, lat, 3, enc, added, 2.0, single_batch=True)
    assert torch.equal(c, d) and not torch.equal(a, c)


def test_strat_balanced_and_weights():
    from backsel import balanced_kcenter
    g = torch.randn(32, 5); E = torch.randn(32, 5); E[:20] *= 0.01  # 20 near-duplicates + 12 spread
    idx, G, info = select_backprop_set(g, E, 8, "strat", torch.Generator().manual_seed(0))
    assert idx.numel() == 8 and max(info["cluster_sizes"]) <= 4 and sum(info["cluster_sizes"]) == 32
    for j, i in enumerate(idx.tolist()):
        assert torch.allclose(G[j], g[i] * info["weights"][j])
    # plain kcenter on the same data puts >= half the batch in one cluster
    _, _, kinfo = select_backprop_set(g, E, 8, "kcenter", torch.Generator().manual_seed(0))
    assert max(kinfo["cluster_sizes"]) >= 12


# ---- soft proximity reweighting (--backsel_weighting soft) --------------------

def test_soft_k_ge_n_identity():
    from backsel import select_backprop_set
    g = torch.randn(6, 4); E = torch.randn(6, 5)
    idx, G, info = select_backprop_set(g, E, 6, "uniform", torch.Generator().manual_seed(0),
                                       weighting="soft")
    assert torch.allclose(G, g) and info["rule"] == "identity"


def test_soft_mass_conservation_and_rows():
    from backsel import select_backprop_set
    torch.manual_seed(1)
    n, k = 32, 8
    g = torch.randn(n, 3); E = torch.randn(n, 7)
    idx, G, info = select_backprop_set(g, E, k, "uniform", torch.Generator().manual_seed(3),
                                       weighting="soft")
    assert idx.numel() == k and G.shape == (k, 3)
    # total mass = N (every sample's gradient is handed to exactly one unit of weight)
    assert abs(sum(info["mass"]) - n) < 1e-4
    # G sums to the full-batch gradient sum (softmax rows sum to 1)
    assert torch.allclose(G.sum(0), g.sum(0), atol=1e-4)


def test_soft_tau_limits():
    from backsel import soft_reweight, select_uniform
    torch.manual_seed(2)
    n, k = 16, 4
    g = torch.randn(n, 2); E = torch.randn(n, 3)
    idx, _, _ = select_uniform(g, k, torch.Generator().manual_seed(0))
    # tau -> inf: every non-selected sample split equally over the k representatives
    G_inf, info_inf = soft_reweight(g, E, idx, tau_scale=1e9)
    mask = torch.ones(n, dtype=torch.bool); mask[idx] = False
    expect = g[idx] + g[mask].sum(0, keepdim=True) / k
    assert torch.allclose(G_inf, expect, atol=1e-4)
    # tau -> 0: hard nearest-representative assignment
    G_0, _ = soft_reweight(g, E, idx, tau_scale=1e-9)
    d = torch.cdist(E[mask], E[idx]); nearest = d.argmin(1)
    expect0 = g[idx].clone()
    for j, i in enumerate(nearest.tolist()):
        expect0[i] += g[mask][j]
    assert torch.allclose(G_0, expect0, atol=1e-4)


def test_soft_tau_modes_differ_but_conserve_mass():
    from backsel import soft_reweight, select_uniform
    torch.manual_seed(5)
    n, k = 32, 8
    g = torch.randn(n, 3)
    # clustered embeddings: 4 tight clusters far apart -> local scale << global bandwidth
    centers = torch.randn(4, 16) * 10.0
    E = centers.repeat_interleave(n // 4, dim=0) + 0.1 * torch.randn(n, 16)
    idx, _, _ = select_uniform(g, k, torch.Generator().manual_seed(1))
    Gl, il = soft_reweight(g, E, idx, 1.0, "local")
    Gb, ib = soft_reweight(g, E, idx, 1.0, "bandwidth")
    assert il["tau"] < ib["tau"]                       # local scale is the smaller one
    for G in (Gl, Gb):
        assert torch.allclose(G.sum(0), g.sum(0), atol=1e-4)


# ---- shared-framework adapters: SD adapter == tfg for the same inputs -------------

def test_sd_backsel_adapters_equal_tfg():
    import tfg.backsel as tb
    from backsel import GeneratorTape, select_backprop_set, soft_reweight
    torch.manual_seed(7)
    n, k = 24, 6
    g = torch.randn(n, 5); E = torch.randn(n, 5)
    for rule, fn in (("uniform", lambda t: tb.select_uniform(g, k, t, ())),
                     ("is", lambda t: tb.select_importance(g, k, t, (), floor=0.25)),
                     ("kcenter", lambda t: tb.select_kcenter(E, g, k, t, ())),
                     ("strat", lambda t: tb.select_stratified_balanced(E, g, k, t, ())[:2])):
        idx_sd, G_sd, _ = select_backprop_set(g, E, k, rule, torch.Generator().manual_seed(11))
        idx_t, G_t = fn(GeneratorTape(torch.Generator().manual_seed(11)))
        assert idx_sd.tolist() == list(idx_t), rule
        assert torch.allclose(G_sd, G_t, atol=1e-6), rule
    idx = [1, 4, 9, 15]
    for mode in ("local", "bandwidth"):
        G_sd, info = soft_reweight(g, E, idx, 0.7, mode)
        tau = tb.soft_tau(E, idx, mode=mode, scale=0.7)
        assert abs(info["tau"] - tau) < 1e-9
        assert torch.allclose(G_sd, tb.soft_aggregate(E, g, idx, tau).float(), atol=1e-6)


def test_sd_trust_adapter_equals_tfg():
    from tfg.trust import clip_step, noise_cap
    from trust import apply_trust, trust_cap
    import math
    for abar in (0.1, 0.75, 0.999):
        assert trust_cap(0.3, abar, 16384) == noise_cap(0.3, math.sqrt(1 - abar), numel=16384)
    v = torch.randn(4, 8, 8) * 5
    cap = 3.0
    out, scale, _ = apply_trust(v, cap)
    assert torch.allclose(out, clip_step(v, cap, eps=0.0))
    assert abs(scale - cap / v.norm().item()) < 1e-6


def test_tfg_engine_noise_prev_rms_matches_sd_cap():
    """The engine's step_clip='noise_prev_rms' applies exactly the SD cap."""
    from tfg.config import TFGConfig, TemporalConfig
    from tfg.schedule import DiffusionSchedule
    from tfg.trust import noise_cap
    sch = DiffusionSchedule(T=10)
    cfg = TFGConfig(T=10, temporal=TemporalConfig(step_clip="noise_prev_rms", step_tau=0.25,
                                                  step_min_noise=0.03)).validate()
    from tfg.engine import GeneralizedTFG
    eng = GeneralizedTFG(lambda x, t: torch.zeros_like(x), lambda x: x.sum(), sch,
                         __import__("tfg.noise_tape", fromlist=["NoiseTape"]).NoiseTape(0), cfg)
    D = torch.ones(4, 16, dtype=torch.float64) * 10
    for t in (10, 5, 1):
        got = eng._step_clip(D, t, D, D)
        cap = noise_cap(0.25, float(sch.sqrt_one_minus_ab(t - 1)), numel=64, min_noise=0.03)
        assert abs(got.norm().item() - min(cap, D.norm().item())) < 1e-9
