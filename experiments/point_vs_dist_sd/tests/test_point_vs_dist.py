"""CPU tests for the point-vs-distributional experiment (no GPU/diffusers)."""
import json
import os
import sys
import types

import numpy as np
import pytest
import torch

_EXP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _EXP)
try:
    import torchvision  # noqa: F401
except ImportError:  # metrics.py needs the stub (CPU box)
    _tv = types.ModuleType("torchvision"); _tr = types.ModuleType("torchvision.transforms")
    _tf = types.ModuleType("torchvision.transforms.functional")
    _tv.transforms = _tr; _tr.functional = _tf
    sys.modules.update({"torchvision": _tv, "torchvision.transforms": _tr,
                        "torchvision.transforms.functional": _tf})

import common  # noqa: E402
from metrics import compute_mmd, compute_point_loss  # noqa: E402  (SD src on path via common)


# ---------------------------- point loss -----------------------------------
def test_point_loss_value_and_target_precedence():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.randn(5, 2)
    a = torch.tensor([[1.0, 1.0]])
    got = compute_point_loss(x, y, point_target=a)          # y must be ignored
    assert abs(float(got) - 1.0) < 1e-6                      # mean(1, 1)
    # centroid fallback
    got2 = compute_point_loss(x, torch.zeros(3, 2))
    assert abs(float(got2) - 1.0) < 1e-6                     # mean(||x_i||^2)


def test_point_loss_grad_flows_only_through_x():
    x = torch.randn(4, 8, requires_grad=True)
    a = torch.randn(1, 8)
    loss = compute_point_loss(x, None, point_target=a)
    g, = torch.autograd.grad(loss, x)
    assert torch.allclose(g, 2 * (x.detach() - a) / 4, atol=1e-6)


def test_variance_decomposition_mechanism():
    """E||y - a||^2 = Var + ||mean - a||^2 -- the Scenario-B mechanism."""
    g = torch.Generator().manual_seed(0)
    y = torch.randn(20000, 3, generator=g) * 2.0 + 1.0
    a = torch.tensor([[1.0, 1.0, 1.0]])
    lhs = float(compute_point_loss(y, None, point_target=a))
    var = float(y.var(dim=0, unbiased=False).sum())
    bias = float(((y.mean(0) - a) ** 2).sum())
    assert abs(lhs - (var + bias)) < 1e-3
    # collapsing the variance strictly lowers the point loss at fixed mean
    y_collapsed = y.mean(0, keepdim=True).expand_as(y)
    assert compute_point_loss(y_collapsed, None, point_target=a) < lhs


# ---------------------------- projection -----------------------------------
def test_project_linf_ball_and_pixel_range():
    x0 = torch.rand(1, 3, 8, 8)
    delta = torch.randn(1, 3, 8, 8)
    eps = 32 / 255
    d = common.project_linf(delta, x0, eps)
    assert float(d.abs().max()) <= eps + 1e-8
    assert float((x0 + d).min()) >= -1e-8 and float((x0 + d).max()) <= 1 + 1e-8
    inside = torch.full_like(delta, eps / 2) * 0  # zero stays zero
    assert torch.equal(common.project_linf(inside, x0, eps), inside)


# ---------------------------- statistics -----------------------------------
def test_p_male_and_cis():
    t = np.eye(2, 768 // 2 * 2)[:, :768] if False else None
    man = np.zeros(768); man[0] = 1.0
    woman = np.zeros(768); woman[1] = 1.0
    txt = np.stack([man, woman])
    male_emb = np.tile(man, (10, 1))
    pm = common.p_male(male_emb, txt)
    assert (pm > 0.99).all()
    m, lo, hi = common.mean_ci95([0.4, 0.5, 0.6])
    assert lo < 0.5 < hi and abs(m - 0.5) < 1e-9
    p, lo, hi = common.wilson_ci95(50, 100)
    assert abs(p - 0.5) < 1e-9 and 0.4 < lo < 0.5 < hi < 0.6


def test_pc1_variance_ratio_detects_collapse():
    g = np.random.default_rng(0)
    axis = np.zeros(768); axis[0] = 1.0
    target = np.concatenate([axis * 2 + g.normal(0, 0.1, (50, 768)),
                             -axis * 2 + g.normal(0, 0.1, (50, 768))])
    collapsed = g.normal(0, 0.1, (500, 768))                # androgynous midpoint
    split = np.concatenate([axis * 2 + g.normal(0, 0.1, (250, 768)),
                            -axis * 2 + g.normal(0, 0.1, (250, 768))])
    rc = common.pc1_stats(target, collapsed)
    rs = common.pc1_stats(target, split)
    assert rc["var_ratio"] < 0.05                            # P-B1 detectable
    assert 0.5 < rs["var_ratio"] < 1.5                       # P-B2 range
    assert abs(rc["gen_mean"] - rc["target_mean"]) < 0.3     # mean is blind (P-B4)


# ---------------------------- cache format ---------------------------------
def test_cache_round_trip_matches_run_mlgd_f_format(tmp_path):
    imgs = {"Man": [np.zeros((8, 8, 3), np.uint8)] * 2,
            "Woman": [np.full((8, 8, 3), 255, np.uint8)] * 2}
    src = np.ones((8, 8, 3), np.uint8)
    scr = np.full((8, 8, 3), 7, np.uint8)
    groups = [("Man", "p1", 2), ("Woman", "p2", 2)]
    common.write_cache(str(tmp_path), imgs, src, scr, groups)
    # exact filenames run_mlgd_f.build_targets_cached expects
    assert os.path.exists(tmp_path / "targets_cache.npz")
    assert os.path.exists(tmp_path / "target_groups.json")
    data = np.load(tmp_path / "targets_cache.npz")
    assert set(data.files) == {"group__Man", "group__Woman", "source_portrait", "scribble"}
    imgs2, src2, scr2, groups2 = common.read_cache(str(tmp_path))
    assert (scr2 == scr).all() and groups2 == [["Man", "p1", 2], ["Woman", "p2", 2]]
    # and the loader in run_mlgd_f uses the same path helper (needs transformers;
    # verified on the cluster env, skipped on a bare CPU box)
    try:
        from run_mlgd_f import _target_cache_path
    except ModuleNotFoundError as e:
        pytest.skip(f"run_mlgd_f unimportable here ({e.name}); format asserted above")
    assert _target_cache_path(str(tmp_path)) == str(tmp_path / "targets_cache.npz")


# ---------------------------- seeds ----------------------------------------
def test_seed_ranges_disjoint_full_config():
    assert common.seed_ranges_disjoint(seed=42, n_guided_steps=125, n_var=100,
                                       n_pgd=200, n_eval=2000)


def test_centroid_equals_half_mixture():
    g = torch.Generator().manual_seed(1)
    m = torch.randn(50, 768, generator=g) + 1
    w = torch.randn(50, 768, generator=g) - 1
    all_e = torch.cat([m, w])
    assert torch.allclose(all_e.mean(0), 0.5 * m.mean(0) + 0.5 * w.mean(0), atol=1e-6)
