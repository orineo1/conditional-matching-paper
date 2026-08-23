"""Tests for the deterministic nested target hierarchy and its losses."""

import pytest
import torch

from tfg.target_hierarchy import (
    TargetHierarchy,
    pointwise_loss,
    weighted_mmd2,
)


@pytest.fixture
def S_G():
    torch.manual_seed(0)
    a = torch.randn(120, 1, dtype=torch.float64) * 0.35 + 5.0
    b = torch.randn(130, 1, dtype=torch.float64) * 0.35 - 5.0
    return torch.vstack([a, b])


def test_k1_centre_is_the_target_mean(S_G):
    h = TargetHierarchy(S_G)
    c, w, _ = h.level(1)
    assert c.shape == (1, 1) and w.shape == (1,)
    assert float(w[0]) == pytest.approx(1.0)
    assert torch.allclose(c.reshape(-1), S_G.mean(0))


def test_weights_always_sum_to_one(S_G):
    h = TargetHierarchy(S_G)
    for K in (1, 2, 3, 5, 17, 64, 250):
        _, w, _ = h.level(K)
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-12)
        assert (w > 0).all()


def test_number_of_clusters_equals_K(S_G):
    h = TargetHierarchy(S_G)
    for K in (1, 2, 7, 33, 128, 250):
        c, w, cl = h.level(K)
        assert c.shape[0] == K and w.shape[0] == K and len(cl) == K


def test_partition_is_exact(S_G):
    """Every point belongs to exactly one cluster, at every level."""
    h = TargetHierarchy(S_G)
    for K in (1, 3, 9, 40, 250):
        _, _, cl = h.level(K)
        idx = torch.cat(cl).sort().values
        assert torch.equal(idx, torch.arange(S_G.shape[0]))


def test_hierarchy_is_nested(S_G):
    """The K+1 partition refines the K partition -- the curriculum property."""
    h = TargetHierarchy(S_G)
    for K in (1, 2, 5, 12, 40):
        _, _, coarse = h.level(K)
        _, _, fine = h.level(K + 1)
        coarse_sets = [set(t.tolist()) for t in coarse]
        for f in fine:
            fs = set(f.tolist())
            assert any(fs <= cs for cs in coarse_sets), (
                f"cluster {sorted(fs)[:5]}... at K={K+1} is not contained in any "
                f"cluster of K={K}"
            )


def test_full_resolution_recovers_the_empirical_target(S_G):
    h = TargetHierarchy(S_G)
    c, w, _ = h.level(S_G.shape[0])
    assert torch.allclose(w, torch.full_like(w, 1.0 / S_G.shape[0]))
    assert torch.allclose(c.reshape(-1).sort().values,
                          S_G.reshape(-1).sort().values, atol=1e-12)


def test_construction_is_deterministic(S_G):
    a = TargetHierarchy(S_G).level(37)
    b = TargetHierarchy(S_G).level(37)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_determinism_is_independent_of_global_rng(S_G):
    torch.manual_seed(1)
    a, aw, _ = TargetHierarchy(S_G).level(23)
    torch.manual_seed(999)
    b, bw, _ = TargetHierarchy(S_G).level(23)
    assert torch.equal(a, b) and torch.equal(aw, bw)


def test_split_targets_the_largest_weighted_variance(S_G):
    """K=2 must separate the two well-separated modes."""
    h = TargetHierarchy(S_G)
    c, w, _ = h.level(2)
    lo, hi = float(c.min()), float(c.max())
    assert lo < -3.0 and hi > 3.0, (c.reshape(-1).tolist())
    assert 0.3 < float(w.min()) < 0.7


def test_out_of_range_K_rejected(S_G):
    h = TargetHierarchy(S_G)
    with pytest.raises(ValueError):
        h.level(0)
    with pytest.raises(ValueError):
        h.level(S_G.shape[0] + 1)


def test_descriptor_is_stable_and_hashed(S_G):
    d1 = TargetHierarchy(S_G).descriptor(levels=(1, 2, 8))
    d2 = TargetHierarchy(S_G).descriptor(levels=(1, 2, 8))
    assert d1 == d2
    assert d1["deterministic"] is True and d1["seed"] is None
    assert set(d1["levels"]) == {"1", "2", "8"}


# -- losses -----------------------------------------------------------------

def test_pointwise_loss_is_zero_at_the_target_mean(S_G):
    h = TargetHierarchy(S_G)
    y = h.mean().reshape(1, 1)
    assert float(pointwise_loss(y, h.mean())) == pytest.approx(0.0, abs=1e-20)


def test_pointwise_loss_is_squared_distance(S_G):
    h = TargetHierarchy(S_G)
    y = (h.mean() + 2.5).reshape(1, 1)
    assert float(pointwise_loss(y, h.mean())) == pytest.approx(6.25, rel=1e-12)


def test_weighted_mmd_is_zero_for_identical_atoms():
    c = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    w = torch.tensor([0.5, 0.5], dtype=torch.float64)
    val = float(weighted_mmd2(c, c, w, bandwidth=2.0))
    assert abs(val) < 1e-12


def test_weighted_mmd_grows_with_separation():
    c = torch.tensor([[0.0]], dtype=torch.float64)
    w = torch.tensor([1.0], dtype=torch.float64)
    near = float(weighted_mmd2(torch.tensor([[0.5]], dtype=torch.float64), c, w, 2.0))
    far = float(weighted_mmd2(torch.tensor([[5.0]], dtype=torch.float64), c, w, 2.0))
    assert far > near >= 0


def test_weighted_mmd_respects_prototype_weights():
    y = torch.tensor([[0.0]], dtype=torch.float64)
    c = torch.tensor([[-3.0], [3.0]], dtype=torch.float64)
    a = float(weighted_mmd2(y, c, torch.tensor([0.99, 0.01], dtype=torch.float64), 2.0))
    b = float(weighted_mmd2(y, c, torch.tensor([0.5, 0.5], dtype=torch.float64), 2.0))
    assert a != pytest.approx(b)


def test_weighted_mmd_matches_repo_mmd_at_full_resolution(S_G):
    """With K = |S_G| and uniform weights this must equal the repository's
    V-statistic MMD at the same fixed bandwidth."""
    from tfg._compat import ensure_ot_stub
    ensure_ot_stub()
    from LossFunctions import MMDLoss, RBF

    h = TargetHierarchy(S_G)
    c, w, _ = h.level(S_G.shape[0])
    torch.manual_seed(0)
    y = torch.randn(40, 1, dtype=torch.float64) * 2.0

    ours = float(weighted_mmd2(y, c, w, bandwidth=3.0))
    theirs = float(MMDLoss(kernel=RBF(bandwidth=3.0, device="cpu"),
                           device="cpu")(y, S_G))
    assert ours == pytest.approx(theirs, rel=1e-9), (ours, theirs)


def test_weighted_mmd_is_differentiable():
    y = torch.tensor([[0.3], [1.1]], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([[-2.0], [2.0]], dtype=torch.float64)
    w = torch.tensor([0.4, 0.6], dtype=torch.float64)
    g, = torch.autograd.grad(weighted_mmd2(y, c, w, 2.0), y)
    assert torch.isfinite(g).all() and g.abs().sum() > 0
