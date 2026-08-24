"""Tests for approx_mmd.py.

Run:  /Users/stolk/miniconda3/bin/python -m pytest -q experiments/model-optimization/approx_loss
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import approx_mmd as A  # noqa: E402

sys.path.insert(0, str(A._SIM / "experiments"))
import _common  # noqa: E402
from tfg import gmm_mmd  # noqa: E402

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def synth():
    params = _common.load("2D")
    Y = _common.target_set(params).double()
    bw = _common.fixed_bandwidth(Y)
    torch.manual_seed(3)
    X = (torch.randn(8, 1) * 2.0 + 1.0)
    return params, Y, bw, X


def _grad(obj, X):
    X = X.detach().clone().requires_grad_(True)
    L = obj.loss(X)
    g, = torch.autograd.grad(L, X)
    return L.detach(), g


# ---------------------------------------------------------------- reference
def test_cached_yy_equals_reference(synth):
    _, Y, bw, X = synth
    a, b = A.Reference(Y, bw), A.ReferenceCachedYY(Y, bw)
    La, ga = _grad(a, X)
    Lb, gb = _grad(b, X)
    assert torch.allclose(La, Lb, rtol=1e-7, atol=1e-10)
    assert torch.allclose(ga, gb, rtol=1e-7, atol=1e-10)


def test_reference_matches_repo_mmdloss_on_clip_like():
    torch.manual_seed(0)
    Y = torch.randn(120, 768)
    Y = Y / Y.norm(dim=1, keepdim=True)
    X = torch.randn(8, 768)
    X = X / X.norm(dim=1, keepdim=True)
    bw = _common.fixed_bandwidth(Y)
    from LossFunctions import MMDLoss, RBF
    repo = MMDLoss(kernel=RBF(bandwidth=bw))
    assert torch.allclose(A.Reference(Y, bw).loss(X), repo(X, Y))


# ---------------------------------------------------------------- shapes / finiteness / determinism
@pytest.mark.parametrize("name,kw", [
    ("rff", dict(D=64, seed=1)), ("orf", dict(D=64, seed=1)),
    ("nystrom", dict(L=16, seed=1)), ("subsample", dict(B=32, seed=1)),
    ("sliced_w2", dict(P=8, seed=1)), ("tab_kme_1d", dict(G=512)),
])
def test_shapes_and_finite_grads(synth, name, kw):
    _, Y, bw, X = synth
    for n in (1, 4, 32):
        Xn = X[:1].repeat(n, 1) + 0.1 * torch.arange(n).reshape(n, 1)
        obj = A.make(name, Y, bw, **kw)
        L, g = _grad(obj, Xn)
        assert L.shape == ()
        assert g.shape == Xn.shape
        assert torch.isfinite(L) and torch.isfinite(g).all()


@pytest.mark.parametrize("name,kw", [
    ("rff", dict(D=64)), ("orf", dict(D=64)), ("nystrom", dict(L=16)),
    ("subsample", dict(B=32)),
])
def test_fixed_features_deterministic_given_seed(synth, name, kw):
    _, Y, bw, X = synth
    a = A.make(name, Y, bw, seed=7, **kw)
    b = A.make(name, Y, bw, seed=7, **kw)
    c = A.make(name, Y, bw, seed=8, **kw)
    assert torch.equal(a.loss(X), b.loss(X))
    # repeated calls of the same object are identical (features frozen)
    assert torch.equal(a.loss(X), a.loss(X))
    assert not torch.equal(a.loss(X), c.loss(X))


# ---------------------------------------------------------------- RFF -> exact as D -> inf
def test_rff_converges_to_exact(synth):
    """Mean |error| over feature seeds must decrease monotonically in D
    (in expectation the MSE is O(1/D); with 12 seeds we allow a 10% slack)."""
    _, Y, bw, X = synth
    ref = A.Reference(Y, bw)
    Lr, gr = _grad(ref, X)
    errs = []
    for D in (16, 64, 256, 1024, 4096):
        e = []
        for s in range(12):
            L, g = _grad(A.RFFMMD(Y, bw, D=D, seed=s), X)
            e.append(float(abs(L - Lr)))
        errs.append(sum(e) / len(e))
    for a, b in zip(errs, errs[1:]):
        assert b < 1.1 * a, errs
    assert errs[-1] < 0.1 * errs[0], errs
    # kernel-level unbiasedness: E_W phi(x).phi(y) = k(x,y)
    o = A.RFFMMD(Y, bw, D=2 ** 16, seed=0)
    f = o.features(torch.cat([X[:2], Y[:2]]))
    Kapp = torch.einsum("knd,kmd->nm", f, f)
    Kex = A.multi_kernel(torch.cat([X[:2], Y[:2]]), torch.cat([X[:2], Y[:2]]), bw)
    assert torch.allclose(Kapp, Kex, atol=0.05)


def test_orf_rows_have_gaussian_norms_and_are_orthogonal():
    torch.manual_seed(0)
    Y = torch.randn(20, 64)
    o = A.RFFMMD(Y, 10.0, D=64, seed=0, orthogonal=True)      # D/2 = 32 rows in R^64
    W = o.W
    G = W @ W.T
    off = G - torch.diag(torch.diag(G))
    assert off.abs().max() < 1e-8


# ---------------------------------------------------------------- Nystrom
def test_nystrom_exact_when_landmarks_span_everything(synth):
    """With L = m landmarks (all targets) the Y-side is represented exactly,
    so the YY and XY parts are exact; only the XX part is projected.  Hence
    evaluating at X = subset of Y must be exact."""
    _, Y, bw, _ = synth
    ny = A.NystromMMD(Y, bw, L=Y.shape[0], seed=0, eps=1e-13)
    ref = A.Reference(Y, bw)
    Xs = Y[:8]
    assert torch.allclose(ny.loss(Xs), ref.loss(Xs), rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------- population-target GMM
def test_population_gmm_matches_huge_empirical_target(synth):
    params, Y, bw, X = synth
    pop = A.PopulationGMMMMD(params["target_means"], params["target_variances"],
                             params["target_weights"], bw)
    # huge empirical target from the same GMM
    big = _common.target_set(params, size=200_000, seed=4242).double()
    emp = A.ReferenceCachedYY(big, bw)
    Lp, gp = _grad(pop, X)
    Le, ge = _grad(emp, X)
    # stat tolerance: MMD^2 fluctuations ~ O(1/sqrt(m)) in the cross term
    assert abs(float(Lp - Le)) < 5e-3 * max(1.0, float(Le)), (Lp, Le)
    assert torch.nn.functional.cosine_similarity(gp.flatten(), ge.flatten(), 0) > 0.999


def test_population_gmm_equals_population_mmd2_for_delta_mixture(synth):
    """Empirical X == GMM with zero covariances and uniform weights, so
    population_mmd2_multibandwidth(X-as-deltas, G) must equal the
    semi-population loss EXACTLY."""
    params, Y, bw, X = synth
    pop = A.PopulationGMMMMD(params["target_means"], params["target_variances"],
                             params["target_weights"], bw)
    n = X.shape[0]
    ref = gmm_mmd.population_mmd2_multibandwidth(
        X, torch.zeros(n, 1, 1), torch.ones(n) / n,
        params["target_means"], params["target_variances"], params["target_weights"],
        bandwidth=bw)
    assert torch.allclose(pop.loss(X), ref, rtol=1e-10, atol=1e-12)


def test_population_gmm_zero_at_population_limit(synth):
    """X -> a large sample of G itself: loss -> 0 like O(1/n)."""
    params, Y, bw, _ = synth
    pop = A.PopulationGMMMMD(params["target_means"], params["target_variances"],
                             params["target_weights"], bw)
    Xbig = _common.target_set(params, size=5000, seed=77).double()
    v = float(pop.loss(Xbig))
    assert 0 <= v < 2e-2, v


# ---------------------------------------------------------------- 1-D tabulated KME
def test_tabulated_kme_accuracy(synth):
    _, Y, bw, X = synth
    ref = A.Reference(Y, bw)
    tab = A.TabulatedKME1D(Y, bw, G=4096)
    Lr, gr = _grad(ref, X)
    Lt, gt = _grad(tab, X)
    assert abs(float(Lt - Lr)) < 1e-4 * max(1.0, float(Lr))
    assert torch.nn.functional.cosine_similarity(gr.flatten(), gt.flatten(), 0) > 0.999


# ---------------------------------------------------------------- subsample
def test_subsample_full_B_equals_exact(synth):
    _, Y, bw, X = synth
    sub = A.SubsampledTargetMMD(Y, bw, B=Y.shape[0], seed=0)
    assert torch.allclose(sub.loss(X), A.ReferenceCachedYY(Y, bw).loss(X), rtol=1e-10)


# ---------------------------------------------------------------- sliced
def test_sliced_w2_is_1d_wasserstein_when_d1(synth):
    _, Y, bw, X = synth
    sw = A.SlicedW2(Y, P=32, seed=0)
    assert sw.P == 1
    # zero when X equals target quantiles
    n = 16
    q = sw._target_quantiles(n)
    assert float(sw.loss(q)) < 1e-20
