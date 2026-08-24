"""Equivalence tests: fast_mmd variants vs simulations/src/LossFunctions.py.

Run:  cd simulations && python -m pytest ../experiments/model-optimization/exact_loss -q
"""
import itertools
import math
import sys
import types
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT / "simulations" / "src", HERE, ROOT / "SD_cond_SD_controlnet" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# LossFunctions imports POT (`ot`) at module scope; SD metrics imports torchvision.
# Neither is needed for the MMD code paths -- stub them like simulations/tests/conftest.py.
for name in ("ot", "torchvision", "torchvision.transforms", "torchvision.transforms.functional"):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)

from LossFunctions import MMDLoss, RBF                       # noqa: E402
import fast_mmd as fm                                        # noqa: E402
from fast_mmd import MMDFixedTarget                          # noqa: E402

try:
    from metrics import compute_mmd as sd_compute_mmd       # SD reference
except Exception:                                            # pragma: no cover
    sd_compute_mmd = None

RTOL, ATOL = 1e-12, 1e-12


@pytest.fixture(autouse=True)
def _float64_default():
    """The reference builds its bandwidth multipliers in the DEFAULT dtype; with
    float32 default it silently rounds the bandwidth to float32 (see
    fast_mmd.repo_scales).  Equivalence at 1e-12 therefore needs float64 default,
    as simulations/tests/conftest.py sets."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _data(n, m, d, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    X = (scale * torch.randn(n, d, generator=g, dtype=torch.float64)).requires_grad_(True)
    Y = scale * torch.randn(m, d, generator=g, dtype=torch.float64) + 0.3
    return X, Y


def _ref(X, Y, bw):
    loss = MMDLoss(kernel=RBF(bandwidth=bw, device="cpu"), device="cpu")(X, Y)
    (g,) = torch.autograd.grad(loss, X)
    return loss.detach(), g


def _naive_mmd(X, Y, scales, alpha=1.0, unbiased=False):
    """Double-loop definition (no cdist, no broadcasting tricks)."""
    def k(a, b):
        d2 = ((a - b) ** 2).sum()
        return sum(torch.exp(-((d2 / s) ** alpha)) for s in scales)
    n, m = X.shape[0], Y.shape[0]
    xx = sum(k(X[i], X[j]) for i in range(n) for j in range(n) if (not unbiased or i != j))
    yy = sum(k(Y[i], Y[j]) for i in range(m) for j in range(m) if (not unbiased or i != j))
    xy = sum(k(X[i], Y[j]) for i in range(n) for j in range(m))
    if unbiased:
        xx = xx / (n * (n - 1)) if n > 1 else 0.0
        yy = yy / (m * (m - 1)) if m > 1 else 0.0
    else:
        xx, yy = xx / n ** 2, yy / m ** 2
    return xx - 2 * xy / (n * m) + yy


VARIANTS = {
    "cdist": dict(dist="cdist"),
    "mm": dict(dist="mm"),
    "powchain": dict(dist="mm", kernel_eval="powchain"),
    "loop": dict(dist="mm", kernel_eval="loop"),
    "chunked": dict(dist="mm", chunk=64),
    "chunked_cdist": dict(dist="cdist", chunk=100),
    "yy_autograd": dict(dist="mm", reattach_yy="autograd"),
}


# --------------------------------------------------------------------------- #
# 1. main equivalence grid: loss and dL/dX vs reference, fixed + adaptive bandwidth
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("bw_mode", ["fixed", "adaptive"])
@pytest.mark.parametrize("n,m,d", [
    (1, 5, 1), (2, 5, 2), (8, 5, 8), (32, 5, 2),
    (1, 250, 2), (2, 250, 1), (8, 250, 8), (32, 250, 2), (8, 250, 768), (32, 250, 768),
    (100, 250, 2),
])
def test_matches_reference(variant, bw_mode, n, m, d):
    X, Y = _data(n, m, d)
    bw = None if bw_mode == "adaptive" else 0.7 + 0.1 * d ** 0.5
    ref_loss, ref_grad = _ref(X, Y, bw)
    f = MMDFixedTarget(Y, bw, **VARIANTS[variant])
    loss = f(X)
    (g,) = torch.autograd.grad(loss, X)
    assert torch.allclose(loss, ref_loss, rtol=RTOL, atol=ATOL), (loss.item(), ref_loss.item())
    assert torch.allclose(g, ref_grad, rtol=RTOL, atol=ATOL), (g - ref_grad).abs().max().item()
    # second call must give the same answer (cache reuse)
    loss2 = f(X)
    assert torch.equal(loss2.detach(), loss.detach())


def test_reference_matches_naive_definition():
    X, Y = _data(4, 6, 3)
    bw = 1.1
    ref_loss, _ = _ref(X, Y, bw)
    scales = [bw * 2.0 ** (k - 2) for k in range(5)]
    naive = _naive_mmd(X, Y, scales)
    assert torch.allclose(ref_loss, naive.detach(), rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# 2. batched: B sets at once == B separate reference calls
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bw_mode", ["fixed", "adaptive"])
@pytest.mark.parametrize("dist", ["cdist", "mm"])
@pytest.mark.parametrize("B,n,m,d", [(3, 8, 250, 2), (4, 1, 30, 5), (2, 16, 250, 768)])
def test_batched(bw_mode, dist, B, n, m, d):
    g = torch.Generator().manual_seed(1)
    Xb = torch.randn(B, n, d, generator=g, dtype=torch.float64).requires_grad_(True)
    Y = torch.randn(m, d, generator=g, dtype=torch.float64)
    bw = None if bw_mode == "adaptive" else 1.3
    f = MMDFixedTarget(Y, bw, dist=dist)
    Lb = f.batched(Xb)
    (gb,) = torch.autograd.grad(Lb.sum(), Xb)
    ref = MMDLoss(kernel=RBF(bandwidth=bw))
    Ls = torch.stack([ref(x, Y) for x in Xb])
    (gs,) = torch.autograd.grad(Ls.sum(), Xb)
    assert torch.allclose(Lb, Ls, rtol=RTOL, atol=ATOL)
    assert torch.allclose(gb, gs, rtol=RTOL, atol=ATOL)
    # and each element equals the single-set path of the same object
    for b in range(B):
        assert torch.allclose(Lb[b], f(Xb[b]).detach(), rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# 3. mmd_reference_like (stacked ablation) and kernel_eval equivalences
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bw", [None, 0.9])
@pytest.mark.parametrize("dist,ke", [("mm", "exp"), ("cdist", "powchain"), ("mm", "loop")])
def test_reference_like(bw, dist, ke):
    X, Y = _data(8, 40, 3)
    ref_loss, ref_grad = _ref(X, Y, bw)
    L = fm.mmd_reference_like(X, Y, bw, dist=dist, kernel_eval=ke)
    (g,) = torch.autograd.grad(L, X)
    assert torch.allclose(L, ref_loss, rtol=RTOL, atol=ATOL)
    assert torch.allclose(g, ref_grad, rtol=RTOL, atol=ATOL)


def test_powchain_elementwise():
    D = torch.linspace(0, 50, 1001, dtype=torch.float64)
    scales = fm.repo_scales(0.8)
    a = fm.kernel_from_d2(D, scales, 1.0, "exp")
    b = fm.kernel_from_d2(D, scales, 1.0, "powchain")
    assert torch.allclose(a, b, rtol=1e-14, atol=1e-300)


# --------------------------------------------------------------------------- #
# 4. generalised kernel (alpha), unbiased U-statistic, SD conventions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [1.0, 2.0])
@pytest.mark.parametrize("unbiased", [False, True])
@pytest.mark.parametrize("n,m,d", [(1, 7, 3), (3, 7, 3), (8, 20, 16)])
@pytest.mark.parametrize("chunk", [None, 5])
def test_alpha_unbiased_vs_naive(alpha, unbiased, n, m, d, chunk):
    X, Y = _data(n, m, d, seed=3)
    bw = 1.7
    f = MMDFixedTarget(Y, bw, kernel="sd", alpha=alpha, unbiased=unbiased, dist="mm", chunk=chunk)
    L = f(X)
    (g,) = torch.autograd.grad(L, X)
    naive = _naive_mmd(X, Y, [2 * bw ** 2], alpha=alpha, unbiased=unbiased)
    (gn,) = torch.autograd.grad(naive, X)
    assert torch.allclose(L, naive.detach(), rtol=RTOL, atol=ATOL)
    assert torch.allclose(g, gn, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(sd_compute_mmd is None, reason="SD metrics not importable")
@pytest.mark.parametrize("alpha", [1.0, 2.0])
@pytest.mark.parametrize("n,m", [(1, 120), (8, 120), (32, 100)])
def test_sd_compute_mmd(alpha, n, m):
    """SD pipeline: float32, detached median bandwidth, U-statistic, sqrt(|.|+1e-8)."""
    g = torch.Generator().manual_seed(5)
    X = torch.nn.functional.normalize(torch.randn(n, 768, generator=g), dim=1).float().requires_grad_(True)
    Y = torch.nn.functional.normalize(torch.randn(m, 768, generator=g), dim=1).float()
    ref = sd_compute_mmd(X, Y, bandwidth_scale=1.0, kernel_alpha=alpha)
    (gr,) = torch.autograd.grad(ref, X)
    bw = fm.sd_median_bandwidth(X, Y)
    f = MMDFixedTarget(Y, bw, kernel="sd", alpha=alpha, unbiased=True, sd_output=True, dist="mm")
    L = f(X)
    (gl,) = torch.autograd.grad(L, X)
    assert torch.allclose(L, ref.detach(), rtol=1e-5, atol=1e-6)
    assert torch.allclose(gl, gr, rtol=1e-4, atol=1e-6), (gl - gr).abs().max().item()


# --------------------------------------------------------------------------- #
# 5. finite-difference gradient checks (incl. the custom chunked Function)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bw", [None, 0.8])
@pytest.mark.parametrize("kw", [dict(dist="cdist"), dict(dist="mm", chunk=3),
                                dict(dist="mm", kernel_eval="powchain"),
                                dict(kernel="sd", alpha=2.0, unbiased=True, dist="mm", chunk=4)])
def test_gradcheck(bw, kw):
    if kw.get("kernel") == "sd" and bw is None:
        bw = 1.0     # adaptive repo rule is for the repo kernel
    X, Y = _data(4, 9, 2, seed=7)
    f = MMDFixedTarget(Y, bw, **kw)
    assert torch.autograd.gradcheck(lambda x: f(x), (X,), eps=1e-6, atol=1e-7, rtol=1e-6)


def test_chunked_scale_gradient_fd():
    """d loss / d scales through the fused chunked kernel sum vs finite differences."""
    X, Y = _data(3, 8, 2, seed=11)
    s = torch.tensor([0.4, 0.8, 1.6], dtype=torch.float64, requires_grad=True)
    fn = lambda x, sc: fm.chunked_kernel_sum(x, Y, sc, alpha=1.0, chunk=3)
    assert torch.autograd.gradcheck(fn, (X, s), eps=1e-6, atol=1e-7)
    fn2 = lambda x, sc: fm.chunked_kernel_sum(x, Y, sc, alpha=2.0, chunk=5)
    assert torch.autograd.gradcheck(fn2, (X, s), eps=1e-6, atol=1e-7)


# --------------------------------------------------------------------------- #
# 6. edge cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_identical_rows_and_n1(variant):
    # X rows identical (and one equal to a target row): zero distances on and off diagonal
    g = torch.Generator().manual_seed(2)
    Y = torch.randn(12, 3, generator=g, dtype=torch.float64)
    X = Y[:1].repeat(4, 1).clone().requires_grad_(True)
    for bw in (0.5, None):
        ref_loss, ref_grad = _ref(X, Y, bw)
        f = MMDFixedTarget(Y, bw, **VARIANTS[variant])
        L = f(X)
        (gr,) = torch.autograd.grad(L, X)
        assert torch.isfinite(L) and torch.isfinite(gr).all()
        assert torch.allclose(L, ref_loss, rtol=RTOL, atol=ATOL)
        assert torch.allclose(gr, ref_grad, rtol=RTOL, atol=ATOL)
    X1 = Y[3:4].clone().requires_grad_(True)            # n = 1
    ref_loss, ref_grad = _ref(X1, Y, 0.5)
    L = MMDFixedTarget(Y, 0.5, **VARIANTS[variant])(X1)
    (gr,) = torch.autograd.grad(L, X1)
    assert torch.allclose(L, ref_loss, rtol=RTOL, atol=ATOL)
    assert torch.allclose(gr, ref_grad, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_near_zero_bandwidth(variant):
    """bw = 1e-12.  Mathematically every off-diagonal kernel value underflows to
    exactly 0 and the diagonal is exp(0) = 1 per kernel, so the loss is
    K/n + K/m (K = 5) with zero gradient -- no NaN/inf in the reference.

    BUT the reference's diagonal is NOT exactly zero: for a stacked matrix with
    > 25 rows ``torch.cdist`` switches to the matmul formula and the diagonal
    carries rounding noise ~1e-15, which divided by bw*0.25 = 2.5e-13 is O(1e-2)
    inside the exp.  So in this regime the reference value itself is rounding-
    noise dependent (here 0.95831 instead of 0.95833) and the variants (exact
    zero diagonal for n <= 25 via cdist, different noise via mm) can only agree
    with it to that noise level.  The regime is irrelevant in practice (the repo
    bandwidth is the mean squared distance, O(1)); for bw >~ 1e-3 * ||x||^2 the
    1e-12 equivalence of the main grid holds.  bw = 0 exactly gives 0/0 = NaN on
    the diagonal in the reference and is not reproduced."""
    X, Y = _data(6, 40, 2, seed=4)
    bw = 1e-12
    ref_loss, ref_grad = _ref(X, Y, bw)
    ideal = torch.tensor(5.0 / 6 + 5.0 / 40, dtype=torch.float64)
    assert torch.isfinite(ref_loss)
    assert (ref_loss - ideal).abs() < 1e-3
    assert torch.equal(ref_grad, torch.zeros_like(ref_grad))
    f = MMDFixedTarget(Y, bw, **VARIANTS[variant])
    L = f(X)
    (g,) = torch.autograd.grad(L, X)
    assert torch.isfinite(L)
    assert (L - ideal).abs() < 1e-3
    assert (L - ref_loss).abs() < 1e-3
    assert torch.equal(g, torch.zeros_like(g))


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_small_bandwidth_still_exact(variant):
    """bw = 1e-2 (kernels mostly underflowing, sharp): equivalence still 1e-12."""
    X, Y = _data(6, 40, 2, seed=4)
    for bw in (1e-2, 5e-2):
        ref_loss, ref_grad = _ref(X, Y, bw)
        L = MMDFixedTarget(Y, bw, **VARIANTS[variant])(X)
        (g,) = torch.autograd.grad(L, X)
        assert torch.allclose(L, ref_loss, rtol=RTOL, atol=ATOL)
        assert torch.allclose(g, ref_grad, rtol=RTOL, atol=ATOL)


def test_large_bandwidth_and_scale():
    """Large bw / large coordinates: kernels near 1; checks the closed-form D_xy sum
    in the chunked adaptive path and the cancellation-prone mm distances."""
    X, Y = _data(8, 100, 4, seed=9, scale=30.0)
    for bw in (None, 5e3):
        ref_loss, ref_grad = _ref(X, Y, bw)
        for kw in (dict(dist="mm"), dict(dist="mm", chunk=16), dict(dist="cdist", chunk=16)):
            L = MMDFixedTarget(Y, bw, **kw)(X)
            (g,) = torch.autograd.grad(L, X)
            assert torch.allclose(L, ref_loss, rtol=1e-10, atol=1e-12)
            assert torch.allclose(g, ref_grad, rtol=1e-10, atol=1e-12)


def test_default_float32_rounds_bandwidth_in_reference():
    """Documented quirk: under default float32 the reference rounds bw*mult to float32
    even for float64 inputs, and fast_mmd reproduces it (so they still agree)."""
    torch.set_default_dtype(torch.float32)
    X, Y = _data(8, 50, 2)
    ref_loss, ref_grad = _ref(X, Y, 1.3)
    L = MMDFixedTarget(Y, 1.3, dist="mm")(X)
    (g,) = torch.autograd.grad(L, X)
    assert torch.allclose(L, ref_loss, rtol=RTOL, atol=ATOL)
    assert torch.allclose(g, ref_grad, rtol=RTOL, atol=ATOL)
    torch.set_default_dtype(torch.float64)
    L64, _ = _ref(X, Y, 1.3)
    assert (L64 - ref_loss).abs() > 1e-10        # the rounding is visible
