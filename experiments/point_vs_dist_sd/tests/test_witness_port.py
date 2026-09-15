"""
Equivalence of OUR ported witness back-selection with upstream/main's.

The reference implementations below are copied VERBATIM from
upstream/main:SD_cond_SD_controlnet/src/{generation.py,metrics.py} (retrieved
2026-09-13 via `git show upstream/main:...`). If these tests fail, our branch is
running a DIFFERENT method from Ori's and the ablation is not comparing what it
claims to compare.
"""
import os
import sys
import types

import pytest
import torch

_SD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                   "SD_cond_SD_controlnet", "src"))
sys.path.insert(0, _SD)
try:
    import torchvision  # noqa: F401
except ImportError:                       # metrics.py imports it for the eval helper only
    _tv = types.ModuleType("torchvision"); _tr = types.ModuleType("torchvision.transforms")
    _tf = types.ModuleType("torchvision.transforms.functional")
    _tv.transforms = _tr; _tr.functional = _tf
    sys.modules.update({"torchvision": _tv, "torchvision.transforms": _tr,
                        "torchvision.transforms.functional": _tf})

from generation import (_expand_rows_by_count, _witness_select_indices)   # ours
from metrics import compute_witness_scores                                # ours


# ---------------------------------------------------------------------------
# REFERENCE — verbatim from upstream/main
# ---------------------------------------------------------------------------
def ref_rbf_kernel(a, b, bw, alpha):
    a_sq = (a ** 2).sum(dim=1, keepdim=True)
    b_sq = (b ** 2).sum(dim=1, keepdim=True)
    dist_sq = a_sq + b_sq.T - 2 * torch.mm(a, b.T)
    return torch.exp(-(dist_sq / (2 * bw ** 2)) ** alpha)


def ref_estimate_bandwidth(x, y, bandwidth_scale=1.0):
    dev = x.device
    ss = min(1000, x.shape[0], y.shape[0])
    with torch.no_grad():
        x_sq = (x[:ss].detach() ** 2).sum(dim=1, keepdim=True)
        y_sq = (y[:ss].detach() ** 2).sum(dim=1, keepdim=True)
        dists = x_sq + y_sq.T - 2 * torch.mm(x[:ss].detach(), y[:ss].detach().T)
        dists = dists[dists > 0]
        bandwidth = (torch.sqrt(torch.median(dists) / 2) if len(dists) > 0
                     else torch.tensor(1.0, device=dev))
    return bandwidth.detach() * bandwidth_scale


def ref_compute_witness_scores(x, y, bandwidth=None, bandwidth_scale=1.0, kernel_alpha=1.0):
    with torch.no_grad():
        x = x.float().detach()
        y = y.float().detach()
        if bandwidth is None:
            bandwidth = ref_estimate_bandwidth(x, y, bandwidth_scale)
        K_xx = ref_rbf_kernel(x, x, bandwidth, kernel_alpha)
        K_xy = ref_rbf_kernel(x, y, bandwidth, kernel_alpha)
        scores = K_xx.mean(dim=1) - K_xy.mean(dim=1)
    return scores, bandwidth


def ref_witness_select_indices(scores, k, witness_floor, witness_temperature,
                               replacement, generator):
    n = scores.shape[0]
    probs = scores.abs().double()
    if witness_temperature != 1.0:
        probs = probs.clamp_min(1e-12) ** (1.0 / witness_temperature)
    probs = (1.0 - witness_floor) * probs / probs.sum().clamp_min(1e-12) + witness_floor / n
    probs = probs / probs.sum()
    idx_t = torch.multinomial(probs.cpu(), k, replacement=replacement, generator=generator)
    unique_idx, counts = torch.unique(idx_t, return_counts=True)
    grad_counts = dict(zip(unique_idx.tolist(), counts.tolist()))
    return grad_counts, probs


def ref_expand_rows_by_count(rows, counts):
    out = []
    for i, row in enumerate(rows):
        c = counts.get(i, 0)
        out.extend([row] * c if c > 0 else [row.detach()])
    return out


# ---------------------------------------------------------------------------
# Equivalence
# ---------------------------------------------------------------------------
def _xy(n=40, m=25, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randn(m, d, generator=g)


@pytest.mark.parametrize("bw_scale,alpha", [(1.0, 1.0), (0.5, 1.0), (1.0, 2.0), (2.0, 0.5)])
def test_scores_match_main(bw_scale, alpha):
    x, y = _xy()
    ours, bw_ours = compute_witness_scores(x, y, bandwidth_scale=bw_scale, kernel_alpha=alpha)
    ref, bw_ref = ref_compute_witness_scores(x, y, bandwidth_scale=bw_scale, kernel_alpha=alpha)
    # equal_nan: at alpha < 1 the kernel takes (dist_sq)^alpha of a diagonal that is
    # ~-1e-6 from float round-off, so BOTH implementations produce NaN in the same
    # places. That is a property of the shared formula, not a porting difference —
    # the point of this assert is that the two agree element-for-element including
    # where they are NaN. (The configs we run use alpha = 1, which has no NaNs.)
    assert torch.equal(torch.isnan(ours), torch.isnan(ref))
    m = ~torch.isnan(ours)
    assert torch.equal(ours[m], ref[m])
    assert float(bw_ours) == float(bw_ref)
    if alpha >= 1.0:
        assert not torch.isnan(ours).any()


@pytest.mark.parametrize("k", [1, 5, 20, 40])
@pytest.mark.parametrize("floor", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("temp", [0.3, 1.0, 2.0])
@pytest.mark.parametrize("repl", [False, True])
def test_selection_matches_main_bitwise(k, floor, temp, repl):
    """Same scores + same generator seed => identical draws and probabilities."""
    x, y = _xy()
    scores, _ = compute_witness_scores(x, y)
    ours, p_ours = _witness_select_indices(
        scores, k, floor, temp, repl, torch.Generator().manual_seed(1234))
    ref, p_ref = ref_witness_select_indices(
        scores, k, floor, temp, repl, torch.Generator().manual_seed(1234))
    assert ours == ref, (ours, ref)
    assert torch.equal(p_ours, p_ref)
    assert sum(ours.values()) == k                      # k draws, always
    if not repl:
        assert all(c == 1 for c in ours.values())       # distinct indices


def test_ori_config_selection_matches_main():
    """Ori's actual gender config: k=50 of 100, floor 0.0, temperature 0.3."""
    x, y = _xy(n=100, m=20, d=768, seed=7)
    scores, _ = compute_witness_scores(x, y)
    ours, p_ours = _witness_select_indices(
        scores, 50, 0.0, 0.3, False, torch.Generator().manual_seed(42))
    ref, p_ref = ref_witness_select_indices(
        scores, 50, 0.0, 0.3, False, torch.Generator().manual_seed(42))
    assert ours == ref and torch.equal(p_ours, p_ref)
    assert len(ours) == 50


def test_probability_semantics():
    scores = torch.tensor([10.0, -8.0, 0.0, 1.0])
    _, p_uniform = _witness_select_indices(scores, 2, 1.0, 1.0, False,
                                           torch.Generator().manual_seed(0))
    assert torch.allclose(p_uniform, torch.full((4,), 0.25, dtype=torch.float64))
    _, p_pure = _witness_select_indices(scores, 2, 0.0, 1.0, False,
                                        torch.Generator().manual_seed(0))
    assert torch.allclose(p_pure, scores.abs().double() / scores.abs().sum())
    assert float(p_pure[2]) == 0.0                       # |w| = 0 is never drawn at floor 0
    _, p_sharp = _witness_select_indices(scores, 2, 0.0, 0.3, False,
                                         torch.Generator().manual_seed(0))
    assert p_sharp[0] > p_pure[0]                        # T<1 sharpens toward the top score


def test_expand_rows_matches_main_and_detaches():
    rows = [torch.randn(1, 4, requires_grad=True) for _ in range(5)]
    counts = {1: 1, 3: 2}
    ours = _expand_rows_by_count(rows, counts)
    ref = ref_expand_rows_by_count(rows, counts)
    assert len(ours) == len(ref) == 6                    # 1 + 2 selected + 3 detached
    for a, b in zip(ours, ref):
        assert torch.equal(a, b) and a.requires_grad == b.requires_grad
    assert [r.requires_grad for r in ours] == [False, True, False, True, True, False]


def test_gradient_flows_only_through_selected_rows():
    """The property the method relies on: detached rows contribute value, not gradient."""
    base = torch.randn(1, 6, requires_grad=True)
    rows = [base * (i + 1.0) for i in range(4)]
    counts = {2: 1}
    batch = torch.cat(_expand_rows_by_count(rows, counts), dim=0)
    loss = (batch ** 2).sum()
    g, = torch.autograd.grad(loss, base, retain_graph=True)
    expected, = torch.autograd.grad((rows[2] ** 2).sum(), base)
    assert torch.allclose(g, expected)
