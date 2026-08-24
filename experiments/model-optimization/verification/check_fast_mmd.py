"""Agent 6 -- independent float64 check of exact_loss/fast_mmd.py against
simulations/src/LossFunctions.MMDLoss (value AND gradient w.r.t. X).

    cd simulations && /Users/stolk/miniconda3/bin/python ../experiments/model-optimization/verification/check_fast_mmd.py

Covers: unequal n vs m, multi-bandwidth (5 kernels, mul_factor 2), fixed and
adaptive (stacked, gradient through the bandwidth) bandwidth, dist cdist/mm,
kernel_eval exp/powchain/loop, chunked XY, batched(); plus the float32
default-dtype behaviour that the production loop actually runs under.
Written by the verifier; does not import anything from the Agent-2 tests.
"""
import itertools
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT / "simulations" / "src", HERE.parent / "exact_loss"):
    sys.path.insert(0, str(p))
if "ot" not in sys.modules:
    try:
        import ot  # noqa: F401
    except ImportError:
        sys.modules["ot"] = types.ModuleType("ot")

import torch  # noqa: E402
from LossFunctions import MMDLoss, RBF  # noqa: E402
from fast_mmd import MMDFixedTarget  # noqa: E402


def ref(X, Y, bw):
    return MMDLoss(kernel=RBF(bandwidth=bw))(X, Y)


def one(n, m, d, bw_mode, variant, seed, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    X0 = torch.randn(n, d, generator=g, dtype=dtype) * 1.7
    Y = torch.randn(m, d, generator=g, dtype=dtype) + 0.5
    bw = None if bw_mode == "adaptive" else float(torch.cdist(Y, Y) .pow(2).sum() / (m * m - m))
    X = X0.clone().requires_grad_(True)
    r = ref(X, Y, bw)
    gr, = torch.autograd.grad(r, X)
    f = MMDFixedTarget(Y, bw, **variant)
    X2 = X0.clone().requires_grad_(True)
    v = f(X2)
    gv, = torch.autograd.grad(v, X2)
    # batched: 3 copies incl. a perturbed one
    Xb = torch.stack([X0, X0 + 0.1, X0 * 0.9]).requires_grad_(True)
    vb = f.batched(Xb)
    gb, = torch.autograd.grad(vb.sum(), Xb)
    X3 = (X0 * 0.9).clone().requires_grad_(True)
    r3 = ref(X3, Y, bw)
    gr3, = torch.autograd.grad(r3, X3)
    return {
        "val": float((r - v).abs() / r.abs().clamp_min(1e-30)),
        "grad": float((gr - gv).abs().max() / gr.abs().max().clamp_min(1e-30)),
        "bat_val": float((vb[0] - r).abs() / r.abs().clamp_min(1e-30)),
        "bat_val3": float((vb[2] - r3).abs() / r3.abs().clamp_min(1e-30)),
        "bat_grad": float((gb[0] - gr).abs().max() / gr.abs().max().clamp_min(1e-30)),
        "bat_grad3": float((gb[2] - gr3).abs().max() / gr3.abs().max().clamp_min(1e-30)),
    }


def main():
    torch.set_default_dtype(torch.float64)   # the reference builds multipliers in the default dtype
    variants = {
        "cdist_exp": dict(dist="cdist", kernel_eval="exp"),
        "mm_exp": dict(dist="mm", kernel_eval="exp"),
        "mm_powchain": dict(dist="mm", kernel_eval="powchain"),
        "mm_loop": dict(dist="mm", kernel_eval="loop"),
        "mm_exp_chunk64": dict(dist="mm", kernel_eval="exp", chunk=64),
        "mm_powchain_chunk7": dict(dist="mm", kernel_eval="powchain", chunk=7),
        "cdist_autograd_yy": dict(dist="cdist", kernel_eval="exp", reattach_yy="autograd"),
    }
    worst = {}
    rows = []
    for (n, m, d), bw_mode, (vn, vk), seed in itertools.product(
            [(1, 250, 2), (4, 250, 2), (8, 250, 5), (32, 250, 10), (7, 13, 3), (250, 8, 2), (64, 64, 4)],
            ["fixed", "adaptive"], variants.items(), [0, 1]):
        r = one(n, m, d, bw_mode, vk, seed)
        key = (vn, bw_mode)
        for k, val in r.items():
            worst[(key, k)] = max(worst.get((key, k), 0.0), val)
        rows.append(((n, m, d), bw_mode, vn, seed, r))
    print("float64, relative errors, worst over (n,m,d) in "
          "{(1,250,2),(4,250,2),(8,250,5),(32,250,10),(7,13,3),(250,8,2),(64,64,4)} x 2 seeds")
    print(f"{'variant':<22}{'bandwidth':<10}{'val':>10}{'grad':>10}{'bat_val':>10}{'bat_grad':>10}{'bat_val3':>10}{'bat_grad3':>10}")
    ok = True
    for vn in variants:
        for bw_mode in ["fixed", "adaptive"]:
            key = (vn, bw_mode)
            vals = [worst[(key, k)] for k in ("val", "grad", "bat_val", "bat_grad", "bat_val3", "bat_grad3")]
            print(f"{vn:<22}{bw_mode:<10}" + "".join(f"{v:>10.1e}" for v in vals))
            if max(vals) > 1e-10:
                ok = False
    print("VERDICT float64:", "EXACT to <=1e-10 relative (value and dX-gradient)" if ok else "MISMATCH > 1e-10")

    # float32 behaviour under the production default dtype (float32): the
    # reference rounds the bandwidth to float32 (mult dtype), fast_mmd mirrors it.
    torch.set_default_dtype(torch.float32)
    g = torch.Generator().manual_seed(3)
    X0 = torch.randn(8, 2, generator=g) * 1.7
    Y = torch.randn(250, 2, generator=g) + 0.5
    bw = float(torch.cdist(Y, Y).pow(2).sum() / (250 * 249))
    X = X0.clone().requires_grad_(True)
    r = ref(X, Y, bw); gr, = torch.autograd.grad(r, X)
    out = []
    for vn, vk in variants.items():
        if "autograd" in vn:
            continue
        f = MMDFixedTarget(Y, bw, **vk)
        X2 = X0.clone().requires_grad_(True)
        v = f(X2); gv, = torch.autograd.grad(v, X2)
        out.append((vn, float((r - v).abs()), float((gr - gv).abs().max()), float(gr.abs().max())))
    print("\nfloat32 default dtype, fixed bandwidth, n=8 m=250 d=2 (abs diffs; |grad|max shown):")
    for vn, dv, dg, gm in out:
        print(f"  {vn:<22} |dval|={dv:.2e}  |dgrad|max={dg:.2e}  (|grad|max={gm:.2e})")
    print("  -> float32: agreement at round-off level only (REORDER), not bit-identical;"
          " see end_to_end_results.csv for the chaotic amplification over 99 steps.")


if __name__ == "__main__":
    main()
