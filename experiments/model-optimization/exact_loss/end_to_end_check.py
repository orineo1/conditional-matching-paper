"""End-to-end: _guided.run with the reference MMD vs fast_mmd, via monkeypatching
``_guided.MMDLoss`` (no edit to simulations/).  Reports wall time per run and the
max |x_hat difference| (the loop runs the repo's float32 models, so agreement is at
float32 level, not 1e-12).

    cd simulations && python ../experiments/model-optimization/exact_loss/end_to_end_check.py
"""
import statistics
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIM = ROOT / "simulations"
for p in (SIM / "src", SIM / "experiments", HERE):
    sys.path.insert(0, str(p))
if "ot" not in sys.modules:
    try:
        import ot  # noqa: F401
    except ImportError:
        sys.modules["ot"] = types.ModuleType("ot")

import torch  # noqa: E402
import _guided  # noqa: E402
from _common import fixed_bandwidth, load, target_set  # noqa: E402
from _models import SEEDS, conditional_model, unconditional_model  # noqa: E402
import fast_mmd as fm  # noqa: E402


class StackedMMShim:
    """CONTROL: the reference algorithm itself (stacked (X;Y), YY recomputed) with only
    the distance formula changed to norms+matmul.  Any x_hat difference it shows is
    pure float32 rounding sensitivity of the 99-step guided loop, not a property of
    the cached-target variants."""

    def __init__(self, kernel=None, device="cpu"):
        self.bw = kernel.bandwidth

    def __call__(self, X, Y):
        return fm.mmd_reference_like(X, Y, self.bw, dist="mm")


class FastMMDLossShim:
    """Drop-in for LossFunctions.MMDLoss(kernel=RBF(bandwidth=bw)) with a fixed Y."""

    def __init__(self, kernel=None, device="cpu", **kw):
        self.bw = kernel.bandwidth if kernel is not None else None
        self.kw = kw
        self._f = None
        self._Y = None

    def __call__(self, X, Y):
        if self._f is None or self._Y is not Y:
            self._f = fm.MMDFixedTarget(Y, self.bw, **self.kw)
            self._Y = Y
        return self._f(X)


def timed(fn, repeats):
    fn()
    ts, xs = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        x, info = fn()
        ts.append(time.perf_counter() - t0)
        xs.append(x)
    return statistics.median(ts), xs[0], info


def main(setting="2D", repeats=5):
    TAG = {"2D": "", "5D": "_canonical", "10D": "_canonical"}
    params = load(setting)
    S_G = target_set(params)
    bw = fixed_bandwidth(S_G)
    mc = conditional_model(params, seed=SEEDS[0], tag=TAG[setting])
    mu = unconditional_model(params, seed=SEEDS[0], tag=TAG[setting])
    Ref = _guided.MMDLoss
    rows = []
    for spatial, n in (("no_lgd", 8), ("no_lgd", 32), ("lgd", 8), ("lgd", 32)):
        _guided.MMDLoss = Ref
        t_ref, x_ref, info = timed(lambda: _guided.run(mc, mu, S_G, bw, n, spatial, "none", 0), repeats)
        for name, kw in (("control_stacked_mm", None), ("fixed_mm", dict(dist="mm")),
                         ("fixed_mm_powchain", dict(dist="mm", kernel_eval="powchain"))):
            if kw is None:
                _guided.MMDLoss = StackedMMShim
            else:
                _guided.MMDLoss = lambda kernel=None, device="cpu", _kw=kw: FastMMDLossShim(kernel, device, **_kw)
            t_fast, x_fast, info2 = timed(lambda: _guided.run(mc, mu, S_G, bw, n, spatial, "none", 0), repeats)
            dx = float((x_ref - x_fast).abs().max())
            rows.append((setting, spatial, n, name, t_ref, t_fast, t_ref / t_fast, dx, info["conditional_calls"]))
            print(f"{setting} {spatial:6s} n={n:2d} {name:18s} ref {t_ref*1e3:7.1f} ms  fast {t_fast*1e3:7.1f} ms  "
                  f"speedup {t_ref/t_fast:4.2f}x  max|dx|={dx:.2e}  calls={info['conditional_calls']}", flush=True)
    _guided.MMDLoss = Ref
    out = HERE / "end_to_end_results.csv"
    with open(out, "w") as fh:
        fh.write("setting,spatial,n,variant,ref_wall_s,fast_wall_s,speedup,max_abs_dx,cond_calls\n")
        for r in rows:
            fh.write(",".join(str(v) for v in r) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["2D"]))
