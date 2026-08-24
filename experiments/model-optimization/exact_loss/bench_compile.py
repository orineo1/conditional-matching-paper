"""torch.compile on the best eager variant (MMDFixedTarget dist=mm, powchain).

Reports compile (first-call) time separately from steady state, per shape, and
verifies the compiled value equals eager.  Small grid.  Each cell runs in a FRESH
subprocess: on macOS a second torch.compile after torch._dynamo.reset() (new
shape) hangs in inductor's compile worker pool, and the same hang hit the
bench_mmd worker; one compile per process is reliable.

    cd simulations && KMP_DUPLICATE_LIB_OK=TRUE \
        python ../experiments/model-optimization/exact_loss/bench_compile.py

Writes bench_compile_<device>.csv / .md.  Cells that hang 3x are recorded as nan.
"""
import os
import statistics
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")  # setting this hung on macOS
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT / "simulations" / "src", HERE):
    sys.path.insert(0, str(p))
if "ot" not in sys.modules:
    try:
        import ot  # noqa: F401
    except ImportError:
        sys.modules["ot"] = types.ModuleType("ot")
import torch  # noqa: E402
from LossFunctions import MMDLoss, RBF  # noqa: E402
import fast_mmd as fm  # noqa: E402

CELLS = [(1, 250, 2), (8, 250, 2), (32, 250, 2), (8, 250, 10), (8, 120, 768), (32, 250, 768)]
DTYPES = ["float32", "float64"]
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"


def fb(fn, X):
    L = fn(X)
    (g,) = torch.autograd.grad(L, X)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return L.detach(), g


def med(fn, X, reps):
    fb(fn, X)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fb(fn, X); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def run_cell(dt, n, m, d):
    if True:
        dtype = getattr(torch, dt)
        g = torch.Generator().manual_seed(0)
        Y = torch.randn(m, d, generator=g, dtype=torch.float64).to(dtype).to(DEVICE)
        X = torch.randn(n, d, generator=g, dtype=torch.float64).to(dtype).to(DEVICE).requires_grad_(True)
        with torch.no_grad():
            bw = float((torch.cdist(Y, Y) ** 2).sum() / (m * m - m))
        ref = MMDLoss(kernel=RBF(bandwidth=bw, device=DEVICE), device=DEVICE)
        f = fm.MMDFixedTarget(Y, bw, dist="mm", kernel_eval="powchain")
        torch._dynamo.reset()
        fc = torch.compile(lambda x: f(x), dynamic=False)
        t0 = time.perf_counter(); Lc, gc = fb(fc, X); t_compile = time.perf_counter() - t0
        Le, ge = fb(f, X)
        Lr, gr = fb(lambda x: ref(x, Y), X)
        reps = 50 if d < 100 else 20
        t_c, t_e, t_r = med(fc, X, reps), med(f, X, reps), med(lambda x: ref(x, Y), X, reps)
        err = max(float((Lc - Le).abs()), float((gc - ge).abs().max()))
        row = dict(device=DEVICE, dtype=dt, n_cond=n, n_target=m, dim=d, compile_s=t_compile,
                   compiled_ms=1e3 * t_c, eager_powchain_ms=1e3 * t_e, reference_ms=1e3 * t_r,
                   speedup_vs_eager=t_e / t_c, speedup_vs_reference=t_r / t_c,
                   breakeven_calls=t_compile / max(t_e - t_c, 1e-12), max_abs_err_vs_eager=err)
        return row


import csv  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402

if len(sys.argv) > 2 and sys.argv[2] == "--cell":
    dt, n, m, d = sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    print("ROW " + json.dumps(run_cell(dt, n, m, d)), flush=True)
    sys.exit(0)

rows = []
for dt in DTYPES:
    for (n, m, d) in CELLS:
        line = []
        for attempt in range(3):      # inductor's compile pool hangs intermittently on macOS
            try:
                out = subprocess.run([sys.executable, __file__, DEVICE, "--cell", dt, str(n), str(m), str(d)],
                                     capture_output=True, text=True, env=dict(os.environ), timeout=150)
            except subprocess.TimeoutExpired:
                print(f"cell {dt} n={n} m={m} d={d}: compile hang (attempt {attempt + 1}), retrying", flush=True)
                continue
            line = [l for l in out.stdout.splitlines() if l.startswith("ROW ")]
            if line:
                break
            print("cell failed", dt, n, m, d, out.stderr[-500:], flush=True)
        if not line:
            rows.append(dict(device=DEVICE, dtype=dt, n_cond=n, n_target=m, dim=d, compile_s=float("nan"),
                             compiled_ms=float("nan"), eager_powchain_ms=float("nan"), reference_ms=float("nan"),
                             speedup_vs_eager=float("nan"), speedup_vs_reference=float("nan"),
                             breakeven_calls=float("nan"), max_abs_err_vs_eager=float("nan")))
            continue
        row = json.loads(line[-1][4:])
        rows.append(row)
        print(row, flush=True)

with open(HERE / f"bench_compile_{DEVICE}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
lines = [f"# torch.compile (fixed_mm_powchain), device={DEVICE}, torch {torch.__version__}", "",
         "| dtype | n | m | d | compile s (first call) | compiled ms | eager ms | reference ms | x vs eager | x vs ref | break-even calls | max err vs eager |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['dtype']} | {r['n_cond']} | {r['n_target']} | {r['dim']} | {r['compile_s']:.1f} | {r['compiled_ms']:.3f} | "
                 f"{r['eager_powchain_ms']:.3f} | {r['reference_ms']:.3f} | {r['speedup_vs_eager']:.2f} | {r['speedup_vs_reference']:.2f} | "
                 f"{r['breakeven_calls']:.0f} | {r['max_abs_err_vs_eager']:.1e} |")
(HERE / f"bench_compile_{DEVICE}.md").write_text("\n".join(lines) + "\n")
print("wrote", HERE / f"bench_compile_{DEVICE}.md")
