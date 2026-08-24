"""Benchmark reference MMD (LossFunctions.MMDLoss) vs fast_mmd variants.

Driver mode (default) spawns one worker subprocess per variant so peak RSS is
attributable; each worker runs the whole grid in ascending-size order and records
forward+backward wall time (median of >= 20 repeats after warm-up, 7 repeats when a
single call exceeds 0.1 s) and ``resource.getrusage`` max RSS.  On MPS (float32
only) ``torch.mps.driver_allocated_memory`` is recorded as well.

    python bench_mmd.py --grid full       # full grid (cluster) -> bench_results.csv, bench_summary.md
    python bench_mmd.py --grid quick      # reduced grid
    python bench_mmd.py --grid small      # tiny local validation grid (Mac)
    python bench_mmd.py --device cuda     # GPU (float32 + float64), see submit_bench_gpu.sh
    python bench_mmd.py --worker NAME     # internal

Each worker's rows are written to bench_raw/<device>_<variant>.json as soon as it
finishes, so the driver is resumable (existing raw files are reused unless --force);
``--aggregate`` only rebuilds the CSV/summary from bench_raw/.

Grid (full): n_cond in {1,4,8,16,32,100}, n_target in {100,250,2000}, dim in {1,2,10,768},
dtype in {float32,float64}.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT / "simulations" / "src", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if "ot" not in sys.modules:
    try:
        import ot  # noqa: F401
    except ImportError:
        sys.modules["ot"] = types.ModuleType("ot")

import torch  # noqa: E402

from LossFunctions import MMDLoss, RBF  # noqa: E402
import fast_mmd as fm  # noqa: E402

PY = sys.executable
N_COND = [1, 4, 8, 16, 32, 100]
N_TARGET = [100, 250, 2000]
DIMS = [1, 2, 10, 768]
DTYPES = ["float32", "float64"]
BATCH_B = 3

VARIANTS = {
    # name: (kind, kwargs)
    "reference": ("reference", {}),
    "stacked_mm": ("reference_like", dict(dist="mm", kernel_eval="exp")),
    "stacked_powchain": ("reference_like", dict(dist="cdist", kernel_eval="powchain")),
    "fixed_cdist": ("fixed", dict(dist="cdist")),
    "fixed_mm": ("fixed", dict(dist="mm")),
    "fixed_mm_loop": ("fixed", dict(dist="mm", kernel_eval="loop")),
    "fixed_mm_powchain": ("fixed", dict(dist="mm", kernel_eval="powchain")),
    "fixed_mm_chunked256": ("fixed", dict(dist="mm", chunk=256)),
    "fixed_mm_adaptive": ("fixed_adaptive", dict(dist="mm")),          # bandwidth=None path
    "reference_adaptive": ("reference_adaptive", {}),
    "batched_fixed_mm_B3": ("batched", dict(dist="mm")),                 # B=3 sets in one call
    "batched_reference_B3": ("batched_reference", {}),                   # 3 reference calls
}
# torch.compile is benchmarked separately by bench_compile.py (inductor's compile
# worker pool hangs when run inside the captured-stdout worker subprocess here).


def rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 ** 2) if platform.system() == "Darwin" else ru / 1024


def make_fn(kind, kw, Y, bw, device):
    """Return callable(X) -> scalar loss (or (B,) for batched) for this variant."""
    if kind == "reference":
        loss = MMDLoss(kernel=RBF(bandwidth=bw, device=device), device=device)
        return lambda X: loss(X, Y)
    if kind == "reference_adaptive":
        loss = MMDLoss(kernel=RBF(bandwidth=None, device=device), device=device)
        return lambda X: loss(X, Y)
    if kind == "reference_like":
        return lambda X: fm.mmd_reference_like(X, Y, bw, **kw)
    if kind == "fixed":
        f = fm.MMDFixedTarget(Y, bw, **kw)
        return f
    if kind == "fixed_adaptive":
        f = fm.MMDFixedTarget(Y, None, **kw)
        return f
    if kind == "batched":
        f = fm.MMDFixedTarget(Y, bw, **kw)
        return lambda Xb: f.batched(Xb).sum()
    if kind == "batched_reference":
        loss = MMDLoss(kernel=RBF(bandwidth=bw, device=device), device=device)
        return lambda Xb: torch.stack([loss(x, Y) for x in Xb]).sum()
    if kind == "compiled":
        f = fm.MMDFixedTarget(Y, bw, **kw)
        g = torch.compile(lambda X: f(X), dynamic=False)
        return g
    raise ValueError(kind)


def bench_one(fn, X, repeats, device):
    def step():
        X.grad = None
        L = fn(X)
        (g,) = torch.autograd.grad(L, X)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        return float(L.detach().sum()) if L.dim() else float(L.detach())

    t0 = time.perf_counter()
    val = step()                                   # warm-up / first call
    first = time.perf_counter() - t0
    step()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        step()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), first, val


def make_grid(which):
    grid = list(itertools.product(N_COND, N_TARGET, DIMS, DTYPES))
    if which == "quick":
        grid = [(n, m, d, dt) for n, m, d, dt in grid
                if n in (1, 8, 32) and m in (250, 2000) and d in (2, 768)]
    elif which == "small":
        grid = [(n, m, d, dt) for n, m, d, dt in grid
                if n in (1, 8, 32) and m == 250 and d in (2, 768)]
    return grid


def worker(name, which, device):
    kind, kw = VARIANTS[name]
    grid = make_grid(which)
    if device == "mps":
        grid = [g for g in grid if g[3] == "float32"]
    grid.sort(key=lambda g: (g[0] + g[1]) ** 2 * g[2])    # ascending size
    rows = []
    base_rss = rss_mb()
    for n, m, d, dt in grid:
        dtype = getattr(torch, dt)
        gen = torch.Generator().manual_seed(0)
        Y = (torch.randn(m, d, generator=gen, dtype=torch.float64) * 1.0).to(dtype).to(device)
        bw = fm_fixed_bw(Y)
        if kind.startswith("batched"):
            X = torch.randn(BATCH_B, n, d, generator=gen, dtype=torch.float64).to(dtype).to(device).requires_grad_(True)
        else:
            X = torch.randn(n, d, generator=gen, dtype=torch.float64).to(dtype).to(device).requires_grad_(True)
        try:
            fn = make_fn(kind, kw, Y, bw, device)
            med, mn, first, val = bench_one(fn, X, 3, device)
            repeats = 7 if med > 0.1 else (20 if med > 5e-3 else 50)
            if which == "small":
                repeats = min(repeats, 20)
            med, mn, first2, val = bench_one(fn, X, repeats, device)
            status = "ok"
        except Exception as e:                       # pragma: no cover
            med = mn = first = float("nan")
            val = float("nan")
            status = f"error:{type(e).__name__}:{str(e)[:80]}"
        if device == "mps":
            dev_mem_mb = torch.mps.driver_allocated_memory() / 2 ** 20
        elif device == "cuda":
            dev_mem_mb = torch.cuda.max_memory_allocated() / 2 ** 20
            torch.cuda.reset_peak_memory_stats()
        else:
            dev_mem_mb = float("nan")
        rows.append(dict(variant=name, kind=kind, device=device, dtype=dt, n_cond=n, n_target=m,
                         dim=d, median_s=med, min_s=mn, first_call_s=first, value=val,
                         rss_mb=rss_mb(), rss_delta_mb=rss_mb() - base_rss, dev_mem_mb=dev_mem_mb,
                         status=status))
        if device == "mps":
            torch.mps.empty_cache()
    print(json.dumps(rows))
    raw = HERE / "bench_raw"
    raw.mkdir(exist_ok=True)
    (raw / f"{device}_{name}.json").write_text(json.dumps(rows))


def fm_fixed_bw(Y):
    with torch.no_grad():
        d2 = torch.cdist(Y, Y, p=2) ** 2
        m = Y.shape[0]
        return float(d2.sum() / (m ** 2 - m))


def driver(which, variants, devices, force=False, aggregate_only=False):
    raw = HERE / "bench_raw"
    raw.mkdir(exist_ok=True)
    rows = []
    for dev in devices:
        for name in variants:
            if dev == "mps" and name.endswith("compiled"):
                continue
            f = raw / f"{dev}_{name}.json"
            if aggregate_only or (f.exists() and not force):
                if f.exists():
                    rows.extend(json.loads(f.read_text()))
                    print(f"[bench] {dev} {name}: reused {f.name}")
                continue
            print(f"[bench] {dev} {name} ...", flush=True)
            t0 = time.perf_counter()
            args = [PY, str(HERE / "bench_mmd.py"), "--worker", name, "--device", dev, "--grid", which]
            env = dict(os.environ)
            if name.endswith("compiled"):
                # inductor's compiled C++ links its own libomp next to torch's -> OMP Error #15
                env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            out = subprocess.run(args, capture_output=True, text=True, cwd=str(ROOT / "simulations"), env=env)
            if out.returncode != 0:
                print(out.stderr[-2000:])
                continue
            last = out.stdout.strip().splitlines()[-1]
            rows.extend(json.loads(last))
            print(f"         done in {time.perf_counter() - t0:.1f}s", flush=True)
    if not rows:
        print("no rows"); return
    suffix = "" if which == "full" else f"_{which}"
    out_csv = HERE / f"bench_results{suffix}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_summary(rows, HERE / f"bench_summary{suffix}.md")
    print("wrote", out_csv)


def write_summary(rows, path):
    import collections
    ref = {(r["device"], r["dtype"], r["n_cond"], r["n_target"], r["dim"]): r
           for r in rows if r["variant"] == "reference"}
    refA = {(r["device"], r["dtype"], r["n_cond"], r["n_target"], r["dim"]): r
            for r in rows if r["variant"] == "reference_adaptive"}
    refB = {(r["device"], r["dtype"], r["n_cond"], r["n_target"], r["dim"]): r
            for r in rows if r["variant"] == "batched_reference_B3"}
    lines = ["# MMD benchmark summary", "",
             f"torch {torch.__version__}, {platform.platform()}, {platform.processor() or 'cpu'}, "
             f"threads={torch.get_num_threads()}", "",
             "Speedup = reference median forward+backward time / variant median time "
             "(same device, dtype, n_cond, n_target, dim). Geometric mean over the grid, "
             "then by regime.", ""]
    variants = [v for v in VARIANTS if any(r["variant"] == v for r in rows)]

    def base_for(v):
        if v == "fixed_mm_adaptive":
            return refA
        if v == "batched_fixed_mm_B3":
            return refB
        return ref

    def gmean(xs):
        xs = [x for x in xs if x == x and x > 0]
        return (statistics.geometric_mean(xs) if xs else float("nan"))

    for dev in sorted({r["device"] for r in rows}):
        lines += [f"## device = {dev}", "", "| variant | dtype | gmean speedup (all) | n<=8,m=250 | n=100,m=2000 | dim=768 | dim<=10 |", "|---|---|---|---|---|---|---|"]
        for v in variants:
            if v in ("reference", "reference_adaptive", "batched_reference_B3"):
                continue
            for dt in DTYPES:
                sp = collections.defaultdict(list)
                for r in rows:
                    if r["variant"] != v or r["device"] != dev or r["dtype"] != dt or r["status"] != "ok":
                        continue
                    key = (r["device"], r["dtype"], r["n_cond"], r["n_target"], r["dim"])
                    b = base_for(v).get(key)
                    if b is None or b["status"] != "ok":
                        continue
                    s = b["median_s"] / r["median_s"]
                    sp["all"].append(s)
                    if r["n_cond"] <= 8 and r["n_target"] == 250:
                        sp["small"].append(s)
                    if r["n_cond"] == 100 and r["n_target"] == 2000:
                        sp["large"].append(s)
                    if r["dim"] == 768:
                        sp["d768"].append(s)
                    if r["dim"] <= 10:
                        sp["dlow"].append(s)
                if not sp["all"]:
                    continue
                lines.append(f"| {v} | {dt} | {gmean(sp['all']):.2f} | {gmean(sp['small']):.2f} | "
                             f"{gmean(sp['large']):.2f} | {gmean(sp['d768']):.2f} | {gmean(sp['dlow']):.2f} |")
        lines.append("")
        # absolute times for a few representative cells
        lines += ["### Absolute median forward+backward times (ms)", "",
                  "| n_cond | n_target | dim | dtype | " + " | ".join(variants) + " |",
                  "|---|---|---|---|" + "---|" * len(variants)]
        cells = [(1, 250, 2), (8, 250, 2), (32, 250, 2), (8, 250, 10), (8, 100, 768), (32, 250, 768),
                 (100, 2000, 10), (100, 2000, 768)]
        for (n, m, d) in cells:
            for dt in DTYPES:
                vals = []
                for v in variants:
                    r = next((r for r in rows if r["variant"] == v and r["device"] == dev and r["dtype"] == dt
                              and r["n_cond"] == n and r["n_target"] == m and r["dim"] == d), None)
                    vals.append("-" if r is None or r["status"] != "ok" else f"{1e3 * r['median_s']:.3f}")
                if any(x != "-" for x in vals):
                    lines.append(f"| {n} | {m} | {d} | {dt} | " + " | ".join(vals) + " |")
        lines.append("")
        if dev == "cuda":
            lines += ["### CUDA peak allocated memory (MB) per cell, largest cells", "",
                      "| variant | n=32,m=250,d=768 f32 | n=100,m=2000,d=768 f32 | n=100,m=2000,d=768 f64 |", "|---|---|---|---|"]
            for v in variants:
                vals = []
                for (n, m, d, dt) in ((32, 250, 768, "float32"), (100, 2000, 768, "float32"), (100, 2000, 768, "float64")):
                    r = next((r for r in rows if r["variant"] == v and r["device"] == dev and r["dtype"] == dt
                              and r["n_cond"] == n and r["n_target"] == m and r["dim"] == d), None)
                    vals.append("-" if r is None or r["status"] != "ok" else f"{r['dev_mem_mb']:.1f}")
                lines.append(f"| {v} | " + " | ".join(vals) + " |")
            lines.append("")
        # memory: rss delta at the largest config
        lines += ["### Max RSS growth over the grid (MB, per worker process; the grid is run in ascending size order so this is dominated by the largest cell n=100, m=2000, d=768 float64)", "",
                  "| variant | rss_delta_mb (end of grid) | first-call s (largest cell) |", "|---|---|---|"]
        for v in variants:
            rs = [r for r in rows if r["variant"] == v and r["device"] == dev]
            if rs:
                big = max(rs, key=lambda r: (r["n_cond"] + r["n_target"]) ** 2 * r["dim"])
                lines.append(f"| {v} | {rs[-1]['rss_delta_mb']:.1f} | {big['first_call_s']:.3f} |")
        lines.append("")
    comp = [r for r in rows if r["variant"] == "fixed_mm_powchain_compiled"]   # from bench_compile.py
    if comp:
        lines += ["## torch.compile (fixed_mm_powchain)", "",
                  "| dtype | n_cond | n_target | dim | first call (compile) s | steady median ms | eager powchain ms | reference ms |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in comp:
            if r["status"] != "ok":
                lines.append(f"| {r['dtype']} | {r['n_cond']} | {r['n_target']} | {r['dim']} | {r['status']} | | | |")
                continue
            key = (r["device"], r["dtype"], r["n_cond"], r["n_target"], r["dim"])
            e = next((x for x in rows if x["variant"] == "fixed_mm_powchain" and
                      (x["device"], x["dtype"], x["n_cond"], x["n_target"], x["dim"]) == key), None)
            b = ref.get(key)
            lines.append(f"| {r['dtype']} | {r['n_cond']} | {r['n_target']} | {r['dim']} | {r['first_call_s']:.2f} | "
                         f"{1e3 * r['median_s']:.3f} | {1e3 * e['median_s'] if e else float('nan'):.3f} | "
                         f"{1e3 * b['median_s'] if b else float('nan'):.3f} |")
        lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--grid", default="full", choices=["full", "quick", "small"])
    ap.add_argument("--variants", default=None, help="comma list")
    ap.add_argument("--force", action="store_true", help="re-run even if bench_raw json exists")
    ap.add_argument("--aggregate", action="store_true", help="only rebuild csv/summary from bench_raw")
    a = ap.parse_args()
    if a.worker:
        torch.manual_seed(0)
        worker(a.worker, a.grid, a.device)
    else:
        vs = a.variants.split(",") if a.variants else list(VARIANTS)
        devs = [d.strip() for d in a.device.split(",")]
        driver(a.grid, vs, devs, force=a.force, aggregate_only=a.aggregate)
