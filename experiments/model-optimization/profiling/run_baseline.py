"""Agent 1 -- baseline reproduction of simulations/experiments/_guided.py::run.

Each (setting, arm, n) cell runs in a fresh subprocess so that peak RSS is
per-cell. Inside a cell: for each restart, one warm-up run, then REPEATS timed
runs (identical seeds -> identical trajectory), median/min/max wall time.

    python experiments/model-optimization/profiling/run_baseline.py            # driver
    python experiments/model-optimization/profiling/run_baseline.py --cell 2D no_lgd none 8

Writes profiling/baseline_rows.csv (results.csv column set) and
profiling/baseline_runs.json (full per-run detail).
"""
import argparse
import csv
import json
import os
import platform
import resource
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIM = ROOT / "simulations"
sys.path.insert(0, str(SIM / "src"))
sys.path.insert(0, str(SIM / "experiments"))

PY = "/Users/stolk/miniconda3/bin/python"
COMMIT = "6af2081"
TAG = {"2D": "", "5D": "_canonical", "10D": "_canonical"}   # checkpoint tags used by exp3
ARMS = [("no_lgd", "none"), ("no_lgd", "adam"), ("lgd", "none")]
NS = [8, 32]
RESTARTS = list(range(5))
REPEATS = 5


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def run_cell(setting, spatial, temporal, n, restarts, repeats):
    import torch
    from _common import fixed_bandwidth, load, target_set
    from _guided import evaluate, run
    from _models import SEEDS, conditional_model, unconditional_model
    torch.set_num_threads(torch.get_num_threads())
    params = load(setting)
    S_G = target_set(params)
    bw = fixed_bandwidth(S_G)
    mc = conditional_model(params, seed=SEEDS[0], tag=TAG[setting])
    mu = unconditional_model(params, seed=SEEDS[0], tag=TAG[setting])
    rss_after_load = peak_rss_mb()
    out = []
    # global warm-up (kernels, allocator)
    run(mc, mu, S_G, bw, n, spatial, temporal, restarts[0])
    for r in restarts:
        x0, info0 = run(mc, mu, S_G, bw, n, spatial, temporal, r)   # warm-up for this cell
        times, xs = [], []
        for _ in range(repeats):
            t0 = time.perf_counter()
            x, info = run(mc, mu, S_G, bw, n, spatial, temporal, r)
            times.append(time.perf_counter() - t0)
            xs.append(x)
        ev = evaluate(x, params, info)
        same = all(torch.equal(xs[0], xi) for xi in xs[1:]) and torch.equal(x0, xs[0])
        # final guidance loss: recompute MMD^2 at the final x via one more
        # conditional draw (not part of the loop); cheap diagnostic.
        from LossFunctions import MMDLoss, RBF
        from _models import PAPER_TS
        from _common import key_seed
        mmd = MMDLoss(kernel=RBF(bandwidth=bw))
        torch.manual_seed(key_seed("cond", r, 0, 0))
        with torch.no_grad():
            y, _, _ = mc.sample(nsamples=n, condition_x=x.reshape(1, -1).repeat(n, 1).float(),
                                ts=PAPER_TS)
            final_mmd2 = float(mmd(y, S_G))
        out.append({"setting": setting, "spatial": spatial, "temporal": temporal, "n": n,
                    "restart": r, "wall_median": st.median(times), "wall_min": min(times),
                    "wall_max": max(times), "wall_all": times,
                    "deterministic_repeats": bool(same),
                    "conditional_calls": info["conditional_calls"],
                    "cond_sampler_calls": info["steps"] * (3 if spatial == "lgd" else 1),
                    "steps": info["steps"], "diverged": info["diverged"],
                    "L2": ev["L2"], "L2_squared": ev.get("L2_squared"),
                    "abs_err": ev["abs_err"], "x_hat": ev["x_hat"],
                    "final_mmd2": final_mmd2, "bandwidth": bw})
    return {"rows": out, "peak_rss_mb": peak_rss_mb(), "rss_after_load_mb": rss_after_load,
            "dim_x": int(params["x_star"].numel()), "torch": torch.__version__,
            "threads": torch.get_num_threads()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", nargs=4, metavar=("SETTING", "SPATIAL", "TEMPORAL", "N"))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--restarts", type=int, default=len(RESTARTS))
    a = ap.parse_args()
    if a.cell:
        s, sp, tp, n = a.cell
        res = run_cell(s, sp, tp, int(n), list(range(a.restarts)), a.repeats)
        print("@@JSON@@" + json.dumps(res))
        return
    cells, rows = [], []
    for setting in ("2D", "5D", "10D"):
        for spatial, temporal in ARMS:
            for n in NS:
                cmd = [PY, str(Path(__file__).resolve()), "--cell", setting, spatial, temporal,
                       str(n), "--repeats", str(a.repeats), "--restarts", str(a.restarts)]
                t0 = time.perf_counter()
                p = subprocess.run(cmd, cwd=str(SIM), capture_output=True, text=True)
                if p.returncode != 0:
                    print(p.stderr[-3000:]); raise SystemExit(1)
                res = json.loads(p.stdout.split("@@JSON@@")[1])
                res.update({"setting": setting, "spatial": spatial, "temporal": temporal,
                            "n": n, "cell_wall_s": time.perf_counter() - t0})
                cells.append(res)
                meds = [r["wall_median"] for r in res["rows"]]
                print(f"{setting:>3} {spatial:<6} {temporal:<4} n={n:<3} "
                      f"wall med={st.median(meds):.3f}s [{min(meds):.3f},{max(meds):.3f}] "
                      f"calls={res['rows'][0]['conditional_calls']} "
                      f"peakRSS={res['peak_rss_mb']:.0f}MB "
                      f"L2={[round(r['L2'],3) for r in res['rows']]}", flush=True)
                for r in res["rows"]:
                    rows.append({
                        "commit": COMMIT, "candidate": "baseline",
                        "task": f"synthetic_{setting}", "target": "S_G250_seed987654",
                        "seed": r["restart"],
                        "config": f"spatial={spatial};temporal={temporal};n={n};schedule=constant;rho=0.4;T=100;ts=PAPER_TS",
                        "hardware": f"{platform.machine()}-cpu-{res['threads']}thr",
                        "dtype": "float32",
                        "wall_s": f"{r['wall_median']:.4f}",
                        "peak_mem_mb": f"{res['peak_rss_mb']:.1f}",
                        "score_calls": r["steps"],
                        "cond_calls": r["cond_sampler_calls"],
                        "cond_samples": r["conditional_calls"],
                        "opt_loss": f"{r['final_mmd2']:.6g}",
                        "eval_metric": f"{r['L2']:.6g}",
                        "status": "diverged" if r["diverged"] else "ok"})
    with open(HERE / "baseline_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (HERE / "baseline_runs.json").write_text(json.dumps(
        {"commit": COMMIT, "python": sys.version.split()[0],
         "platform": platform.platform(), "cells": cells}, indent=1))
    print("written baseline_rows.csv, baseline_runs.json")


if __name__ == "__main__":
    main()
