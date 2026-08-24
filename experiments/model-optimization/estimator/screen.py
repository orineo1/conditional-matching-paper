"""Agent 4 -- Stage-1 synthetic screening driver (paired seeds, engine vs engine).

    python experiments/model-optimization/estimator/screen.py list   [--settings 2D 5D 10D] [--round 2]
    python experiments/model-optimization/estimator/screen.py run --index I [--restarts 40]   # one cell (cluster array task)
    python experiments/model-optimization/estimator/screen.py run   [--jobs 1] [--restarts 40] [--settings 2D]   # all cells, local
    python experiments/model-optimization/estimator/screen.py report

`list` prints the cell table (index -> cell) that `submit_screen.sh` indexes
with SLURM_ARRAY_TASK_ID. `run` executes each cell in a fresh subprocess (so
peak RSS is per cell) and stores `runs/<name>.json` (skipped if present).
`report` pairs each candidate cell with its baseline cell (same setting, n,
spatial, temporal, restarts 0..R-1, same tape seeds) and writes
`screening_rows.csv` (results.csv columns) and `screening_tables.md`.
Restart offset 0 is used; offsets >= 1000 are reserved for the verifier.
"""
import argparse
import csv
import itertools
import json
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIM = ROOT / "simulations"
sys.path.insert(0, str(SIM / "experiments"))
sys.path.insert(0, str(SIM / "src"))
PY = sys.executable
CELLS = HERE / "runs"
COMMIT = "6af2081"

# candidates run on the no-LGD / none arm unless listed otherwise
CANDIDATES_NONE = ["baseline", "norm_only", "clip0.5", "clip0.1", "unit0.4", "unit0.1",
                   "adapt_agree0.5", "adapt_agree0.8", "adapt_improve", "crn", "antithetic",
                   "stale2", "stale3", "recur2_next_state_tweedie",
                   "bw_pooled", "bw_pooled_floor", "sqrt_abs_eps", "sqrt_floor"]
CANDIDATES_ADAM = ["baseline", "crn", "antithetic", "adapt_agree0.5", "clip0.5"]
CANDIDATES_LGD = ["baseline"]

# round 2: scale-free clipping rules, combinations, Pareto cells (n in {4,8,16,32})
R2_NONE = ["baseline", "clip0.5", "clip0.1", "sqrt_floor",
           "relclip0.5", "relclip1", "relclip2",
           "relclip_ema0.5", "relclip_ema1", "relclip_ema2",
           "qclip0.5", "qclip0.75",
           "trust_noise0.1", "trust_noise0.3", "trust_noise1",
           "trust_ddim0.1", "trust_ddim0.3", "trust_ddim1",
           "sqrtfloor_clip0.5", "sqrtfloor_clip0.1", "sqrtfloor_relclip1"]
R2_ADAM = ["baseline", "clip0.5", "clip0.1", "relclip1"]
R2_LGD = ["baseline", "clip0.5", "clip0.1", "relclip1"]
R2_NS, R2_NS_LGD = [4, 8, 16, 32], [8, 32]


def cell_list_round2(settings):
    cells = []
    for s in settings:
        for n in R2_NS:
            for c in R2_NONE:
                cells.append((s, n, "no_lgd", "none", c, "tape"))
            for c in R2_ADAM:
                cells.append((s, n, "no_lgd", "adam", c, "tape"))
        for n in R2_NS_LGD:
            for c in R2_LGD:
                cells.append((s, n, "lgd", "none", c, "tape"))
    return cells


def cell_list(settings, ns, extra_n=()):
    cells = []
    for s in settings:
        for n in ns:
            for c in CANDIDATES_NONE:
                cells.append((s, n, "no_lgd", "none", c, "tape"))
            for c in CANDIDATES_ADAM:
                cells.append((s, n, "no_lgd", "adam", c, "tape"))
            for c in CANDIDATES_LGD:
                cells.append((s, n, "lgd", "none", c, "tape"))
            # legacy-RNG baselines: the README's numbers, restarts 0..R-1
            for arm in (("no_lgd", "none"), ("no_lgd", "adam"), ("lgd", "none")):
                cells.append((s, n, arm[0], arm[1], "baseline", "legacy"))
        for n in extra_n:      # budget-matched partners for stale-k (n*k) and recur (n/2)
            for c in ("baseline",):
                cells.append((s, n, "no_lgd", "none", c, "tape"))
    return cells


def cell_name(s, n, sp, tp, c, rng):
    return f"{s}_n{n}_{sp}_{tp}_{c}_{rng}"


def run_cell(args, restarts, offset):
    s, n, sp, tp, c, rng = args
    out = CELLS / (cell_name(*args) + ".json")
    if out.exists():
        return out
    cmd = [PY, str(HERE / "engine_runner.py"), "--setting", s, "--n", str(n), "--spatial", sp,
           "--temporal", tp, "--candidate", c, "--rng", rng, "--restarts", str(restarts),
           "--offset", str(offset), "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SIM))
    if r.returncode != 0:
        print("FAILED", args, r.stderr[-2000:], flush=True)
    else:
        print(r.stdout.strip().splitlines()[-1], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run", "report"])
    ap.add_argument("--index", type=int, default=None, help="run only this cell of `list`")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--restarts", type=int, default=40)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--settings", nargs="*", default=["2D", "5D", "10D"])
    ap.add_argument("--ns", nargs="*", type=int, default=[4, 8, 32])
    ap.add_argument("--extra-n", nargs="*", type=int, default=[2, 16, 24, 64, 96])
    ap.add_argument("--only", nargs="*", default=None, help="candidate names to restrict to")
    ap.add_argument("--round", type=int, default=1, help="1: round-1 grid; 2: scale-free clipping / combos / Pareto")
    a = ap.parse_args()
    CELLS.mkdir(exist_ok=True)
    cells = (cell_list(a.settings, a.ns, a.extra_n) if a.round == 1
             else cell_list_round2(a.settings))
    if a.only:
        cells = [c for c in cells if c[4] in a.only]
    if a.mode == "list":
        for i, c in enumerate(cells):
            print(i, cell_name(*c))
        print(f"# {len(cells)} cells", file=sys.stderr)
    elif a.mode == "run":
        if a.index is not None:
            run_cell(cells[a.index], a.restarts, a.offset)
            return
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            list(ex.map(lambda c: run_cell(c, a.restarts, a.offset), cells))
    else:
        from report import build_report
        build_report(a.settings)


if __name__ == "__main__":
    main()
