"""Agent M -- sample-replay screening driver (mirrors estimator/screen.py).

    python experiments/model-optimization/replay/cells.py list
    python experiments/model-optimization/replay/cells.py run --index I [--restarts 40] [--offset 0]
    python experiments/model-optimization/replay/cells.py run --only replay30 baseline --settings 2D --ns 8 --restarts 10   # local smoke
    python experiments/model-optimization/replay/cells.py report

`list` prints the cell table indexed by SLURM_ARRAY_TASK_ID in
`submit_replay.sh`.  `run` executes cells in fresh subprocesses through
`estimator/engine_runner.py` and stores `runs/<name>.json` (skipped when
present).  `report` pairs every candidate with its comparator on the shared
restart indices -- plain candidates against `baseline`, `_trust` candidates
against BOTH `baseline` and `trust_noise1` -- and writes `replay_rows.csv` +
`replay_tables.md`.  Restarts 0..39 offset 0; offsets >= 1000 are the
verifier's.  Candidates are pre-registered in `hypotheses/agentM.yaml` and
implemented in `simulations/src/tfg/replay.py` (parse rules documented in
`engine_runner.candidate_spec`).
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIM = ROOT / "simulations"
for p in (SIM / "experiments", SIM / "src", ROOT / "experiments" / "model-optimization" / "estimator"):
    sys.path.insert(0, str(p))
PY = sys.executable
RUNNER = HERE.parent / "estimator" / "engine_runner.py"
CELLS = HERE / "runs"

SETTINGS = ["2D", "5D", "10D"]
NS = [4, 8, 32]
CANDIDATES = [
    # comparators (baseline + the promoted champion), in every setting
    "baseline", "trust_noise1",
    # Ori's depth-1 reuse (batch = n, fresh calls cut)
    "replay30", "replay50",
    # geometric replay (batch = n)
    "replay_geo0.3d3", "replay_geo0.5d3", "replay_geo0.7d5",
    # augment arms (fresh = n, equal calls, batch grows) -- diagnostic M-4
    "replay30_aug", "replay_geo0.5d3_aug",
    # weighted-V-statistic twins -- M-5
    "replay_w30", "replay_wgeo0.5d3",
    # + trust_noise1 combinations -- M-6
    "replay30_trust", "replay_geo0.5d3_trust", "replay30_aug_trust",
    "replay_geo0.7d5_trust",
]


def cell_list(settings=SETTINGS, ns=NS, only=None):
    cells = [(s, n, "no_lgd", "none", c, "tape")
             for s in settings for n in ns for c in CANDIDATES
             if only is None or c in only]
    return cells


def cell_name(s, n, sp, tp, c, rng):
    return f"{s}_n{n}_{sp}_{tp}_{c}_{rng}"


def run_cell(args, restarts, offset):
    s, n, sp, tp, c, rng = args
    out = CELLS / (cell_name(*args) + ".json")
    if out.exists():
        return out
    cmd = [PY, str(RUNNER), "--setting", s, "--n", str(n), "--spatial", sp,
           "--temporal", tp, "--candidate", c, "--rng", rng,
           "--restarts", str(restarts), "--offset", str(offset), "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SIM))
    if r.returncode != 0:
        print("FAILED", args, r.stderr[-2000:], flush=True)
    else:
        print(r.stdout.strip().splitlines()[-1], flush=True)
    return out


def _load(name):
    p = CELLS / (name + ".json")
    return json.loads(p.read_text()) if p.exists() else None


def report(settings=SETTINGS, ns=NS):
    from _common import paired_stats
    rows, lines = [], ["# Sample-replay screening (Agent M)\n",
                       "score = failure-penalised mean exact GMM L2 (lower better); "
                       "diff = paired comparator - candidate (+ = candidate better); "
                       "p = paired permutation.\n"]
    for s in settings:
        lines.append(f"\n## {s}\n")
        lines.append("| n | candidate | score | calls/run | vs baseline diff (p) | vs trust_noise1 diff (p) |")
        lines.append("|---|---|---|---|---|---|")
        for n in ns:
            base = _load(cell_name(s, n, "no_lgd", "none", "baseline", "tape"))
            trust = _load(cell_name(s, n, "no_lgd", "none", "trust_noise1", "tape"))
            for c in CANDIDATES:
                d = _load(cell_name(s, n, "no_lgd", "none", c, "tape"))
                if d is None:
                    continue
                cols = [str(n), c, f"{d['score']:.4f}", f"{d['cm_samples_mean']:.0f}"]
                for comp in (base, trust):
                    if comp is None or comp is d or c in ("baseline",):
                        cols.append("-")
                        continue
                    st_ = paired_stats(comp["scores"], d["scores"])
                    cols.append(f"{st_['mean_diff']:+.3f} (p={st_['perm_p']:.3f})")
                lines.append("| " + " | ".join(cols) + " |")
                rows.append({"setting": s, "n": n, "candidate": c,
                             "score": d["score"], "success": d["success_rate"],
                             "diverged": d["diverged"],
                             "cm_samples": d["cm_samples_mean"],
                             "seconds_per_run": d["seconds_per_run"]})
    (HERE / "replay_tables.md").write_text("\n".join(lines) + "\n")
    with open(HERE / "replay_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> replay_rows.csv, replay_tables.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run", "report"])
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--restarts", type=int, default=40)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--settings", nargs="*", default=SETTINGS)
    ap.add_argument("--ns", nargs="*", type=int, default=NS)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    CELLS.mkdir(exist_ok=True)
    cells = cell_list(a.settings, a.ns, a.only)
    if a.mode == "list":
        for i, c in enumerate(cells):
            print(i, cell_name(*c))
        print(f"# {len(cells)} cells", file=sys.stderr)
    elif a.mode == "run":
        if a.index is not None:
            run_cell(cells[a.index], a.restarts, a.offset)
        else:
            for c in cells:
                run_cell(c, a.restarts, a.offset)
    else:
        report(a.settings, a.ns)


if __name__ == "__main__":
    main()
