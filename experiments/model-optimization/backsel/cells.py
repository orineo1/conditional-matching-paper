"""Agent B -- importance-selected backprop screening driver (mirrors replay/cells.py).

    python experiments/model-optimization/backsel/cells.py list
    python experiments/model-optimization/backsel/cells.py run --index I [--restarts 40] [--offset 0]
    python experiments/model-optimization/backsel/cells.py run --only backsel_is_k2_trust trust_noise1 --settings 2D --ns 8 --restarts 10 --dir smoke
    python experiments/model-optimization/backsel/cells.py report [--dir runs]

`list` prints the cell table indexed by SLURM_ARRAY_TASK_ID in
`submit_backsel.sh`.  `run` executes cells in fresh subprocesses through
`estimator/engine_runner.py` (fast MMD backend) and stores
`<dir>/<name>.json` (skipped when present).  `report` pairs every candidate
with its comparators on the shared restart indices -- plain arms against
`baseline`, `_trust` arms against `trust_noise1`, `_cohort` arms against
Agent C's `replay_cohort<B>_trust` -- and writes `backsel_rows.csv` +
`backsel_tables.md` with BOTH cost currencies (cm_samples = conditional
forwards, diff_samples = differentiated samples), wall s/run and peak RSS.
Restarts 0..39 offset 0; offsets >= 1000 are the verifier's.  Candidates are
pre-registered in `hypotheses/agentB.yaml`, implemented in
`simulations/src/tfg/backsel.py`, parsed in `engine_runner.candidate_spec`.
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

SETTINGS = ["2D", "5D", "10D"]
NS = [8, 32]
RULES = ["uni", "is", "clust"]
KS = [2, 4]


def candidates(n):
    """Candidate names for batch size ``n`` (the cohort batch is ``2n``)."""
    c = ["baseline", "trust_noise1"]
    c += [f"backsel_{r}_k{k}" for r in RULES for k in KS]                   # B-1..B-3
    c += [f"backsel_{r}_k{k}_trust" for r in RULES for k in KS]             # B-4
    B = 2 * n                                                                # x Agent C cohort
    c += [f"replay_cohort{B}_trust", f"backsel_is_k4_cohort{B}_trust",
          f"backsel_clust_k4_cohort{B}_trust"]
    return c


def comparator_of(name, n):
    if name == "baseline":
        return None
    if "_cohort" in name and name.startswith("backsel"):
        return f"replay_cohort{2 * n}_trust"
    if name.endswith("_trust"):
        return "trust_noise1"
    return "baseline"


def cell_list(settings=SETTINGS, ns=NS, only=None):
    return [(s, n, "no_lgd", "none", c, "tape")
            for s in settings for n in ns for c in candidates(n)
            if only is None or c in only]


def cell_name(s, n, sp, tp, c, rng):
    return f"{s}_n{n}_{sp}_{tp}_{c}_{rng}"


def run_cell(args, restarts, offset, out_dir):
    s, n, sp, tp, c, rng = args
    out = out_dir / (cell_name(*args) + ".json")
    if out.exists():
        return out
    cmd = [PY, str(RUNNER), "--setting", s, "--n", str(n), "--spatial", sp,
           "--temporal", tp, "--candidate", c, "--rng", rng, "--loss", "fast",
           "--restarts", str(restarts), "--offset", str(offset), "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SIM))
    if r.returncode != 0:
        print("FAILED", args, r.stderr[-2000:], flush=True)
    else:
        print(r.stdout.strip().splitlines()[-1], flush=True)
    return out


def _load(out_dir, name):
    p = out_dir / (name + ".json")
    return json.loads(p.read_text()) if p.exists() else None


def report(out_dir, settings=SETTINGS, ns=NS, tag=""):
    from _common import paired_stats
    rows, lines = [], [f"# Importance-selected backprop screening (Agent B){tag}\n",
                       "score = failure-penalised mean exact GMM L2 (lower better); "
                       "diff = paired comparator - candidate (+ = candidate better); "
                       "p = paired permutation.  fwd = conditional forward samples per "
                       "run (cm_samples), diff_s = DIFFERENTIATED samples per run "
                       "(graphs + backward).\n"]
    for s in settings:
        lines.append(f"\n## {s}\n")
        lines.append("| n | candidate | score | succ | div | fwd/run | diff_s/run | s/run | RSS MB | comparator | diff (p) | vs baseline diff (p) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for n in ns:
            base = _load(out_dir, cell_name(s, n, "no_lgd", "none", "baseline", "tape"))
            for c in candidates(n):
                d = _load(out_dir, cell_name(s, n, "no_lgd", "none", c, "tape"))
                if d is None:
                    continue
                comp_name = comparator_of(c, n)
                comp = _load(out_dir, cell_name(s, n, "no_lgd", "none", comp_name, "tape")) if comp_name else None
                cols = [str(n), c, f"{d['score']:.4f}", f"{d['success_rate']:.0%}", str(d["diverged"]),
                        f"{d['cm_samples_mean']:.0f}", f"{d.get('diff_samples_mean', d['cm_samples_mean']):.0f}",
                        f"{d['seconds_per_run']:.2f}", f"{d['peak_mem_mb']:.0f}", comp_name or "-"]
                rec = {"setting": s, "n": n, "candidate": c, "score": d["score"],
                       "success": d["success_rate"], "diverged": d["diverged"],
                       "cm_samples": d["cm_samples_mean"],
                       "diff_samples": d.get("diff_samples_mean", d["cm_samples_mean"]),
                       "seconds_per_run": d["seconds_per_run"], "peak_mem_mb": d["peak_mem_mb"],
                       "comparator": comp_name}
                for cc, key in ((comp, "diff_vs_comparator"), (base, "diff_vs_baseline")):
                    if cc is None or cc is d or c == "baseline":
                        cols.append("-")
                        rec[key], rec[key + "_p"] = None, None
                        continue
                    st_ = paired_stats(cc["scores"], d["scores"])
                    cols.append(f"{st_['mean_diff']:+.3f} (p={st_['perm_p']:.3f})")
                    rec[key], rec[key + "_p"] = st_["mean_diff"], st_["perm_p"]
                lines.append("| " + " | ".join(cols) + " |")
                rows.append(rec)
    (HERE / f"backsel_tables{tag}.md").write_text("\n".join(lines) + "\n")
    if rows:
        with open(HERE / f"backsel_rows{tag}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} rows -> backsel_rows{tag}.csv, backsel_tables{tag}.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run", "report"])
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--restarts", type=int, default=40)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--settings", nargs="*", default=SETTINGS)
    ap.add_argument("--ns", nargs="*", type=int, default=NS)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--dir", default="runs", help="output subdirectory (runs | smoke)")
    a = ap.parse_args()
    out_dir = HERE / a.dir
    out_dir.mkdir(exist_ok=True)
    cells = cell_list(a.settings, a.ns, a.only)
    if a.mode == "list":
        for i, c in enumerate(cells):
            print(i, cell_name(*c))
        print(f"# {len(cells)} cells", file=sys.stderr)
    elif a.mode == "run":
        if a.index is not None:
            run_cell(cells[a.index], a.restarts, a.offset, out_dir)
        else:
            for c in cells:
                run_cell(c, a.restarts, a.offset, out_dir)
    else:
        report(out_dir, a.settings, a.ns, tag=("" if a.dir == "runs" else f"_{a.dir}"))


if __name__ == "__main__":
    main()
