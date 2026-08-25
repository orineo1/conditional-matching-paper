"""Agent P -- round-3 preconditioning screening driver (paired seeds, engine vs engine).

    python experiments/model-optimization/precond/cells.py list
    python experiments/model-optimization/precond/cells.py run --index I [--restarts 40]   # one cell (cluster array task)
    python experiments/model-optimization/precond/cells.py run [--jobs 1]                  # all cells, local
    python experiments/model-optimization/precond/cells.py report

`list` prints the cell table (index -> cell) that `submit_precond.sh` indexes
with SLURM_ARRAY_TASK_ID. `run` executes each cell in a fresh subprocess via
`estimator/engine_runner.py` (unchanged screening path, rng=tape, float32) and
stores `runs/<name>.json` (skipped if present). Every setting carries the
`baseline` AND `trust_noise1` comparator arms so each preconditioner is judged
against both the un-regularised baseline and the promoted champion on the SAME
paired restarts. Restarts 0..39 at offset 0; offsets >= 1000 stay reserved for
the verifier.

`report` pairs every candidate cell with its baseline cell (and with
trust_noise1) and prints a per-setting markdown table with the paired diff,
bootstrap 95% CI and permutation p (_common.paired_stats), plus
`precond_rows.csv` in the results.csv column convention.
"""
import argparse
import csv
import json
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIM = ROOT / "simulations"
EST = HERE.parent / "estimator"
sys.path.insert(0, str(SIM / "experiments"))
sys.path.insert(0, str(SIM / "src"))
PY = sys.executable
CELLS = HERE / "runs"
COMMIT = "6af2081+P"

SETTINGS = ["2D", "5D", "10D"]
NS = [4, 8, 32]
# baseline + champion comparators first, then the 4 preconditioners x {alone, +trust}
CANDIDATES = ["baseline", "trust_noise1",
              "precond_cov", "precond_cov_trust",
              "precond_diag", "precond_diag_trust",
              "precond_sign", "precond_sign_trust",
              "precond_median", "precond_median_trust"]


def cell_list(settings=SETTINGS, ns=NS, candidates=CANDIDATES):
    return [(s, n, "no_lgd", "none", c, "tape")
            for s in settings for n in ns for c in candidates]


def cell_name(s, n, sp, tp, c, rng):
    return f"{s}_n{n}_{sp}_{tp}_{c}_{rng}"


def run_cell(args, restarts, offset):
    s, n, sp, tp, c, rng = args
    CELLS.mkdir(exist_ok=True)
    out = CELLS / (cell_name(*args) + ".json")
    if out.exists():
        return out
    cmd = [PY, str(EST / "engine_runner.py"), "--setting", s, "--n", str(n),
           "--spatial", sp, "--temporal", tp, "--candidate", c, "--rng", rng,
           "--restarts", str(restarts), "--offset", str(offset), "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SIM))
    if r.returncode != 0:
        print("FAILED", args, r.stderr[-2000:], flush=True)
    else:
        print(r.stdout.strip().splitlines()[-1], flush=True)
    return out


def load_cell(args):
    p = CELLS / (cell_name(*args) + ".json")
    return json.loads(p.read_text()) if p.exists() else None


def report():
    from _common import paired_stats
    rows = []
    lines = ["# Agent P -- preconditioning screening (paired, restarts 0..39)",
             "",
             "diff = paired mean (comparator - candidate), + = candidate better;",
             "p = paired permutation; vsB = vs baseline, vsT = vs trust_noise1.", ""]
    for s in SETTINGS:
        lines += [f"## {s}", "",
                  "| candidate | " + " | ".join(
                      f"n={n} score / vsB (p) / vsT (p)" for n in NS) + " |",
                  "|---|" + "---|" * len(NS)]
        for c in CANDIDATES:
            parts = []
            for n in NS:
                cand = load_cell((s, n, "no_lgd", "none", c, "tape"))
                base = load_cell((s, n, "no_lgd", "none", "baseline", "tape"))
                champ = load_cell((s, n, "no_lgd", "none", "trust_noise1", "tape"))
                if cand is None:
                    parts.append("--")
                    continue
                cell_txt = f"{cand['score']:.3f}"
                for ref, tag in ((base, "vsB"), (champ, "vsT")):
                    if ref is None or ref is cand or c == "baseline" and tag == "vsB" \
                       or c == "trust_noise1" and tag == "vsT":
                        cell_txt += " / --"
                        continue
                    st_ = paired_stats(ref["scores"], cand["scores"])
                    cell_txt += (f" / {st_['mean_diff']:+.3f} "
                                 f"[{st_['ci95'][0]:+.2f},{st_['ci95'][1]:+.2f}] "
                                 f"(p={st_['perm_p']:.3f})")
                parts.append(cell_txt)
                rows.append({"commit": COMMIT, "candidate": c,
                             "task": f"{s}_n{n}_no_lgd_none", "target": "synthetic",
                             "seed": "0..39", "config": "precond round3",
                             "hardware": platform.processor() or platform.machine(),
                             "dtype": "float32", "wall_s": cand["wall_s"],
                             "peak_mem_mb": cand["peak_mem_mb"],
                             "score_calls": "", "cond_calls": cand["conditional_calls_mean"],
                             "cond_samples": cand["cm_samples_mean"],
                             "opt_loss": "", "eval_metric": cand["score"],
                             "status": "ok"})
            lines.append("| " + c + " | " + " | ".join(parts) + " |")
        lines.append("")
    (HERE / "precond_tables.md").write_text("\n".join(lines))
    with open(HERE / "precond_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {HERE / 'precond_tables.md'} and precond_rows.csv ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run", "report"])
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--restarts", type=int, default=40)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--settings", nargs="*", default=SETTINGS)
    ap.add_argument("--ns", nargs="*", type=int, default=NS)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    cands = a.only or CANDIDATES
    cells = cell_list(a.settings, a.ns, cands)
    if a.mode == "list":
        for i, cl in enumerate(cells):
            print(i, cell_name(*cl))
    elif a.mode == "run":
        if a.index is not None:
            run_cell(cells[a.index], a.restarts, a.offset)
        else:
            with ThreadPoolExecutor(max_workers=a.jobs) as ex:
                list(ex.map(lambda cl: run_cell(cl, a.restarts, a.offset), cells))
    else:
        report()


if __name__ == "__main__":
    main()
