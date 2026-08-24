"""Aggregate estimator/runs/*.json into screening_rows.csv and screening_tables.md."""
import csv
import json
import platform
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIM = ROOT / "simulations"
sys.path.insert(0, str(SIM / "experiments"))
sys.path.insert(0, str(SIM / "src"))
from _common import paired_stats   # noqa: E402

COMMIT = "6af2081"
HW = f"{platform.machine()} cpu"


def load_cells(runs_dir=None):
    cells = {}
    for p in sorted((runs_dir or HERE / "runs").glob("*.json")):
        d = json.loads(p.read_text())
        d["_rng"] = p.stem.split("_")[-1]
        cells[p.stem] = d
    return cells


def rows_for(cells):
    rows = []
    for name, d in cells.items():
        cfg = f"{d['spatial']}/{d['temporal']}/rng={d['_rng']}/restarts={d['restarts']}@{d['offset']}"
        rows.append({
            "commit": COMMIT, "candidate": f"A4:{d['candidate']}", "task": f"synthetic_{d['setting']}",
            "target": "S_G_250_seed987654", "seed": f"restarts_{d['offset']}..{d['offset']+d['restarts']-1}",
            "config": f"n={d['n']};{cfg}", "hardware": HW, "dtype": d["dtype"],
            "wall_s": f"{d['seconds_per_run']:.3f}", "peak_mem_mb": f"{d['peak_mem_mb']:.0f}",
            "score_calls": "", "cond_calls": f"{d['cm_samples_mean']:.0f}",
            "cond_samples": f"{d['cm_samples_mean']:.0f}",
            "opt_loss": f"{d['grad_norm_median']:.4g}" if d.get("grad_norm_median") is not None else "",
            "eval_metric": f"{d['score']:.4f}",
            "status": ("diverged=%d;success=%.2f" % (d["diverged"], d["success_rate"])),
        })
    return rows


def build_report(settings, runs_dir=None, suffix=""):
    cells = load_cells(runs_dir)
    cols = ["commit", "candidate", "task", "target", "seed", "config", "hardware", "dtype",
            "wall_s", "peak_mem_mb", "score_calls", "cond_calls", "cond_samples", "opt_loss",
            "eval_metric", "status"]
    with open(HERE / f"screening_rows{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_for(cells):
            w.writerow(r)
    lines = []
    for s in settings:
        for n in sorted({d["n"] for d in cells.values() if d["setting"] == s}):
            for sp, tp in (("no_lgd", "none"), ("no_lgd", "adam"), ("lgd", "none")):
                base_name = f"{s}_n{n}_{sp}_{tp}_baseline_tape"
                if base_name not in cells:
                    continue
                base = cells[base_name]
                lines.append(f"\n### {s}, n={n}, {sp}/{tp}  (baseline score {base['score']:.4f}, "
                             f"success {base['success_rate']:.0%}, calls {base['cm_samples_mean']:.0f}, "
                             f"{base['seconds_per_run']:.2f}s/run)\n")
                lines.append("| candidate | score | success | div | calls | s/run | RSS MB | grad-norm med | "
                             "paired diff (base-cand, + = better) | 95% CI | wins | p |")
                lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
                for name, d in sorted(cells.items()):
                    if not (d["setting"] == s and d["n"] == n and d["spatial"] == sp
                            and d["temporal"] == tp):
                        continue
                    tag = d["candidate"] + ("" if d["_rng"] == "tape" else " (legacy rng)")
                    if d["restarts"] == base["restarts"] and d["offset"] == base["offset"] \
                            and d["_rng"] == "tape" and name != base_name:
                        ps = paired_stats(base["scores"], d["scores"])
                        pair = (f"{ps['mean_diff']:+.4f} | [{ps['ci95'][0]:+.3f}, {ps['ci95'][1]:+.3f}] | "
                                f"{ps['wins_for_b']}/{ps['n']} | {ps['perm_p']:.3f}")
                    else:
                        pair = " | | | "
                    gn = d.get("grad_norm_median")
                    lines.append(f"| {tag} | {d['score']:.4f} | {d['success_rate']:.0%} | {d['diverged']} | "
                                 f"{d['cm_samples_mean']:.0f} | {d['seconds_per_run']:.2f} | "
                                 f"{d['peak_mem_mb']:.0f} | {gn:.4f} | {pair} |"
                                 if gn is not None else
                                 f"| {tag} | {d['score']:.4f} | {d['success_rate']:.0%} | {d['diverged']} | "
                                 f"{d['cm_samples_mean']:.0f} | {d['seconds_per_run']:.2f} | "
                                 f"{d['peak_mem_mb']:.0f} | | {pair} |")
    (HERE / f"screening_tables{suffix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="*", default=["2D", "5D", "10D"])
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    build_report(a.settings, Path(a.runs_dir) if a.runs_dir else None, a.suffix)
