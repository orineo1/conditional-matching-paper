"""Agent 6 -- round-4 analysis (M-10 FIFO/cohort replay vs trust_noise1 at equal fresh cost).

    cd simulations && /Users/stolk/miniconda3/bin/python ../experiments/model-optimization/verification/analyze_r4.py [--offset 5000]
Writes verification/heldout_r4_tables.md.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "simulations" / "experiments"))
sys.path.insert(0, str(ROOT / "simulations" / "src"))
from _common import PENALTY, paired_stats  # noqa: E402

RUNS = HERE / "heldout_runs"
M9 = HERE.parent / "replay" / "runs_m9"
M10 = HERE.parent / "replay" / "runs_m10"
TRUST, FIFO, COH = "trust_noise1", "replay_fifo16_trust", "replay_cohort16_trust"


def pen(d):
    return [min(q["L2"], PENALTY) if not q["diverged"] else PENALTY for q in d["runs"]]


def penm(d):
    return [min(q["mmd2_eval"], 1.0) if q.get("mmd2_eval") is not None else 1.0 for q in d["runs"]]


def succ(d):
    return sum(1 for q in d["runs"] if not q["diverged"] and q["abs_err"] < 0.5) / len(d["runs"])


def screening_diff(s, f, arm):
    a = M9 / f"{s}_f{f}_A.json"
    c = M10 / f"{s}_f{f}_{'fifo16' if arm == FIFO else 'cohort16'}.json"
    if a.exists() and c.exists():
        ps = paired_stats(json.load(open(a))["scores"], json.load(open(c))["scores"])
        return f"{ps['mean_diff']:+.4f} (p={ps['perm_p']:.3f})"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=5000)
    a = ap.parse_args()
    cells = {}
    for p in sorted(RUNS.glob(f"*_off{a.offset}.json")):
        d = json.load(open(p))
        cells[(d["setting"], d["candidate"], d["n"])] = d
    if not cells:
        print("no runs at offset", a.offset)
        return
    md = [f"# Round-4 held-out (offset {a.offset}, Agent 6): FIFO/cohort replay vs trust_noise1 at equal fresh cost\n",
          f"{len(cells)} cells; paired diff = trust_noise1@f - candidate@f (+ = candidate better); score = "
          "failure-penalised exact GMM L2; mmd2_eval = objective at x_hat (256 fresh draws); 'M-10' = the "
          "implementer's offset-4000 estimate of the same diff.\n"]
    for s in ("2D", "5D", "10D"):
        if not any(k[0] == s for k in cells):
            continue
        ref = cells.get((s, TRUST, 8))
        md.append(f"\n## {s}" + (f"  (reference trust_noise1@8, 792 calls: score {st.mean(pen(ref)):.4f})" if ref else "") + "\n")
        md.append("| f | arm | calls | score | success | div | mmd2_eval | diff L2 vs trust@f | 95% CI | wins | p | diff mmd2_eval | p | M-10 (off 4000) | vs trust@8 diff |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for f in (2, 4):
            t = cells.get((s, TRUST, f))
            for arm in (TRUST, FIFO, COH):
                d = cells.get((s, arm, f))
                if d is None:
                    continue
                row = [str(f), arm, f"{d['cm_samples_mean']:.0f}", f"{st.mean(pen(d)):.4f}", f"{succ(d):.0%}",
                       str(d["diverged"]), f"{st.mean(penm(d)):.4f}"]
                if arm != TRUST and t is not None:
                    ps, pm = paired_stats(pen(t), pen(d)), paired_stats(penm(t), penm(d))
                    row += [f"{ps['mean_diff']:+.4f}", f"[{ps['ci95'][0]:+.3f}, {ps['ci95'][1]:+.3f}]",
                            f"{ps['wins_for_b']}/{ps['n']}", f"{ps['perm_p']:.3f}",
                            f"{pm['mean_diff']:+.4f}", f"{pm['perm_p']:.3f}", screening_diff(s, f, arm)]
                else:
                    row += [""] * 7
                if ref is not None:
                    pr = paired_stats(pen(ref), pen(d))
                    row.append(f"{pr['mean_diff']:+.4f} (p={pr['perm_p']:.3f})")
                else:
                    row.append("")
                md.append("| " + " | ".join(row) + " |")
        # cross-f: candidate@2 (198 calls) vs trust@4 (396)
        t4 = cells.get((s, TRUST, 4))
        for arm in (FIFO, COH):
            d2 = cells.get((s, arm, 2))
            if t4 is not None and d2 is not None:
                ps = paired_stats(pen(t4), pen(d2))
                md.append(f"\ncall-halving check: {arm}@2 (198) vs trust@4 (396): {ps['mean_diff']:+.4f} "
                          f"[{ps['ci95'][0]:+.3f}, {ps['ci95'][1]:+.3f}] p={ps['perm_p']:.3f}")
    (HERE / "heldout_r4_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {HERE / 'heldout_r4_tables.md'}")


if __name__ == "__main__":
    main()
