"""Agent 6 -- round-5 confirmatory analysis (corrected protocol, offset 7000).

    cd simulations && /Users/stolk/miniconda3/bin/python ../experiments/model-optimization/verification/analyze_r5.py [--offset 7000]
Writes verification/heldout_r5_tables.md: A vs B paired diff (all restarts, penalised) and
restricted to pairs where neither arm diverged; A8 (zeta=8) vs B and vs A in 2D; mmd2_eval;
the round-5 offset-6000 estimate alongside.
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

RUNS = HERE / "heldout_runs_r5"
R5 = HERE.parent / "protocol" / "runs_r5"


def pen(d):
    return [min(q["L2"], PENALTY) if not q["diverged"] else PENALTY for q in d["runs"]]


def penm(d):
    return [min(q["mmd2_eval"], 1.0) if q.get("mmd2_eval") is not None else 1.0 for q in d["runs"]]


def succ(d):
    return sum(1 for q in d["runs"] if not q["diverged"] and q["abs_err"] < 0.5) / len(d["runs"])


def r5_est(s, n):
    a, b = R5 / f"{s}_n{n}_A_trust_zt.json", R5 / f"{s}_n{n}_B_notrust_zn.json"
    if a.exists() and b.exists():
        A, B = json.load(open(a)), json.load(open(b))
        ps = paired_stats(pen(B), pen(A))
        return f"{ps['mean_diff']:+.4f} (p={ps['perm_p']:.3f})"
    return ""


def fmt(ps):
    return f"{ps['mean_diff']:+.4f} [{ps['ci95'][0]:+.3f}, {ps['ci95'][1]:+.3f}] {ps['wins_for_b']}/{ps['n']} p={ps['perm_p']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=7000)
    a = ap.parse_args()
    cells = {}
    for p in sorted(RUNS.glob(f"*_off{a.offset}.json")):
        d = json.load(open(p))
        cells[(d["setting"], d["n"], d["arm"])] = d
    if not cells:
        print("no runs at offset", a.offset)
        return
    md = [f"# Round-5 confirmatory re-run (offset {a.offset}, corrected protocol, Agent 6)\n",
          f"{len(cells)} cells; diff = B - A (+ = trust better), failure-penalised exact L2 (cap 2.0); "
          "'non-div pairs' = restricted to restarts where neither arm diverged; 'r5 (off 6000)' = "
          "the implementer's estimate of the same diff.\n"]
    for s in ("2D", "5D", "10D"):
        md.append(f"\n## {s}\n")
        md.append("| n | A score (zeta) | A div/succ | B score (zeta) | B div/succ | B-A diff [CI] wins p | non-div pairs diff [CI] p (k) | mmd2_eval diff p | r5 (off 6000) |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for n in (4, 8, 16, 32):
            A, B = cells.get((s, n, "A_trust_zt")), cells.get((s, n, "B_notrust_zn"))
            if not (A and B):
                continue
            sa, sb = pen(A), pen(B)
            ps = paired_stats(sb, sa)
            keep = [i for i in range(len(sa)) if not A["runs"][i]["diverged"] and not B["runs"][i]["diverged"]]
            pn = paired_stats([sb[i] for i in keep], [sa[i] for i in keep]) if len(keep) > 5 else None
            pm = paired_stats(penm(B), penm(A))
            md.append(f"| {n} | {st.mean(sa):.4f} ({A['protocol']['zeta']:g}) | {A['diverged']}/{succ(A):.0%} | "
                      f"{st.mean(sb):.4f} ({B['protocol']['zeta']:g}) | {B['diverged']}/{succ(B):.0%} | {fmt(ps)} | "
                      f"{(pn['mean_diff'] if pn else float('nan')):+.4f} [{(pn['ci95'][0] if pn else float('nan')):+.3f}, {(pn['ci95'][1] if pn else float('nan')):+.3f}] p={(pn['perm_p'] if pn else float('nan')):.3f} ({len(keep)}) | "
                      f"{pm['mean_diff']:+.4f} p={pm['perm_p']:.3f} | {r5_est(s, n)} |")
        if s == "2D":
            md.append("\n2D zeta sensitivity: A8 = trust @ zeta 8 (basin rule) vs B, and vs A (zeta 16)\n")
            md.append("| n | A8 score | A8 div/succ | B - A8 [CI] p | A - A8 (+ = zeta 8 better) [CI] p |")
            md.append("|---|---|---|---|---|")
            for n in (4, 8, 16, 32):
                A8, A, B = cells.get((s, n, "A8_trust_z8")), cells.get((s, n, "A_trust_zt")), cells.get((s, n, "B_notrust_zn"))
                if not (A8 and A and B):
                    continue
                md.append(f"| {n} | {st.mean(pen(A8)):.4f} | {A8['diverged']}/{succ(A8):.0%} | {fmt(paired_stats(pen(B), pen(A8)))} | {fmt(paired_stats(pen(A), pen(A8)))} |")
    hosts = {(d["verifier"]["host"], d["verifier"]["cpu"]) for d in cells.values()}
    md.append(f"\nnodes: {sorted(hosts)}")
    (HERE / "heldout_r5_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {HERE / 'heldout_r5_tables.md'}")


if __name__ == "__main__":
    main()
