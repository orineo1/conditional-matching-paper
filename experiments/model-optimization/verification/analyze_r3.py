"""Agent 6 -- round-3 analysis: replay_geo0.7d5_trust vs trust_noise1/baseline.

    cd simulations && /Users/stolk/miniconda3/bin/python ../experiments/model-optimization/verification/analyze_r3.py [--offset 2000]

Same-n AND cross-n call-matched paired comparisons (all cells share restart
seeds and ran in one process per setting), Pareto table per setting, secondary
metrics.  Writes verification/heldout_r3_tables.md.
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
MMD_PEN = 1.0
TRUST, REPLAY, BASE = "trust_noise1", "replay_geo0.7d5_trust", "baseline"
# (candidate cell, comparator cell) as (name, n) pairs; + = candidate better
COMPARISONS = [
    ("same-n (replay 1/3-1/4 calls)", [((REPLAY, n), (TRUST, n)) for n in (4, 8, 32)]),
    ("call-matched-or-fewer (NOTE: the n32-vs-trust@8 pairing gives the candidate 37% MORE calls; see comment)",
     [((REPLAY, 4), (TRUST, 4)),        # 99 vs 396
      ((REPLAY, 8), (TRUST, 4)),        # 297 vs 396
      ((REPLAY, 32), (TRUST, 8)),       # 1089 vs 792 (candidate has MORE calls)
      ((REPLAY, 32), (BASE, 32)),       # 1089 vs 3168
      ((REPLAY, 32), (BASE, 8))]),      # 1089 vs 792
]


def pen(d):
    return [min(q["L2"], PENALTY) if not q["diverged"] else PENALTY for q in d["runs"]]


def penm(d):
    return [min(q["mmd2_eval"], MMD_PEN) if q.get("mmd2_eval") is not None else MMD_PEN for q in d["runs"]]


def succ(d):
    return sum(1 for q in d["runs"] if not q["diverged"] and q["abs_err"] < 0.5) / len(d["runs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=2000)
    a = ap.parse_args()
    cells = {}
    for p in sorted(RUNS.glob(f"*_off{a.offset}.json")):
        d = json.load(open(p))
        cells[(d["setting"], d["candidate"], d["n"])] = d
    if not cells:
        print("no runs at offset", a.offset)
        return
    md = [f"# Round-3 held-out (offset {a.offset}, Agent 6)\n",
          f"{len(cells)} cells; paired diff = comparator - candidate (+ = candidate better); "
          "score = failure-penalised exact GMM L2; calls = mean fresh conditional samples per run.\n"]
    for s in ("2D", "5D", "10D"):
        if not any(k[0] == s for k in cells):
            continue
        md.append(f"\n## {s}\n")
        md.append("| cell | calls | score | success | div | mmd2_eval |")
        md.append("|---|---|---|---|---|---|")
        for (name, n) in [(TRUST, 4), (REPLAY, 4), (TRUST, 8), (REPLAY, 8), (BASE, 8),
                          (TRUST, 32), (REPLAY, 32), (BASE, 32)]:
            d = cells.get((s, name, n))
            if d:
                md.append(f"| {name}@n={n} | {d['cm_samples_mean']:.0f} | {st.mean(pen(d)):.4f} | "
                          f"{succ(d):.0%} | {d['diverged']} | {st.mean(penm(d)):.4f} |")
        for label, prs in COMPARISONS:
            md.append(f"\n**{label}**\n")
            md.append("| candidate (calls) | comparator (calls) | scores comp -> cand | diff L2 | 95% CI | wins | p | diff mmd2_eval | p |")
            md.append("|---|---|---|---|---|---|---|---|---|")
            for (cn, cnn), (bn, bnn) in prs:
                dc, db = cells.get((s, cn, cnn)), cells.get((s, bn, bnn))
                if not dc or not db:
                    continue
                ps = paired_stats(pen(db), pen(dc))
                pm = paired_stats(penm(db), penm(dc))
                md.append(f"| {cn}@n={cnn} ({dc['cm_samples_mean']:.0f}) | {bn}@n={bnn} ({db['cm_samples_mean']:.0f}) | "
                          f"{st.mean(pen(db)):.4f} -> {st.mean(pen(dc)):.4f} | {ps['mean_diff']:+.4f} | "
                          f"[{ps['ci95'][0]:+.3f}, {ps['ci95'][1]:+.3f}] | {ps['wins_for_b']}/{ps['n']} | {ps['perm_p']:.3f} | "
                          f"{pm['mean_diff']:+.4f} | {pm['perm_p']:.3f} |")
    (HERE / "heldout_r3_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {HERE / 'heldout_r3_tables.md'}")


if __name__ == "__main__":
    main()
