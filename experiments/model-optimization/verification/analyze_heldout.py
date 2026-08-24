"""Agent 6 -- analysis of the held-out runs (Phase 2).

    cd simulations && /Users/stolk/miniconda3/bin/python ../experiments/model-optimization/verification/analyze_heldout.py [--offset 1000]

Reads verification/heldout_runs/*_off<offset>.json, pairs every candidate with
its baseline cell (same setting, n, arm; same restart indices = same tape
seeds), and writes verification/heldout_tables.md and heldout_rows.csv with:
  * paired diff of the failure-penalised L2 (base - cand, + = candidate better),
    bootstrap 95% CI, paired permutation p, wins (``_common.paired_stats``);
  * the same for the independent end-point metric mmd2_eval (penalised: diverged -> max);
  * success-rate and divergence deltas, calls (must match), wall;
  * the screening estimate (40 restarts, offset 0) next to the held-out one, so
    the shrinkage of selected effects is visible;
  * a Pareto table per setting: score vs conditional calls for the no_lgd/none
    baseline at n in {4,8,16,32,64,96}, lgd/none at n in {8,32}, and every
    candidate at n in {4,8,16,32}.
"""
import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MO = HERE.parent
ROOT = MO.parents[1]
sys.path.insert(0, str(ROOT / "simulations" / "experiments"))
sys.path.insert(0, str(ROOT / "simulations" / "src"))
from _common import PENALTY, paired_stats  # noqa: E402

RUNS = HERE / "heldout_runs"
SCREEN = MO / "estimator" / "runs"
MMD_PEN = 1.0      # penalty for diverged runs in the mmd2_eval metric (cap)


def load_all(offset):
    cells = {}
    for p in sorted(RUNS.glob(f"*_off{offset}.json")):
        d = json.load(open(p))
        key = (d["setting"], d["n"], d["spatial"], d["temporal"], d["candidate"])
        cells[key] = d
    return cells


def pen_scores(d):
    return [min(q["L2"], PENALTY) if not q["diverged"] else PENALTY for q in d["runs"]]


def pen_mmd(d):
    return [min(q["mmd2_eval"], MMD_PEN) if (q.get("mmd2_eval") is not None) else MMD_PEN for q in d["runs"]]


def succ(d):
    return sum(1 for q in d["runs"] if not q["diverged"] and q["abs_err"] < 0.5) / len(d["runs"])


def screening_score(key):
    s, n, sp, tp, c = key
    p = SCREEN / f"{s}_n{n}_{sp}_{tp}_{c}_tape.json"
    if not p.exists():
        return None
    return json.load(open(p))


def fmt(x, f="{:+.3f}"):
    return "" if x is None else f.format(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=1000)
    a = ap.parse_args()
    cells = load_all(a.offset)
    if not cells:
        print("no held-out runs found in", RUNS)
        return
    md, rows = [], []
    md.append(f"# Held-out confirmation (offset {a.offset}, Agent 6)\n")
    md.append(f"{len(cells)} cells found. Paired diff = base - cand (+ = candidate better); "
              "score = failure-penalised mean exact GMM L2 (cap 2.0); mmd2_eval = MMD^2 of 256 fresh "
              "conditional samples at x_hat vs S_G (cap 1.0 on divergence); screening = the Agent-4 "
              "40-restart offset-0 estimate of the same paired diff.\n")
    settings = sorted({k[0] for k in cells}, key=lambda s: int(s[:-1]))
    for s in settings:
        for (sp, tp) in (("no_lgd", "none"), ("lgd", "none")):
            ns = sorted({k[1] for k in cells if k[0] == s and k[2] == sp and k[3] == tp})
            for n in ns:
                base = cells.get((s, n, sp, tp, "baseline"))
                if base is None:
                    continue
                bs, bm = pen_scores(base), pen_mmd(base)
                md.append(f"\n### {s}, n={n}, {sp}/{tp}  (baseline score {st.mean(bs):.4f}, success {succ(base):.0%}, "
                          f"div {base['diverged']}, calls {base['cm_samples_mean']:.0f}, mmd2_eval {st.mean(bm):.4f}, "
                          f"{base['seconds_per_run']:.2f}s/run, R={len(bs)})\n")
                md.append("| candidate | score | success | div | calls | s/run | diff L2 | 95% CI | wins | p | "
                          "diff mmd2_eval | 95% CI | p | screening diff (R=40) |")
                md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
                cands = sorted(k[4] for k in cells if k[:4] == (s, n, sp, tp) and k[4] != "baseline")
                for c in cands:
                    d = cells[(s, n, sp, tp, c)]
                    cs, cm = pen_scores(d), pen_mmd(d)
                    if len(cs) != len(bs):
                        md.append(f"| {c} | R mismatch ({len(cs)} vs {len(bs)}) |")
                        continue
                    ps = paired_stats(bs, cs)
                    pm = paired_stats(bm, cm)
                    scr_b, scr_c = screening_score((s, n, sp, tp, "baseline")), screening_score((s, n, sp, tp, c))
                    scr = None
                    if scr_b and scr_c:
                        scr = st.mean(scr_b["scores"]) - st.mean(scr_c["scores"])
                    calls_ok = abs(d["cm_samples_mean"] - base["cm_samples_mean"]) < 1e-6
                    md.append(f"| {c} | {st.mean(cs):.4f} | {succ(d):.0%} | {d['diverged']} | {d['cm_samples_mean']:.0f}"
                              f"{'' if calls_ok else ' (!=base)'} | {d['seconds_per_run']:.2f} | "
                              f"{ps['mean_diff']:+.4f} | [{ps['ci95'][0]:+.3f}, {ps['ci95'][1]:+.3f}] | "
                              f"{ps['wins_for_b']}/{ps['n']} | {ps['perm_p']:.3f} | "
                              f"{pm['mean_diff']:+.4f} | [{pm['ci95'][0]:+.3f}, {pm['ci95'][1]:+.3f}] | {pm['perm_p']:.3f} | "
                              f"{fmt(scr)} |")
                    rows.append({"setting": s, "n": n, "spatial": sp, "temporal": tp, "candidate": c,
                                 "restarts": len(cs), "offset": a.offset,
                                 "base_score": st.mean(bs), "cand_score": st.mean(cs),
                                 "diff": ps["mean_diff"], "ci_lo": ps["ci95"][0], "ci_hi": ps["ci95"][1],
                                 "wins": ps["wins_for_b"], "perm_p": ps["perm_p"],
                                 "base_success": succ(base), "cand_success": succ(d),
                                 "base_div": base["diverged"], "cand_div": d["diverged"],
                                 "calls": d["cm_samples_mean"], "calls_match": calls_ok,
                                 "mmd_diff": pm["mean_diff"], "mmd_ci_lo": pm["ci95"][0], "mmd_ci_hi": pm["ci95"][1],
                                 "mmd_p": pm["perm_p"], "screening_diff": scr,
                                 "s_per_run": d["seconds_per_run"]})
        # Pareto table
        md.append(f"\n### {s}: Pareto (score vs conditional calls), held-out\n")
        md.append("| arm | candidate | n | calls | score | success | div | s/run |")
        md.append("|---|---|---|---|---|---|---|---|")
        pts = []
        for k, d in cells.items():
            if k[0] != s:
                continue
            pts.append((d["cm_samples_mean"], st.mean(pen_scores(d)), k, d))
        for calls, sc, k, d in sorted(pts):
            md.append(f"| {k[2]}/{k[3]} | {k[4]} | {k[1]} | {calls:.0f} | {sc:.4f} | {succ(d):.0%} | {d['diverged']} | {d['seconds_per_run']:.2f} |")
        # frontier
        front, best = [], float("inf")
        for calls, sc, k, d in sorted(pts):
            if sc < best:
                best = sc
                front.append(f"{k[4]}@n={k[1]} ({k[2]}/{k[3]}, {calls:.0f} calls, {sc:.3f})")
        md.append("\nFrontier (non-dominated, increasing calls): " + "; ".join(front))
    (HERE / "heldout_tables.md").write_text("\n".join(md) + "\n")
    if rows:
        with open(HERE / "heldout_rows.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("\n".join(md))
    print(f"\nwrote {HERE / 'heldout_tables.md'} and heldout_rows.csv")


if __name__ == "__main__":
    main()
