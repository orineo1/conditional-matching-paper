"""Compact candidate-vs-baseline matrices from runs/*.json (paired mean diff with
a fast permutation p; the authoritative CIs are in screening_tables.md).

    python experiments/model-optimization/estimator/matrix.py [--arm no_lgd none] [--candidates ...]
"""
import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load():
    cells = {}
    for p in sorted((HERE / "runs").glob("*.json")):
        d = json.loads(p.read_text())
        d["_rng"] = p.stem.split("_")[-1]
        cells[p.stem] = d
    return cells


def quick_p(a, b, P=4000, seed=0):
    diff = [x - y for x, y in zip(a, b)]
    m = st.mean(diff)
    random.seed(seed)
    cnt = sum(1 for _ in range(P)
              if abs(st.mean([d if random.random() < .5 else -d for d in diff])) >= abs(m))
    return m, (cnt + 1) / (P + 1)


def matrix(cells, spatial, temporal, candidates, settings=("2D", "5D", "10D"), ns=(4, 8, 16, 32),
           base_cand="baseline", base_arm=None):
    cols = [(s, n) for s in settings for n in ns]
    head = "| candidate | " + " | ".join(f"{s} n={n}" for s, n in cols) + " |"
    lines = [head, "|---|" + "---|" * len(cols)]
    base_sp, base_tp = base_arm or (spatial, temporal)
    brow = "| baseline score | "
    for s, n in cols:
        b = cells.get(f"{s}_n{n}_{base_sp}_{base_tp}_{base_cand}_tape")
        brow += (f"{b['score']:.3f} ({b['cm_samples_mean']:.0f}) | " if b else "- | ")
    lines.append(brow)
    for c in candidates:
        row = f"| {c} | "
        for s, n in cols:
            b = cells.get(f"{s}_n{n}_{base_sp}_{base_tp}_{base_cand}_tape")
            d = cells.get(f"{s}_n{n}_{spatial}_{temporal}_{c}_tape")
            if b is None or d is None:
                row += "- | "
                continue
            m, p = quick_p(b["scores"], d["scores"])
            mark = "**" if p <= 0.05 else ""
            row += f"{mark}{m:+.3f}{mark} ({d['score']:.3f}) | "
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs=2, default=["no_lgd", "none"])
    ap.add_argument("--candidates", nargs="*", default=None)
    a = ap.parse_args()
    cells = load()
    sp, tp = a.arm
    cands = a.candidates or sorted({d["candidate"] for d in cells.values()
                                    if d["spatial"] == sp and d["temporal"] == tp} - {"baseline"})
    print(matrix(cells, sp, tp, cands))
