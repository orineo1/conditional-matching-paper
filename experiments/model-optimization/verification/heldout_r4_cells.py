"""Agent 6 -- round-4 held-out confirmation: progressive FIFO / cohort replay
(M-10) vs fresh-only trust_noise1 at EQUAL fresh cost.

Claim under test (replay/m10_tables.md, hypotheses M-10): replay_fifo16_trust
(f fresh + up to 14 recycled rows from the last 7 steps, gradient through the
fresh rows only, + trust_noise1) beats trust_noise1@f at f=2 in 2D (+0.089,
p=0.007) and 10D (+0.084, p=0.0005), null in 5D, and at f=4 in 10D (+0.082);
replay_cohort16_trust similar but weaker.  The M-10 numbers are a selection
among 10 policy x f arms on the M-9 seeds (offset 4000), whose comparators
were reused from a DIFFERENT job -> fresh-seed, same-process confirmation.

OFFSET 5000 (never used), R=100.  One array task per SETTING (3 tasks); the 7
cells of a setting run sequentially in ONE process on one node so every
pairing (same-f, cross-f, and vs the trust@8 reference) is same-node:
  trust_noise1@2, replay_fifo16_trust@2, replay_cohort16_trust@2,
  trust_noise1@4, replay_fifo16_trust@4, replay_cohort16_trust@4,
  trust_noise1@8 (reference / ceiling).
Fresh calls: f*99 for every arm at f (verified: 198 / 396; 792 for trust@8).

    cd simulations
    python ../experiments/model-optimization/verification/heldout_r4_cells.py list
    python ../experiments/model-optimization/verification/heldout_r4_cells.py run --index I [--restarts 100] [--offset 5000]
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import heldout_cells as hc                                    # noqa: E402

SETTINGS = ["2D", "5D", "10D"]
FS = [2, 4]
ARMS = ["trust_noise1", "replay_fifo16_trust", "replay_cohort16_trust"]


def group_list(settings=SETTINGS):
    groups = []
    for s in settings:
        g = [(s, f, "no_lgd", "none", a) for f in FS for a in ARMS]
        g.append((s, 8, "no_lgd", "none", "trust_noise1"))
        groups.append(g)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run"])
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--restarts", type=int, default=100)
    ap.add_argument("--offset", type=int, default=5000)
    ap.add_argument("--settings", nargs="*", default=SETTINGS)
    a = ap.parse_args()
    assert a.offset >= 5000, "round-4 held-out must use a never-used offset (>= 5000)"
    groups = group_list(a.settings)
    if a.mode == "list":
        for i, g in enumerate(groups):
            print(i, g[0][0], len(g), "cells")
        print(f"# {len(groups)} groups, {sum(len(g) for g in groups)} cells", file=sys.stderr)
        return
    todo = [groups[a.index]] if a.index is not None else groups
    for g in todo:
        print(f"r4 group {g[0][0]} on {hc.cpu_model()}", flush=True)
        for c in g:
            p = hc.run_cell(c, a.restarts, a.offset)
            import json
            d = json.load(open(p))
            assert abs(d["cm_samples_mean"] - c[1] * 99) < 1e-6, (c, d["cm_samples_mean"])


if __name__ == "__main__":
    main()
