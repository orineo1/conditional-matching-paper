"""Agent 6 -- round-3 held-out confirmation: replay_geo0.7d5_trust vs trust_noise1.

Claim under test (replay/REPORT.md section 4): `replay_geo0.7d5_trust`
(geometric sample replay, decay 0.7, depth 5, subsample, batch_total=n,
composed with trust_noise1) gives MATCHED quality at ~3-4x fewer fresh
conditional calls on 2D/5D/10D, plus a same-calls win at 2D n=32.

OFFSET 2000 -- fully fresh (screening used 0..39; the round-2 verification used
1000..1099, and trust_noise1 was PROMOTED partly on those seeds, so its
offset-1000 scores are selection-tainted for a non-inferiority comparison;
offset 2000 avoids that and lets every arm be re-run in one process).

One SLURM array task per SETTING (3 tasks), running all 8 cells sequentially
in one process on one node so that every pairing (same-n AND cross-n
call-matched) is same-node:
  trust_noise1 at n in {4, 8, 32}    (fresh calls 396 / 792 / 3168)
  replay_geo0.7d5_trust at n in {4, 8, 32}   (fresh calls 99 / 297 / 1089)
  baseline at n in {8, 32}           (792 / 3168)
Decision comparisons (actual fresh-call numbers from the screening, verified):
  same-n:        replay@n vs trust@n (replay has 1/4-1/3 the calls)
  call-matched:  replay@n4 (99) vs trust@n4 (396); replay@n8 (297) vs
                 trust@n4 (396); replay@n32 (1089) vs trust@n8 (792);
                 replay@n32 (1089) vs baseline@n32 (3168)
100 paired restarts; outputs verification/heldout_runs/<cell>_off2000.json.

    cd simulations
    python ../experiments/model-optimization/verification/heldout_r3_cells.py list
    python ../experiments/model-optimization/verification/heldout_r3_cells.py run --index I [--restarts 100] [--offset 2000]
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import heldout_cells as hc                                    # noqa: E402

SETTINGS = ["2D", "5D", "10D"]
NS = [4, 8, 32]


def group_list(settings=SETTINGS):
    groups = []
    for s in settings:
        g = []
        for n in NS:
            g.append((s, n, "no_lgd", "none", "trust_noise1"))
            g.append((s, n, "no_lgd", "none", "replay_geo0.7d5_trust"))
        for n in (8, 32):
            g.append((s, n, "no_lgd", "none", "baseline"))
        groups.append(g)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run"])
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--restarts", type=int, default=100)
    ap.add_argument("--offset", type=int, default=2000)
    ap.add_argument("--settings", nargs="*", default=SETTINGS)
    a = ap.parse_args()
    assert a.offset >= 1000
    groups = group_list(a.settings)
    if a.mode == "list":
        for i, g in enumerate(groups):
            print(i, g[0][0], len(g), "cells")
        print(f"# {len(groups)} groups, {sum(len(g) for g in groups)} cells", file=sys.stderr)
        return
    todo = [groups[a.index]] if a.index is not None else groups
    for g in todo:
        print(f"r3 group {g[0][0]} on {hc.cpu_model()}", flush=True)
        for c in g:
            hc.run_cell(c, a.restarts, a.offset)


if __name__ == "__main__":
    main()
