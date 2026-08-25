"""Agent M -- experiment M-8: recycling vs nothing at equal fresh cost.

Pre-registered in hypotheses/agentM.yaml (M-8) BEFORE any run.  Two-sided
question: at the same fresh conditional-call budget f per step, does adding
30% recycled one-step-old samples to the MMD batch (Ori's reuse_frac=0.3)
help, do nothing, or hurt, vs using the f fresh samples alone?

    python experiments/model-optimization/replay/cells_m8.py list
    python experiments/model-optimization/replay/cells_m8.py run --index I [--restarts 100] [--offset 3000]
    python experiments/model-optimization/replay/cells_m8.py report

One array task = one (setting, fresh budget f) group; its FOUR arms run
sequentially in the same process (same node, same restart seeds -> clean
pairing):

    A        baseline      n = f          (batch f, fresh f)
    B        replay30      n = round(f/0.7) in {10, 20, 40}  (batch n, fresh f)
    A_trust  trust_noise1  n = f
    B_trust  replay30_trust n as B

replay_counts(10|20|40, 3/7, 1) = [7,3] | [14,6] | [28,12], so fresh counts
are EXACTLY f = 7|14|28 in both arms of each pair; asserted at run time from
cm_samples.  Results -> runs_m8/<setting>_f<f>_<arm>.json.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIM = ROOT / "simulations"
for p in (SIM / "experiments", SIM / "src", ROOT / "experiments" / "model-optimization" / "estimator"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
RUNS = HERE / "runs_m8"

SETTINGS = ["2D", "5D", "10D"]
BUDGETS = [7, 14, 28]                       # fresh conditional samples per step
BATCH = {7: 10, 14: 20, 28: 40}             # replay30 batch giving exactly f fresh
ARMS = [                                    # (arm name, candidate, n for budget f)
    ("A", "baseline", lambda f: f),
    ("B", "replay30", lambda f: BATCH[f]),
    ("A_trust", "trust_noise1", lambda f: f),
    ("B_trust", "replay30_trust", lambda f: BATCH[f]),
]


def groups():
    return [(s, f) for s in SETTINGS for f in BUDGETS]


def arm_name(s, f, arm):
    return f"{s}_f{f}_{arm}"


def run_group(s, f, restarts, offset):
    """All four arms of one (setting, budget) in-process, checkpoints loaded once."""
    from tfg.replay import replay_counts
    assert replay_counts(BATCH[f], 0.3 / 0.7, 1)[0] == f, (f, BATCH[f])
    from engine_runner import cell
    RUNS.mkdir(exist_ok=True)
    for arm, cand, n_of in ARMS:
        out = RUNS / (arm_name(s, f, arm) + ".json")
        if out.exists():
            print("skip", out.name, flush=True)
            continue
        summ = cell(s, n_of(f), "no_lgd", "none", cand, restarts, offset)
        # equal-fresh-cost invariant: every arm must spend exactly f fresh
        # conditional samples per step (T = 99 steps)
        expect = f * 99
        assert summ["cm_samples_mean"] == expect, \
            f"{arm}: cm_samples {summ['cm_samples_mean']} != {expect}"
        out.write_text(json.dumps(summ, indent=1, default=str))


def _load(name):
    p = RUNS / (name + ".json")
    return json.loads(p.read_text()) if p.exists() else None


def report():
    from _common import paired_stats
    lines = ["# M-8: recycling vs nothing at equal fresh cost (100 restarts, offset 3000)\n",
             "diff = paired (fresh-only) - (fresh+recycled); **+ = recycling helps**, "
             "- = recycling hurts; two-sided permutation p, bootstrap 95% CI. "
             "Equal fresh calls within every pair (asserted at run time).\n"]
    for pair, a_arm, b_arm in (("no trust: B vs A", "A", "B"),
                               ("with trust: B_trust vs A_trust", "A_trust", "B_trust")):
        lines.append(f"\n## {pair}\n")
        lines.append("| setting | f | fresh-only score | +recycled score | diff | 95% CI | p |")
        lines.append("|---|---|---|---|---|---|---|")
        for s in SETTINGS:
            for f in BUDGETS:
                a, b = _load(arm_name(s, f, a_arm)), _load(arm_name(s, f, b_arm))
                if a is None or b is None:
                    lines.append(f"| {s} | {f} | (missing) | | | | |")
                    continue
                stt = paired_stats(a["scores"], b["scores"])
                sig = "**" if stt["perm_p"] <= 0.05 else ""
                lines.append(f"| {s} | {f} | {a['score']:.4f} | {b['score']:.4f} | "
                             f"{sig}{stt['mean_diff']:+.4f}{sig} | "
                             f"[{stt['ci95'][0]:+.3f}, {stt['ci95'][1]:+.3f}] | "
                             f"{stt['perm_p']:.4f} |")
    out = HERE / "m8_tables.md"
    out.write_text("\n".join(lines) + "\n")
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run", "report"])
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--restarts", type=int, default=100)
    ap.add_argument("--offset", type=int, default=3000)
    a = ap.parse_args()
    gs = groups()
    if a.mode == "list":
        for i, (s, f) in enumerate(gs):
            print(i, f"{s}_f{f}")
        print(f"# {len(gs)} groups x 4 arms", file=sys.stderr)
    elif a.mode == "run":
        todo = [gs[a.index]] if a.index is not None else gs
        for s, f in todo:
            run_group(s, f, a.restarts, a.offset)
    else:
        report()


if __name__ == "__main__":
    main()
