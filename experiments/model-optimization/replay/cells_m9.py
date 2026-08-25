"""Agent M -- experiment M-9: replay as direction information at TINY fresh
budgets.  Pre-registered in hypotheses/agentM.yaml (M-9) BEFORE any run.

Two-sided: at fresh budgets f in {1, 2, 4} (all arms WITH trust_noise1), does
topping the MMD batch up with recycled samples beat fresh-only at equal fresh
cost -- where the trust region cannot fix a badly-estimated direction?

    python experiments/model-optimization/replay/cells_m9.py list
    python experiments/model-optimization/replay/cells_m9.py run --index I [--restarts 100] [--offset 4000]
    python experiments/model-optimization/replay/cells_m9.py report

One array task = one (setting, f) group, its arms in ONE process (same node,
same restart seeds):

    A  trust_noise1               n = f      fresh f, batch f
    B  replay_fill8_geo0.7d5_trust n = f     fresh f, batch topped up to 8
                                             (realised max 6 at f=1 -- the
                                             registered structural cap)
    C  replay30_trust             n = 2|3|6  fresh f, proportional recycle
                                             ([1,1]/[2,1]/[4,2]; 50% at f=1)
    D  trust_noise1               n = 8      quality-ceiling REFERENCE,
                                             shared per setting, 2-8x the
                                             fresh cost, outside the decision
                                             rule

Fresh-call parity (A == B == C = f*99, D = 8*99) is asserted at run time.
Results -> runs_m9/<setting>_f<f>_<arm>.json (D -> <setting>_D8.json).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIM = ROOT / "simulations"
for p in (SIM / "experiments", SIM / "src", ROOT / "experiments" / "model-optimization" / "estimator"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
RUNS = HERE / "runs_m9"

SETTINGS = ["2D", "5D", "10D"]
BUDGETS = [1, 2, 4]
CBATCH = {1: 2, 2: 3, 4: 6}         # replay30 batch giving fresh exactly f
D_N = 8


def groups():
    return [(s, f) for s in SETTINGS for f in BUDGETS]


def run_group(s, f, restarts, offset):
    from tfg.replay import replay_counts
    assert replay_counts(CBATCH[f], 0.3 / 0.7, 1)[0] == f, (f, CBATCH[f])
    from engine_runner import cell
    RUNS.mkdir(exist_ok=True)
    arms = [
        (f"{s}_f{f}_A", "trust_noise1", f, f * 99),
        (f"{s}_f{f}_B", "replay_fill8_geo0.7d5_trust", f, f * 99),
        (f"{s}_f{f}_C", "replay30_trust", CBATCH[f], f * 99),
        (f"{s}_D8", "trust_noise1", D_N, D_N * 99),      # shared reference
    ]
    for name, cand, n, expect in arms:
        out = RUNS / (name + ".json")
        if out.exists():
            print("skip", out.name, flush=True)
            continue
        summ = cell(s, n, "no_lgd", "none", cand, restarts, offset)
        assert summ["cm_samples_mean"] == expect, \
            f"{name}: cm_samples {summ['cm_samples_mean']} != {expect}"
        out.write_text(json.dumps(summ, indent=1, default=str))


def _load(name):
    p = RUNS / (name + ".json")
    return json.loads(p.read_text()) if p.exists() else None


def report():
    from _common import paired_stats
    lines = ["# M-9: replay at tiny fresh budgets, all arms + trust_noise1 "
             "(100 restarts, offset 4000)\n",
             "diff = paired (fresh-only A) - (candidate); **+ = recycling "
             "helps**, - = hurts; two-sided permutation p, bootstrap 95% CI. "
             "Equal fresh calls A == B == C; D (fresh n=8) is a reference at "
             "2-8x the cost.\n",
             "| setting | f | A fresh-only | B top-up8 | B-A diff [CI] p | "
             "C prop. | C-A diff [CI] p | D ref n=8 |",
             "|---|---|---|---|---|---|---|---|"]
    for s in SETTINGS:
        d = _load(f"{s}_D8")
        for f in BUDGETS:
            a = _load(f"{s}_f{f}_A")
            row = [s, str(f)]
            if a is None:
                lines.append("| " + " | ".join(row + ["(missing)"] + [""] * 5) + " |")
                continue
            row.append(f"{a['score']:.4f}")
            for arm in ("B", "C"):
                b = _load(f"{s}_f{f}_{arm}")
                if b is None:
                    row += ["(missing)", ""]
                    continue
                stt = paired_stats(a["scores"], b["scores"])
                sig = "**" if stt["perm_p"] <= 0.05 else ""
                row.append(f"{b['score']:.4f}")
                row.append(f"{sig}{stt['mean_diff']:+.4f}{sig} "
                           f"[{stt['ci95'][0]:+.3f}, {stt['ci95'][1]:+.3f}] "
                           f"p={stt['perm_p']:.4f}")
            row.append(f"{d['score']:.4f}" if d else "(missing)")
            lines.append("| " + " | ".join(row) + " |")
    out = HERE / "m9_tables.md"
    out.write_text("\n".join(lines) + "\n")
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run", "report"])
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--restarts", type=int, default=100)
    ap.add_argument("--offset", type=int, default=4000)
    a = ap.parse_args()
    gs = groups()
    if a.mode == "list":
        for i, (s, f) in enumerate(gs):
            print(i, f"{s}_f{f}")
        print(f"# {len(gs)} groups x 4 arms (D shared per setting)", file=sys.stderr)
    elif a.mode == "run":
        todo = [gs[a.index]] if a.index is not None else gs
        for s, f in todo:
            run_group(s, f, a.restarts, a.offset)
    else:
        report()


if __name__ == "__main__":
    main()
