"""Agent M -- experiment M-11: M-10 winners (fifo16 / cohort16 + trust) vs
fresh-only at EQUAL fresh cost under the CORRECTED protocol.
Pre-registered in hypotheses/agentM.yaml (M-11) BEFORE any run.

Corrected protocol (engine_runner protocol args): x_init=randn, zeta =
zeta*_trust per setting (protocol/zeta_star.json: 16 / 8 / 4 for 2D / 5D /
10D), step_clip=noise, step_tau=1 -- applied identically to EVERY arm.

    python experiments/model-optimization/replay/cells_m11.py list
    python experiments/model-optimization/replay/cells_m11.py run --index I [--restarts 100] [--offset 8000]
    python experiments/model-optimization/replay/cells_m11.py report

One array task = one (setting, f) group, arms in ONE process:
    A  baseline        n = f   fresh f (fresh-only + trust @ zeta*)
    B  replay_fifo16   n = f   fresh f, batch 16 uniform FIFO
    C  replay_cohort16 n = f   fresh f, batch 16 capped cohorts
    D  baseline        n = 8   ceiling reference, shared per setting
Fresh parity A == B == C = f*99 (D = 8*99) asserted at run time.
Results -> runs_m11/<s>_f<f>_<arm>.json (D -> <s>_D8.json).
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
RUNS = HERE / "runs_m11"
ZETA_JSON = HERE.parent / "protocol" / "zeta_star.json"

SETTINGS = ["2D", "5D", "10D"]
BUDGETS = [2, 4]
D_N = 8
ARMS = [("A", "baseline"), ("B", "replay_fifo16"), ("C", "replay_cohort16")]


def zeta_trust(s):
    z = json.loads(ZETA_JSON.read_text())[s]["trust"]["zeta_star"]
    if z is None:
        raise SystemExit(f"no calibrated zeta*_trust for {s}")
    return float(z)


def groups():
    return [(s, f) for s in SETTINGS for f in BUDGETS]


def _run(s, n, cand, restarts, offset):
    from engine_runner import cell
    return cell(s, n, "no_lgd", "none", cand, restarts, offset, x_init="randn",
                zeta=zeta_trust(s), step_clip="noise", step_tau=1.0)


def run_group(s, f, restarts, offset):
    RUNS.mkdir(exist_ok=True)
    jobs = [(f"{s}_f{f}_{arm}", cand, f, f * 99) for arm, cand in ARMS]
    jobs.append((f"{s}_D8", "baseline", D_N, D_N * 99))
    for name, cand, n, expect in jobs:
        out = RUNS / (name + ".json")
        if out.exists():
            print("skip", out.name, flush=True)
            continue
        summ = _run(s, n, cand, restarts, offset)
        assert summ["cm_samples_mean"] == expect, \
            f"{name}: cm_samples {summ['cm_samples_mean']} != {expect}"
        assert summ["protocol"]["zeta"] == zeta_trust(s) and summ["protocol"]["step_clip"] == "noise" \
            and summ["protocol"]["x_init"] == "randn", summ["protocol"]
        out.write_text(json.dumps(summ, indent=1, default=str))


def _load(name):
    p = RUNS / (name + ".json")
    return json.loads(p.read_text()) if p.exists() else None


def report():
    from _common import paired_stats
    lines = ["# M-11: fifo16 / cohort16 vs fresh-only at equal fresh cost, CORRECTED "
             "protocol (x_T~N(0,I), zeta*_trust, trust tau=1; R=100, offset 8000)\n",
             "diff = paired A - candidate (**+ = replay helps**), two-sided permutation p, "
             "bootstrap 95% CI.  D = fresh-only n=8 ceiling (2-4x the cost).\n",
             "| setting | zeta* | f | A fresh-only | B fifo16 | B-A [CI] p | C cohort16 | C-A [CI] p | D n=8 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in SETTINGS:
        d = _load(f"{s}_D8")
        for f in BUDGETS:
            a = _load(f"{s}_f{f}_A")
            row = [s, f"{zeta_trust(s):g}", str(f)]
            if a is None:
                lines.append("| " + " | ".join(row + ["(missing)"] + [""] * 5) + " |")
                continue
            row.append(f"{a['score']:.4f}")
            for arm in ("B", "C"):
                c = _load(f"{s}_f{f}_{arm}")
                if c is None:
                    row += ["(missing)", ""]
                    continue
                st = paired_stats(a["scores"], c["scores"])
                sig = "**" if st["perm_p"] <= 0.05 else ""
                row.append(f"{c['score']:.4f}")
                row.append(f"{sig}{st['mean_diff']:+.4f}{sig} "
                           f"[{st['ci95'][0]:+.3f}, {st['ci95'][1]:+.3f}] p={st['perm_p']:.4f}")
            row.append(f"{d['score']:.4f}" if d else "(missing)")
            lines.append("| " + " | ".join(row) + " |")
    out = HERE / "m11_tables.md"
    out.write_text("\n".join(lines) + "\n")
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run", "report"])
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--restarts", type=int, default=100)
    ap.add_argument("--offset", type=int, default=8000)
    a = ap.parse_args()
    gs = groups()
    if a.mode == "list":
        for i, (s, f) in enumerate(gs):
            print(i, f"{s}_f{f}  zeta*={zeta_trust(s):g}")
        print(f"# {len(gs)} groups x 3 arms + shared D per setting", file=sys.stderr)
    elif a.mode == "run":
        todo = [gs[a.index]] if a.index is not None else gs
        for s, f in todo:
            run_group(s, f, a.restarts, a.offset)
    else:
        report()


if __name__ == "__main__":
    main()
