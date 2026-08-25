"""Agent M -- experiment M-10: progressive temporally-weighted replay --
buffer POLICY comparison at equal fresh cost.  Pre-registered in
hypotheses/agentM.yaml (M-10) BEFORE any run.

    python experiments/model-optimization/replay/cells_m10.py list
    python experiments/model-optimization/replay/cells_m10.py run --index I [--restarts 100] [--offset 4000]
    python experiments/model-optimization/replay/cells_m10.py report

Policies at fresh f in {2, 4} (ALL with trust_noise1):
    geo       geometric decay 0.7 depth 5, batch 8  -- REUSED from M-9 arm B
              (runs_m9/<s>_f<f>_B.json), NOT rerun
    fifo<B>   uniform FIFO to batch B in {8, 16}
    cohort<B> capped-cohort thinning to batch B (cohort k keeps
              min(f, ceil(B/2^k)) rows)
Comparators (REUSED from runs_m9): trust_noise1@f (arm A, equal fresh cost,
PRIMARY), trust_noise1@8 (D, ceiling, descriptive).  Registered degeneracy:
cohort8 at f=4 has fifo8's exact plan [4,4] -> not run; the report reads the
fifo8 cell for both labels.

Seeds: offset 4000, R=100 -- the SAME seeds as M-9, which is what makes the
reuse paired; this deviates from the offset-0/R=40 screening default and is
registered as such in M-10.  6 array tasks = setting x f; each task's cells
run in one process.  Results -> runs_m10/<s>_f<f>_<policy>.json.
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
RUNS = HERE / "runs_m10"
M9 = HERE / "runs_m9"

SETTINGS = ["2D", "5D", "10D"]
BUDGETS = [2, 4]
POLICIES = ["fifo8", "fifo16", "cohort8", "cohort16"]


def new_policies(f):
    return [p for p in POLICIES if not (p == "cohort8" and f == 4)]


def groups():
    return [(s, f) for s in SETTINGS for f in BUDGETS]


def run_group(s, f, restarts, offset):
    from engine_runner import cell
    RUNS.mkdir(exist_ok=True)
    for pol in new_policies(f):
        out = RUNS / f"{s}_f{f}_{pol}.json"
        if out.exists():
            print("skip", out.name, flush=True)
            continue
        summ = cell(s, f, "no_lgd", "none", f"replay_{pol}_trust", restarts, offset)
        assert summ["cm_samples_mean"] == f * 99, \
            f"{pol}: cm_samples {summ['cm_samples_mean']} != {f * 99}"
        out.write_text(json.dumps(summ, indent=1, default=str))


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def cand_cell(s, f, pol):
    if pol == "geo":
        return _load(M9 / f"{s}_f{f}_B.json")                 # reused M-9 arm B
    if pol == "cohort8" and f == 4:
        return _load(RUNS / f"{s}_f{f}_fifo8.json")           # registered degeneracy
    return _load(RUNS / f"{s}_f{f}_{pol}.json")


def report():
    from _common import paired_stats
    lines = ["# M-10: buffer-policy comparison at equal fresh cost, all + trust "
             "(R=100, offset 4000; geo / trust@f / trust@8 reused from M-9)\n",
             "diff = paired comparator - candidate (+ = candidate better), "
             "two-sided permutation p.  PRIMARY comparator: trust_noise1@f "
             "(equal fresh cost).  SECONDARY: the M-9 geometric arm (policy "
             "effect proper).  cohort8@f4 = fifo8@f4 (identical plan, one run).\n"]
    for s in SETTINGS:
        d = _load(M9 / f"{s}_D8.json")
        lines.append(f"\n## {s} (ceiling trust@8 = "
                     f"{d['score']:.4f})\n" if d else f"\n## {s}\n")
        lines.append("| f | policy | score | vs trust@f diff [CI] p | vs geo diff [CI] p |")
        lines.append("|---|---|---|---|---|")
        for f in BUDGETS:
            a = _load(M9 / f"{s}_f{f}_A.json")
            geo = cand_cell(s, f, "geo")
            for pol in ["geo"] + POLICIES:
                c = cand_cell(s, f, pol)
                if c is None:
                    lines.append(f"| {f} | {pol} | (missing) | | |")
                    continue
                row = [str(f), pol + (" (=fifo8)" if pol == "cohort8" and f == 4 else ""),
                       f"{c['score']:.4f}"]
                for comp in (a, geo):
                    if comp is None or comp is c:
                        row.append("-")
                        continue
                    st = paired_stats(comp["scores"], c["scores"])
                    sig = "**" if st["perm_p"] <= 0.05 else ""
                    row.append(f"{sig}{st['mean_diff']:+.4f}{sig} "
                               f"[{st['ci95'][0]:+.3f}, {st['ci95'][1]:+.3f}] "
                               f"p={st['perm_p']:.4f}")
                lines.append("| " + " | ".join(row) + " |")
    out = HERE / "m10_tables.md"
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
        print(f"# {len(gs)} groups; new cells: "
              f"{sum(len(new_policies(f)) for _, f in gs)}", file=sys.stderr)
    elif a.mode == "run":
        todo = [gs[a.index]] if a.index is not None else gs
        for s, f in todo:
            run_group(s, f, a.restarts, a.offset)
    else:
        report()


if __name__ == "__main__":
    main()
