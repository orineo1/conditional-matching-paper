"""Round 5 -- per-arm zeta calibration under the corrected protocol (engine path).

Replicates the criterion of ``simulations/experiments/exp5b_zeta_calibration.py``
(2026-08-24 revision) on the canonical 2D/5D/10D settings, THROUGH THE ENGINE
(engine_runner.run_engine):

  * x_T ~ N(0, I) per restart (``x_init="randn"``, generator 0x5EED0000 ^ restart)
  * n = 128 conditional samples ("large n": the calibration is about the
    landscape, not about sampling noise at the compared n)
  * for each zeta on a log grid: reached = fraction of restarts with
    ||x_hat - x*|| < 0.5, div = fraction diverged (non-finite or |x| > 50)
  * accepted  <=>  reached >= 0.8 and div == 0
  * rule "basin" (exp5b, dim(x)=1 construct): zeta* = the SMALLEST accepted
    zeta; if none is accepted, the basin-rate maximiser among div == 0
    (flagged FALLBACK).  In 5D/10D reached == 0 at EVERY zeta (job 45938560),
    so this rule cannot calibrate higher dims.
  * rule "l2min" (AMENDED 2026-08-24, pre-registered in hypotheses/agent4.yaml
    H-R5 before any round-5 cell ran; the DEFAULT): zeta* = the minimiser of
    the failure-penalised mean exact L2 at n = 128 (the paper's metric) over
    the divergence-free zetas (div == 0).  Secondary diagnostic: the
    relative-radius reach rate ||x - x*|| < 0.5 * sqrt(d_x) (needs the
    per-restart records that runs AFTER this amendment store; the 45938560
    JSONs hold only per-zeta aggregates + the median abs_err).
  * the divergence-free zeta range per (setting, arm) is itself reported: the
    no-trust arm's ceiling is what the trust region lifts.

Arms: ``trust``   = no momentum, step_clip="noise", step_tau=1 (trust_noise1)
      ``notrust`` = no momentum, no trust region
(``adam`` optional, for completeness of the per-arm principle).

Check: 2D / trust must reproduce the other session's zeta_none = 8 (88% reached).

    python calibrate_zeta.py run --setting 2D --arm trust [--restarts 40]
    python calibrate_zeta.py report          # -> zeta_star.json + zeta_star.md
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIM = ROOT / "simulations"
for p in (SIM / "src", SIM / "experiments", HERE.parent / "estimator"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ZETA_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
ARMS = {"trust": dict(temporal="none", step_clip="noise", step_tau=1.0),
        "notrust": dict(temporal="none", step_clip="none", step_tau=1.0),
        "adam": dict(temporal="adam", step_clip="noise", step_tau=1.0)}
SETTINGS = ["2D", "5D", "10D"]
TASKS = [(s, a) for s in SETTINGS for a in ("trust", "notrust")]
RUNS = HERE / "calib_runs"


def run_task(setting, arm, restarts, offset, n_calib, grid, floor, tol):
    from _common import penalised_score
    from _guided import evaluate
    from engine_runner import build_models, run_engine
    params, S_G, bw, mc, mu = build_models(setting)
    spec = ARMS[arm]
    rows = []
    for zeta in grid:
        t0 = time.perf_counter()
        runs, xs = [], []
        for r in range(offset, offset + restarts):
            x, info = run_engine(mc, mu, S_G, bw, n_calib, "no_lgd", spec["temporal"], r,
                                 x_init="randn", zeta=zeta, step_clip=spec["step_clip"],
                                 step_tau=spec["step_tau"])
            ev = evaluate(x, params, info)
            runs.append(ev)
            xs.append(ev["abs_err"])
        reached = sum(1 for e in xs if e < tol) / restarts
        div = sum(1 for q in runs if q["diverged"]) / restarts
        d_x = int(params["x_star"].reshape(-1).numel())
        reached_rel = sum(1 for e in xs if e < tol * d_x ** 0.5) / restarts
        row = {"setting": setting, "arm": arm, "zeta": zeta, "n_calib": n_calib, "d_x": d_x,
               "reached_rel_rate": reached_rel,
               "per_restart": [{"restart": offset + i, "abs_err": q["abs_err"], "L2": q["L2"],
                                "x_hat": q["x_hat"], "diverged": q["diverged"]}
                               for i, q in enumerate(runs)],
               "reached_rate": reached, "diverged_rate": div,
               "score": penalised_score(runs), "restarts": restarts, "offset": offset,
               "accepted": bool(reached >= floor and div == 0.0),
               "abs_err_median": st.median(e for e in xs if e == e and e != float("inf")) if any(e != float("inf") for e in xs) else None,
               "seconds_per_run": (time.perf_counter() - t0) / restarts}
        rows.append(row)
        print(f"{setting} {arm:<8} zeta={zeta:<6g} reached={reached:.0%} div={div:.0%} "
              f"score={row['score']:.4f} {'ACCEPTED' if row['accepted'] else ''}", flush=True)
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{setting}_{arm}.json").write_text(json.dumps(rows, indent=1))
    return rows


def select_basin(rows, floor=0.8):
    ok = [r for r in rows if r["accepted"]]
    nodiv = [r for r in rows if r["diverged_rate"] == 0.0]
    zmax = max(nodiv, key=lambda r: (r["reached_rate"], -r["zeta"]))["zeta"] if nodiv else None
    if ok:
        return {"zeta_star": min(r["zeta"] for r in ok), "rule": "basin:smallest_accepted",
                "fallback": False, "zeta_max_rate": zmax}
    return {"zeta_star": zmax, "rule": "basin:rate_maximiser", "fallback": True,
            "zeta_max_rate": zmax}


def select_l2min(rows):
    """AMENDED rule: argmin of the penalised mean exact L2 over divergence-free zetas."""
    nodiv = [r for r in rows if r["diverged_rate"] == 0.0]
    if not nodiv:
        return {"zeta_star": None, "rule": "l2min", "fallback": True}
    best = min(nodiv, key=lambda r: (r["score"], r["zeta"]))
    return {"zeta_star": best["zeta"], "rule": "l2min", "fallback": False,
            "score_at_star": best["score"]}


def report(rule="l2min"):
    out = {}
    lines = ["| setting | arm | d_x | zeta* (l2min) | score@zeta* | zeta (basin rule) | reached@basin | "
             "div-free zeta range | first zeta with divergence | scores by zeta (div-free only) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for f in sorted(RUNS.glob("*.json"), key=lambda p: (int(p.stem.split("D")[0]), p.stem)):
        rows = json.loads(f.read_text())
        s, a = rows[0]["setting"], rows[0]["arm"]
        d_x = rows[0].get("d_x", {"2D": 1, "5D": 4, "10D": 9}[s])
        by = {r["zeta"]: r for r in rows}
        l2, bs = select_l2min(rows), select_basin(rows)
        free = [r["zeta"] for r in rows if r["diverged_rate"] == 0.0]
        first_div = next((r["zeta"] for r in rows if r["diverged_rate"] > 0.0), None)
        sel = l2 if rule == "l2min" else bs
        out.setdefault(s, {})[a] = {**sel, "basin_rule": bs, "l2min_rule": l2, "d_x": d_x,
                                    "div_free_zetas": free, "first_zeta_with_divergence": first_div,
                                    "grid": [(r["zeta"], r["score"], r["reached_rate"], r["diverged_rate"],
                                              r.get("reached_rel_rate")) for r in rows]}
        scores = ", ".join(f"{r['zeta']:g}:{r['score']:.4f}" for r in rows if r["diverged_rate"] == 0.0)
        lines.append(f"| {s} | {a} | {d_x} | **{l2['zeta_star']}** | {l2.get('score_at_star', float('nan')):.4f} | "
                     f"{bs['zeta_star']}{' (fallback)' if bs['fallback'] else ''} | "
                     f"{by[bs['zeta_star']]['reached_rate']:.0%} | {min(free)}..{max(free)} | {first_div} | {scores} |")
    (HERE / "zeta_star.json").write_text(json.dumps(out, indent=1))
    (HERE / "zeta_star.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    chk = out.get("2D", {}).get("trust", {})
    print(f"\n2D/trust: l2min zeta*={chk.get('zeta_star')}, basin rule={chk.get('basin_rule', {}).get('zeta_star')} "
          f"(other session: 8)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run", "report"])
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--setting", default=None)
    ap.add_argument("--arm", default=None, choices=list(ARMS))
    ap.add_argument("--restarts", type=int, default=40)
    ap.add_argument("--offset", type=int, default=5000, help="calibration block; disjoint from 6000 (round-5 cells)")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--zeta-grid", type=float, nargs="*", default=ZETA_GRID)
    ap.add_argument("--success-floor", type=float, default=0.8)
    ap.add_argument("--success-tol", type=float, default=0.5)
    ap.add_argument("--rule", default="l2min", choices=["l2min", "basin"],
                    help="selection rule written to zeta_star.json (default: the amended l2min)")
    a = ap.parse_args()
    if a.mode == "list":
        for i, (s, arm) in enumerate(TASKS):
            print(i, s, arm)
    elif a.mode == "run":
        s, arm = (TASKS[a.index] if a.index is not None else (a.setting, a.arm))
        run_task(s, arm, a.restarts, a.offset, a.n_calib, a.zeta_grid, a.success_floor, a.success_tol)
    else:
        report(a.rule)


if __name__ == "__main__":
    main()
