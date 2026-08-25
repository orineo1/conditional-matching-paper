"""Round 5 -- the fair trust-region test under the corrected protocol.

Pre-registered as H-R5 in hypotheses/agent4.yaml. Protocol: x_T ~ N(0,I)
(per-restart generator), per-arm calibrated zeta from protocol/zeta_star.json
(calibrate_zeta.py, n=128 basin-of-attraction rule), no momentum, no LGD,
R = 100 paired restarts at the fresh offset 6000, n in {4, 8, 16, 32},
settings 2D/5D/10D. Arms:

  A  trust_noise1 @ zeta*_trust      (step_clip="noise", tau=1)
  B  no trust     @ zeta*_notrust    (the calibrated baseline)
  C  no trust     @ zeta*_trust      (what the cap buys at the SAME scale)

Single pre-specified primary comparison: A vs B (paired penalised L2).
Secondary: A vs C. One cell per array task.

    python cells_r5.py list
    python cells_r5.py run --index I
    python cells_r5.py report        # -> r5_tables.md, r5_rows.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIM = ROOT / "simulations"
for p in (SIM / "src", SIM / "experiments", HERE.parent / "estimator"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SETTINGS, NS = ["2D", "5D", "10D"], [4, 8, 16, 32]
ARMS = {"A_trust_zt": dict(step_clip="noise", zeta_key="trust"),
        "B_notrust_zn": dict(step_clip="none", zeta_key="notrust"),
        "C_notrust_zt": dict(step_clip="none", zeta_key="trust")}
OFFSET, RESTARTS = 6000, 100
RUNS = HERE / "runs_r5"


def zeta_star():
    z = json.loads((HERE / "zeta_star.json").read_text())
    return {s: {arm: z[s].get(arm, {}).get("zeta_star") for arm in ("trust", "notrust")} for s in z}


def cells():
    return [(s, n, arm) for s in SETTINGS for n in NS for arm in ARMS]


def name(s, n, arm):
    return f"{s}_n{n}_{arm}"


def run_index(i, restarts, offset):
    from engine_runner import cell
    s, n, arm = cells()[i]
    z = zeta_star()[s][ARMS[arm]["zeta_key"]]
    if z is None:
        raise SystemExit(f"no calibrated zeta for {s}/{ARMS[arm]['zeta_key']}")
    out = RUNS / (name(s, n, arm) + ".json")
    if out.exists():
        print("exists", out)
        return
    RUNS.mkdir(exist_ok=True)
    summ = cell(s, n, "no_lgd", "none", "baseline", restarts, offset, x_init="randn",
                zeta=z, step_clip=ARMS[arm]["step_clip"], step_tau=1.0)
    summ["arm"] = arm
    out.write_text(json.dumps(summ, indent=1, default=str))


def report():
    from _common import paired_stats
    d = {p.stem: json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))}
    lines = ["| setting | n | A trust@z*_t | B notrust@z*_n | C notrust@z*_t | A-B diff (base B - A, + = trust better) | 95% CI | wins | p | A-C diff | p |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for s in SETTINGS:
        for n in NS:
            A, B, C = (d.get(name(s, n, k)) for k in ARMS)
            if not (A and B):
                continue
            ab = paired_stats(B["scores"], A["scores"])
            ac = paired_stats(C["scores"], A["scores"]) if C else None
            lines.append(f"| {s} | {n} | {A['score']:.4f} (z={A['protocol']['zeta']}) | {B['score']:.4f} (z={B['protocol']['zeta']}) | "
                         f"{C['score']:.4f} | " if C else f"| {s} | {n} | {A['score']:.4f} | {B['score']:.4f} | - | ")
            lines[-1] += (f"{ab['mean_diff']:+.4f} | [{ab['ci95'][0]:+.3f}, {ab['ci95'][1]:+.3f}] | {ab['wins_for_b']}/{ab['n']} | {ab['perm_p']:.4f} | "
                          + (f"{ac['mean_diff']:+.4f} | {ac['perm_p']:.4f} |" if ac else "- | - |"))
            for k, cellj in zip(ARMS, (A, B, C)):
                if cellj:
                    rows.append({"setting": s, "n": n, "arm": k, "zeta": cellj["protocol"]["zeta"],
                                 "score": cellj["score"], "success": cellj["success_rate"],
                                 "diverged": cellj["diverged"], "calls": cellj["cm_samples_mean"],
                                 "s_per_run": cellj["seconds_per_run"], "offset": OFFSET,
                                 "restarts": cellj["restarts"]})
    (HERE / "r5_tables.md").write_text("\n".join(lines) + "\n")
    with open(HERE / "r5_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["setting"])
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run", "report"])
    ap.add_argument("--index", type=int)
    ap.add_argument("--restarts", type=int, default=RESTARTS)
    ap.add_argument("--offset", type=int, default=OFFSET)
    a = ap.parse_args()
    if a.mode == "list":
        for i, c in enumerate(cells()):
            print(i, name(*c))
    elif a.mode == "run":
        run_index(a.index, a.restarts, a.offset)
    else:
        report()


if __name__ == "__main__":
    main()
