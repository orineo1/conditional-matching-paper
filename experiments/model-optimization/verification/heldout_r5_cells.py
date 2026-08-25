"""Agent 6 -- round-5 CONFIRMATORY re-run (corrected protocol) at a fresh offset.

Re-runs the H-R5 primary comparison (A = trust_noise1 @ zeta*_trust vs
B = no trust @ zeta*_notrust, x_T ~ N(0, I), per-arm calibrated zeta from
protocol/zeta_star.json) at OFFSET 7000 (never used), R=100, with all cells
of a setting in ONE process on one node (the round-5 run used one cell per
array task, so A/B same-node pairing is unverifiable there), plus the 2D
sensitivity arm A8 = trust_noise1 @ zeta = 8 (the basin-rule value; the
amended l2-min rule picked 16, the two being within seed noise at n = 128).
Arm C (no trust @ zeta*_trust) is NOT re-run: it diverged on 71-99% of the
restarts in every cell, which no re-run will change.

Cells (28): {2D,5D,10D} x n in {4,8,16,32} x {A, B}; 2D x n x {A8}.

    cd simulations
    python ../experiments/model-optimization/verification/heldout_r5_cells.py list
    python ../experiments/model-optimization/verification/heldout_r5_cells.py run --index I [--restarts 100] [--offset 7000]
Outputs: verification/heldout_runs_r5/<setting>_n<n>_<arm>_off<offset>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MO / "estimator"))
import heldout_cells as hc                                    # noqa: E402
import engine_runner as er                                    # noqa: E402

OUT = HERE / "heldout_runs_r5"
SETTINGS, NS = ["2D", "5D", "10D"], [4, 8, 16, 32]
ARMS = {"A_trust_zt": dict(step_clip="noise", zeta_key="trust"),
        "B_notrust_zn": dict(step_clip="none", zeta_key="notrust"),
        "A8_trust_z8": dict(step_clip="noise", zeta_key=None, zeta=8.0)}   # 2D only


def zeta_star():
    z = json.loads((MO / "protocol" / "zeta_star.json").read_text())
    return {s: {a: z[s][a]["zeta_star"] for a in ("trust", "notrust")} for s in z}


def group_list(settings=SETTINGS):
    groups = []
    for s in settings:
        g = [(s, n, arm) for n in NS for arm in ("A_trust_zt", "B_notrust_zn")]
        if s == "2D":
            g += [(s, n, "A8_trust_z8") for n in NS]
        groups.append(g)
    return groups


def run_cell(c, restarts, offset):
    s, n, arm = c
    spec = ARMS[arm]
    z = spec["zeta"] if spec.get("zeta") else zeta_star()[s][spec["zeta_key"]]
    out = OUT / f"{s}_n{n}_{arm}_off{offset}.json"
    if out.exists():
        print("exists", out.name)
        return out
    t0 = time.perf_counter()
    summ = er.cell(s, n, "no_lgd", "none", "baseline", restarts, offset, "float32", "tape",
                   x_init="randn", zeta=float(z), step_clip=spec["step_clip"], step_tau=1.0)
    summ = hc.add_eval_mmd(summ, s, offset)
    import platform, socket, torch
    summ["arm"] = arm
    summ["verifier"] = {"agent": 6, "round": 5, "offset": offset, "restarts": restarts,
                        "total_wall_s": time.perf_counter() - t0, "host": socket.gethostname(),
                        "cpu": hc.cpu_model(), "platform": platform.platform(),
                        "torch": torch.__version__, "threads": torch.get_num_threads()}
    assert summ["protocol"]["x_init"] == "randn" and abs(summ["protocol"]["zeta"] - float(z)) < 1e-9
    assert abs(summ["cm_samples_mean"] - n * 99) < 1e-6
    OUT.mkdir(exist_ok=True)
    out.write_text(json.dumps(summ, indent=1, default=str))
    print(f"wrote {out.name} score={summ['score']:.4f} div={summ['diverged']} zeta={z}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run"])
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--restarts", type=int, default=100)
    ap.add_argument("--offset", type=int, default=7000)
    ap.add_argument("--settings", nargs="*", default=SETTINGS)
    a = ap.parse_args()
    assert a.offset >= 7000, "round-5 confirmatory offset must be >= 7000 (5000/6000 used)"
    groups = group_list(a.settings)
    if a.mode == "list":
        for i, g in enumerate(groups):
            print(i, g[0][0], len(g), "cells")
        print(f"# {len(groups)} groups, {sum(len(g) for g in groups)} cells", file=sys.stderr)
        return
    for g in ([groups[a.index]] if a.index is not None else groups):
        print(f"r5-confirm group {g[0][0]} on {hc.cpu_model()}", flush=True)
        for c in g:
            run_cell(c, a.restarts, a.offset)


if __name__ == "__main__":
    main()
