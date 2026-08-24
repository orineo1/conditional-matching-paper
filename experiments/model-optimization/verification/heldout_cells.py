"""Agent 6 -- HELD-OUT confirmation of the Agent-4 estimator candidates.

Same engine path as the screening (estimator/engine_runner.py, imported, NOT
edited), but restarts 1000..1099 (offset 1000 -- never touched by the
screening, which used restarts 0..39 at offset 0), 100 restarts per cell.

    cd simulations
    python ../experiments/model-optimization/verification/heldout_cells.py list          # 24 groups (108 cells)
    python ../experiments/model-optimization/verification/heldout_cells.py run --index I [--restarts 100] [--offset 1000]   # one group
    python ../experiments/model-optimization/verification/heldout_cells.py run            # all groups, sequential (cluster only)

Cells (108): settings 2D/5D/10D x n in {4,8,16,32} x arm no_lgd/none x
{baseline, relclip2, relclip_ema2, trust_noise1, sqrt_floor, clip0.5, relclip1,
sqrtfloor_clip0.5}; Pareto baselines no_lgd/none n in {64,96}; lgd/none
baseline n in {8,32}.

GROUPING (differs from the screening on purpose): the float32 trajectory is
chaotic at round-off level (see PHASE1.md: the same restart lands in a
different mode on the Mac and on the cluster), so a paired comparison is only
clean when baseline and candidate run on the SAME CPU type.  One array task
therefore runs one GROUP = (setting, n, arm) with its baseline and all its
candidates sequentially in one process (18 groups; the 12 paired groups take
~10-30 min each), and the CPU model is recorded in every JSON.
Output: verification/heldout_runs/<cell>_off<offset>.json
with the engine_runner summary (score, success_rate, diverged, cm_samples,
wall_s, per-restart `runs` with L2/abs_err/diverged/calls/seconds) PLUS an
independent end-point metric per restart: ``mmd2_eval`` = the optimisation
objective MMD^2(Y_eval(x_hat), S_G) with n_eval=256 fresh conditional draws
keyed ("heldout_eval", restart) (the same for every candidate -> paired).
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MO = HERE.parent
sys.path.insert(0, str(MO / "estimator"))
import engine_runner as er                                   # noqa: E402  (sets up sys.path to simulations)

import torch                                                 # noqa: E402
from tfg.distributional import CMSampler, DistributionalLoss  # noqa: E402
from tfg.noise_tape import NoiseTape                         # noqa: E402
from _models import PAPER_TS                                 # noqa: E402

OUT = HERE / "heldout_runs"
SETTINGS = ["2D", "5D", "10D"]
NS = [4, 8, 16, 32]
CANDS = ["baseline", "relclip2", "relclip_ema2", "trust_noise1", "sqrt_floor",
         "clip0.5", "relclip1", "sqrtfloor_clip0.5"]
PARETO_NS = [64, 96]
LGD_NS = [8, 32]
N_EVAL = 256


def group_list(settings=SETTINGS):
    """One group per array task: a list of cells run in one process (same node)."""
    groups = []
    for s in settings:
        for n in NS:
            groups.append([(s, n, "no_lgd", "none", c) for c in CANDS])
        groups.append([(s, n, "no_lgd", "none", "baseline") for n in PARETO_NS])
        groups.append([(s, n, "lgd", "none", "baseline") for n in LGD_NS])
    return groups


def cell_list(settings=SETTINGS):
    return [c for g in group_list(settings) for c in g]


def group_name(g):
    s, n, sp, tp, c = g[0]
    if len(g) > 1 and len({x[1] for x in g}) > 1:
        return f"{s}_{sp}_{tp}_baseline_n{'-'.join(str(x[1]) for x in g)}"
    return f"{s}_n{n}_{sp}_{tp}_{len(g)}cands"


def cpu_model():
    import platform
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def cell_name(s, n, sp, tp, c, offset):
    return f"{s}_n{n}_{sp}_{tp}_{c}_tape_off{offset}"


def add_eval_mmd(summ, setting, offset):
    """Independent end-point metric: MMD^2 of n_eval fresh conditional samples at
    x_hat vs S_G, fixed bandwidth, keyed on the restart only (paired across
    candidates, independent of the guidance noise)."""
    params, S_G, bw, mc, mu = er.build_models(setting, torch.float32)
    loss = DistributionalLoss(S_G, bandwidth="fixed", bandwidth_value=bw, transform="mmd2")
    for i, run in enumerate(summ["runs"]):
        r = offset + i
        if run["diverged"]:
            run["mmd2_eval"] = None
            continue
        tape = NoiseTape(seed=10_000_000 + r, dtype=torch.float32)
        sampler = CMSampler(mc, PAPER_TS, tape, source="tape", dtype=torch.float32)
        x = torch.tensor(run["x_hat"], dtype=torch.float32).reshape(1, -1)
        keys = [("heldout_eval", r, i_) for i_ in range(N_EVAL)]
        with torch.no_grad():
            run["mmd2_eval"] = float(loss(sampler(x, keys)))
    vals = [q["mmd2_eval"] for q in summ["runs"] if q["mmd2_eval"] is not None]
    summ["mmd2_eval_mean"] = sum(vals) / len(vals) if vals else None
    summ["n_eval"] = N_EVAL
    return summ


def run_cell(c, restarts, offset):
    s, n, sp, tp, cand = c
    out = OUT / (cell_name(*c, offset) + ".json")
    if out.exists():
        print("exists", out.name)
        return out
    t0 = time.perf_counter()
    summ = er.cell(s, n, sp, tp, cand, restarts, offset, "float32", "tape")
    summ = add_eval_mmd(summ, s, offset)
    import platform, socket
    summ["verifier"] = {"agent": 6, "offset": offset, "restarts": restarts,
                        "commit": er.COMMIT, "total_wall_s": time.perf_counter() - t0,
                        "host": socket.gethostname(), "cpu": cpu_model(),
                        "platform": platform.platform(), "torch": torch.__version__,
                        "threads": torch.get_num_threads()}
    OUT.mkdir(exist_ok=True)
    out.write_text(json.dumps(summ, indent=1, default=str))
    print(f"wrote {out.name}  score={summ['score']:.4f} mmd2_eval={summ['mmd2_eval_mean']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run"])
    ap.add_argument("--index", type=int, default=None, help="group index from `list`")
    ap.add_argument("--restarts", type=int, default=100)
    ap.add_argument("--offset", type=int, default=1000)
    ap.add_argument("--settings", nargs="*", default=SETTINGS)
    a = ap.parse_args()
    assert a.offset >= 1000, "held-out offsets must be >= 1000 (0..39 were used for screening)"
    groups = group_list(a.settings)
    if a.mode == "list":
        for i, g in enumerate(groups):
            print(i, group_name(g), len(g), "cells")
        print(f"# {len(groups)} groups, {sum(len(g) for g in groups)} cells", file=sys.stderr)
        return
    todo = [groups[a.index]] if a.index is not None else groups
    for g in todo:
        print(f"group {group_name(g)} on {cpu_model()}", flush=True)
        for c in g:
            run_cell(c, a.restarts, a.offset)


if __name__ == "__main__":
    main()
