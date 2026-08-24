"""Agent 1 -- hierarchical profile of simulations/experiments/_guided.py::run.

Monkeypatches call sites (instance attributes / class attributes / torch
functions) inside THIS process only; no repo file is edited.

Buckets (wall, perf_counter, exclusive unless stated):
  ddim          model_uncond.sample_ddim_step (includes uncond forward)
    uncond_fwd      DiffusionModel.forward
  cond_sample   model_cond.sample (n samples x 5 network evals over the 6-entry ladder)
    cond_fwd        ConsistencyModeliCT.forward (sum over ladder)
  mmd           MMDLoss.forward (vstack + kernel + 3 block means)
    kernel          RBF.forward (cdist + exp + sum over 5 bandwidths)
      cdist           torch.cdist inside RBF.forward
      exp             torch.exp  inside RBF.forward
  backward      torch.autograd.grad
  adam          AdamGuidance.step
  other         run() total - sum of top-level buckets (python, randn, logsumexp,
                requires_grad_, clone, isfinite, ...)

    python experiments/model-optimization/profiling/profile_guided.py
"""
import json
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parents[2] / "simulations"
sys.path.insert(0, str(SIM / "src"))
sys.path.insert(0, str(SIM / "experiments"))

import torch                                                   # noqa: E402
from _common import fixed_bandwidth, load, target_set          # noqa: E402  (installs the ot stub)
import LossFunctions                                           # noqa: E402
from LossFunctions import MMDLoss, RBF                         # noqa: E402
from tfg.adam_guidance import AdamGuidance                     # noqa: E402
from _guided import run                                        # noqa: E402
from _models import SEEDS, conditional_model, unconditional_model  # noqa: E402

TAG = {"2D": "", "5D": "_canonical", "10D": "_canonical"}
acc, cnt = defaultdict(float), defaultdict(int)
state = {"in_kernel": False}
shapes = defaultdict(int)


def timed(name, fn, flag=None):
    def w(*a, **k):
        if flag is not None:
            state[flag] = True
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            acc[name] += time.perf_counter() - t0
            cnt[name] += 1
            if flag is not None:
                state[flag] = False
    return w


_cdist, _exp, _grad = torch.cdist, torch.exp, torch.autograd.grad


def cdist_w(*a, **k):
    if not state["in_kernel"]:
        return _cdist(*a, **k)
    t0 = time.perf_counter()
    try:
        r = _cdist(*a, **k)
        shapes[("cdist", tuple(a[0].shape))] += 1
        return r
    finally:
        acc["cdist"] += time.perf_counter() - t0
        cnt["cdist"] += 1


def exp_w(*a, **k):
    if not state["in_kernel"]:
        return _exp(*a, **k)
    t0 = time.perf_counter()
    try:
        return _exp(*a, **k)
    finally:
        acc["exp"] += time.perf_counter() - t0
        cnt["exp"] += 1


_installed = {"global": False}


def install(mc, mu):
    mu.sample_ddim_step = timed("ddim", mu.sample_ddim_step)
    mu.forward = timed("uncond_fwd", mu.forward)
    mc.sample = timed("cond_sample", mc.sample)
    mc.forward = timed("cond_fwd", mc.forward)
    if _installed["global"]:          # class/torch-level patches only once
        return
    _installed["global"] = True
    MMDLoss.forward = timed("mmd", MMDLoss.forward)
    RBF.forward = timed("kernel", RBF.forward, flag="in_kernel")
    torch.cdist = cdist_w
    torch.exp = exp_w
    LossFunctions.torch.cdist = cdist_w
    LossFunctions.torch.exp = exp_w
    torch.autograd.grad = timed("backward", torch.autograd.grad)
    AdamGuidance.step = timed("adam", AdamGuidance.step)


def profile_cell(setting, spatial, temporal, n, restarts=(0, 1, 2, 3, 4), repeats=5):
    params = load(setting)
    S_G = target_set(params)
    bw = fixed_bandwidth(S_G)
    mc = conditional_model(params, seed=SEEDS[0], tag=TAG[setting])
    mu = unconditional_model(params, seed=SEEDS[0], tag=TAG[setting])
    install(mc, mu)
    run(mc, mu, S_G, bw, n, spatial, temporal, 0)          # warm-up
    per_run = []
    for r in restarts:
        for _ in range(repeats):
            acc.clear(); cnt.clear(); shapes.clear()
            t0 = time.perf_counter()
            _, info = run(mc, mu, S_G, bw, n, spatial, temporal, r)
            total = time.perf_counter() - t0
            top = ["ddim", "cond_sample", "mmd", "backward", "adam"]
            d = {k: acc[k] for k in acc}
            d["total"] = total
            d["other"] = total - sum(acc[k] for k in top)
            d["_cnt"] = dict(cnt)
            d["_steps"] = info["steps"]
            per_run.append(d)
    keys = sorted({k for d in per_run for k in d if not k.startswith("_")})
    med = {k: st.median([d.get(k, 0.0) for d in per_run]) for k in keys}
    iqr = {k: (min(d.get(k, 0.0) for d in per_run), max(d.get(k, 0.0) for d in per_run))
           for k in keys}
    counts = per_run[-1]["_cnt"]
    steps = per_run[-1]["_steps"]
    M = 3 if spatial == "lgd" else 1
    m = S_G.shape[0]
    nb = 5
    accounting = {
        "steps": steps,
        "denoiser_calls": counts.get("uncond_fwd", 0),
        "cond_sampler_calls": counts.get("cond_sample", 0),
        "cond_network_evals": counts.get("cond_fwd", 0),
        "conditional_samples": steps * M * n,
        "target_samples": m,
        "mmd_evals": counts.get("mmd", 0),
        "kernel_entries_per_mmd": (n + m) ** 2 * nb,
        "kernel_entries_per_run": counts.get("mmd", 0) * (n + m) ** 2 * nb,
        "target_target_entries_per_mmd": m * m * nb,
        "target_target_recomputed_per_run": counts.get("mmd", 0),
        "target_target_fraction_of_kernel": (m * m) / ((n + m) ** 2),
        "backward_calls": counts.get("backward", 0),
        "adam_steps": counts.get("adam", 0),
        "cdist_shapes": {str(k[1]): v for k, v in shapes.items()},
    }
    return {"setting": setting, "spatial": spatial, "temporal": temporal, "n": n,
            "restarts": list(restarts), "repeats": repeats,
            "median_s": med, "minmax_s": iqr, "accounting": accounting,
            "per_step_ms": {k: 1e3 * v / steps for k, v in med.items()}}


def fmt(res):
    med = res["median_s"]
    tot = med["total"]
    lines = [f"## {res['setting']} {res['spatial']}/{res['temporal']} n={res['n']}  "
             f"total median {tot*1e3:.1f} ms/run, {tot*1e3/res['accounting']['steps']:.2f} ms/step",
             "| bucket | median s | % of total | per step ms | min s | max s |", "|---|---|---|---|---|---|"]
    order = ["ddim", "uncond_fwd", "cond_sample", "cond_fwd", "mmd", "kernel", "cdist", "exp",
             "backward", "adam", "other", "total"]
    ind = {"uncond_fwd": "  ", "cond_fwd": "  ", "kernel": "  ", "cdist": "    ", "exp": "    "}
    for k in order:
        if k not in med:
            continue
        v = med[k]
        lo, hi = res["minmax_s"][k]
        lines.append(f"| {ind.get(k,'')}{k} | {v:.4f} | {100*v/tot:.1f}% | "
                     f"{1e3*v/res['accounting']['steps']:.3f} | {lo:.4f} | {hi:.4f} |")
    return "\n".join(lines)


def main():
    cells = [("2D", "no_lgd", "none", 8), ("2D", "no_lgd", "none", 32),
             ("2D", "lgd", "none", 8), ("2D", "lgd", "none", 32),
             ("2D", "no_lgd", "adam", 32),
             ("10D", "no_lgd", "none", 32), ("10D", "lgd", "none", 32)]
    out = []
    for c in cells:
        r = profile_cell(*c)
        out.append(r)
        print(fmt(r)); print(json.dumps(r["accounting"], indent=None)); print(flush=True)
    (HERE / "profile_buckets.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
