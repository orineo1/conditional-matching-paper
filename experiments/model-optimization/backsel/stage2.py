"""Agent B, B-R6b stage 2 -- end-to-end quality at n = 128 (pre-registered in
hypotheses/agentB.yaml before running).

    cd simulations
    python ../experiments/model-optimization/backsel/stage2.py run --setting 10D --restarts 5 --dir stage2_smoke   # smoke
    python ../experiments/model-optimization/backsel/stage2.py run --setting 10D              # R=100, offset 9000
    python ../experiments/model-optimization/backsel/stage2.py report [--dir stage2_runs]

One process per setting runs EVERY arm on the same restarts (paired).  The
128 candidate conditional samples at (restart, t) are identical across arms
(noise keyed per sample); backsel arms regenerate only the selected rows with
graphs; fresh-k arms use the first k rows.  Corrected protocol: x_T ~ N(0, I)
per restart, trust_noise1, calibrated zeta (fidelity.py).
"""
import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIM = ROOT / "simulations"
for p in (SIM / "src", SIM / "experiments", HERE.parent / "estimator", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fidelity import zeta_dimy                                              # noqa: E402
from tfg.backsel import output_gradients, select_kcenter, select_uniform     # noqa: E402
from tfg.noise_tape import NoiseTape                                        # noqa: E402

N_GEN = 128
ARMS = ["full128", "kcenter16", "kcenter32", "kcenter_mean2_16", "uniform32", "fresh16", "fresh32"]
PENALTY = 2.0


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == "darwin" else r / 2 ** 10


# ---------------------------------------------------------------------------
# settings with a split sampler: draw(restart, t, n) -> noise ; build(x0, noise, idx) -> y
# ---------------------------------------------------------------------------

def build(setting):
    from _models import SEEDS, unconditional_model
    if setting.startswith(("dimy", "nuis")):
        from exp5_dimy_scaling import evaluate as ev5, target_samples
        from tfg import oracle
        from tfg.dimy_benchmark import as_params, as_params_nuisance
        from LossFunctions import MMDLoss, RBF
        d = int(setting[4:])
        params = as_params_nuisance(d - 1) if setting.startswith("nuis") else as_params(d)
        S_G = target_samples(params, 250, torch.Generator().manual_seed(987654))
        d2 = torch.cdist(S_G, S_G, p=2) ** 2
        bw = float(d2.sum() / (S_G.shape[0] ** 2 - S_G.shape[0]))
        mu = unconditional_model(params, seed=SEEDS[0], tag=f"_dimy{d}")
        mmd = MMDLoss(kernel=RBF(bandwidth=bw, device="cpu"), device="cpu")
        S32 = S_G.float()
        d_y = params["target_means"].shape[1]

        def draw(restart, t, n):
            # per-row uniform (component, inverted at build time against the
            # x-dependent responsibilities -- the multinomial's detached role)
            # and Gaussian noise: reproducible per row, so subsets regenerate exactly
            gen = torch.Generator().manual_seed(abs(hash(("s2", restart, t))) % (2 ** 31))
            u = torch.rand(n, generator=gen, dtype=torch.float64)
            eps = torch.randn(n, d_y, generator=gen, dtype=torch.float64)
            return (u, eps)

        def build_y(x0, noise, idx=None):
            u, eps = noise
            if idx is not None:
                u, eps = u[idx], eps[idx]
            cm, cc, w = oracle.conditional_params(
                x0.reshape(-1).double(), params["mu_list"], params["Sigma_list"], params["alpha"])
            L = torch.linalg.cholesky(cc if cc.dim() == 2 else cc[0])
            comp = torch.searchsorted(torch.cumsum(w.detach(), 0), u).clamp(max=cm.shape[0] - 1)
            return (cm[comp] + eps @ L.T).float()

        def loss(y):
            return mmd(y, S32)

        def evaluate(x, info):
            return ev5(float(x.reshape(-1)[0]), params, info)
        meta = {"dim_x": 1, "dim_y": d, "bw": bw, "zeta": zeta_dimy(d),
                "protocol": "x_init=randn, step_clip=noise tau=1, zeta=exp5b_v2"}
        return mu, draw, build_y, loss, evaluate, meta
    if setting in ("10D", "10D_z003"):
        # primary: the campaign's corrected-protocol calibration
        # (protocol/zeta_star.json, 10D trust: l2-min over divergence-free zeta
        # at n=128, trust on) = 4.0.  "10D_z003": labelled SENSITIVITY group at
        # exp12's 0.03125 (another session's number; guidance nearly inert).
        zs = json.loads((HERE.parent / "protocol" / "zeta_star.json").read_text())
        zeta10 = float(zs["10D"]["trust"]["zeta_star"])
        if setting == "10D_z003":
            zeta10 = 0.03125
        from _common import fixed_bandwidth, load, target_set
        from _guided import evaluate as evg
        from _models import PAPER_TS, conditional_model
        from tfg.distributional import CMSampler, DistributionalLoss
        params = load("10D")
        S_G = target_set(params)
        bw = fixed_bandwidth(S_G)
        mc = conditional_model(params, seed=SEEDS[0])
        mu = unconditional_model(params, seed=SEEDS[0])
        tape = NoiseTape(seed=777, dtype=torch.float32)
        cms = CMSampler(mc, PAPER_TS, tape, source="tape", dtype=torch.float32)
        dl = DistributionalLoss(S_G, bandwidth="fixed", bandwidth_value=bw, backend="fast")

        def draw(restart, t, n):
            return [("s2", int(restart), int(t), i) for i in range(n)]

        def build_y(x0, keys, idx=None):
            if idx is not None:
                keys = [keys[i] for i in idx]
            return cms(x0.reshape(1, -1), keys)

        def evaluate(x, info):
            return evg(x.detach().reshape(-1), params, info)
        src = ("protocol/zeta_star.json 10D.trust (l2min, n=128, trust on)" if setting == "10D"
               else "SENSITIVITY: exp12 none-arm 0.03125 (not the campaign calibration)")
        meta = {"dim_x": 9, "dim_y": 1, "bw": bw, "zeta": zeta10,
                "protocol": f"x_init=randn, step_clip=noise tau=1, zeta={src}"}
        return mu, draw, build_y, dl, evaluate, meta
    raise ValueError(setting)


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def kcenter_mean2_pairs(Y, g, k, tape, key):
    kc = max(1, k // 2)
    centers, g_eff = select_kcenter(Y, g, kc, tape, key)
    Yd = Y.double()
    assign = torch.cdist(Yd, Yd[centers]).argmin(dim=1)
    u = tape.randn(key + ("m2",), (Y.shape[0],), dtype=torch.float64)
    idx, geff_rows = [], []
    for c, r in enumerate(centers):
        members = [i for i in torch.nonzero(assign == c).reshape(-1).tolist() if i != r]
        if members:
            other = max(members, key=lambda i: float(u[i]))
            idx += [r, other]
            geff_rows += [0.5 * g_eff[c], 0.5 * g_eff[c]]
        else:
            idx.append(r)
            geff_rows.append(g_eff[c])
    return idx, torch.stack(geff_rows)


def guidance_loss(arm, x0, restart, t, draw, build_y, loss, tape, stats, diag):
    """Return a scalar whose gradient w.r.t. x_t is the arm's estimator."""
    if arm == "full128":
        y = build_y(x0, draw(restart, t, N_GEN))
        stats["fwd"] += N_GEN; stats["diff"] += N_GEN
        return loss(y)
    if arm.startswith("fresh"):
        k = int(arm[5:])
        y = build_y(x0, draw(restart, t, N_GEN), list(range(k)))
        stats["fwd"] += k; stats["diff"] += k
        return loss(y)
    noise = draw(restart, t, N_GEN)
    with torch.no_grad():
        Y = build_y(x0, noise)
    g, val = output_gradients(loss, Y)
    key = ("s2sel", int(restart), int(t))
    if arm.startswith("kcenter_mean2"):
        idx, g_eff = kcenter_mean2_pairs(Y, g, int(arm.split("_")[-1]), tape, key)
    elif arm.startswith("kcenter"):
        k = int(arm[7:])
        idx, g_eff = select_kcenter(Y, g, k, tape, key)
        if diag is not None and k == 16:
            Yd = Y.double()
            assign = torch.cdist(Yd, Yd[idx]).argmin(dim=1)
            sizes = torch.bincount(assign, minlength=len(idx)).tolist()
            prev = diag.get("prev")
            rec = {"t": int(t), "sizes": sizes}
            if prev is not None:
                a, b = set(prev["idx"]), set(idx)
                rec["jaccard"] = len(a & b) / len(a | b)
                C_prev, C = prev["centers"], Yd[idx]
                disp = torch.cdist(C, C_prev).min(dim=1).values.mean()
                spacing = torch.cdist(C, C)[~torch.eye(len(idx), dtype=bool)].mean()
                rec["center_disp_rel"] = float(disp / (spacing + 1e-12))
            diag["prev"] = {"idx": idx, "centers": Yd[idx].clone()}
            diag.setdefault("steps", []).append(rec)
    elif arm.startswith("uniform"):
        idx, g_eff = select_uniform(g, int(arm[7:]), tape, key)
    else:
        raise ValueError(arm)
    y_sel = build_y(x0, noise, idx)
    stats["fwd"] += N_GEN + len(idx); stats["diff"] += len(idx)
    surr = (y_sel * g_eff.to(y_sel.dtype)).sum()
    return val.to(surr.dtype) + surr - surr.detach()


def run_one(arm, mu, draw, build_y, loss, zeta, restart, tape, diag=None):
    T = mu.diffusion_steps
    g0 = torch.Generator().manual_seed(0x5EED0000 ^ int(restart))
    x = torch.randn(1, mu.nfeatures, generator=g0, dtype=torch.float32)
    stats = {"fwd": 0, "diff": 0}
    diverged, t0 = False, time.perf_counter()
    for t in range(T - 1, 0, -1):
        x = x.detach().clone().requires_grad_(True)
        x_prev, pred_x0 = mu.sample_ddim_step(x, t, condition_x=None, device="cpu", eta=0.0)
        L = guidance_loss(arm, pred_x0, restart, t, draw, build_y, loss, tape, stats, diag)
        g, = torch.autograd.grad(L, x, allow_unused=True)
        g = torch.zeros_like(x) if g is None else g
        upd = zeta * g.detach()
        with torch.no_grad():
            ref = (1 - mu.baralphas[t]).sqrt()
            upd = upd * torch.clamp(ref / (upd.norm() + 1e-12), max=1.0)
            x = x_prev.detach() - upd
        if not torch.isfinite(x).all() or float(x.abs().max()) > 50.0:
            diverged = True
            break
    return x.detach(), {"conditional_calls": stats["fwd"], "fwd_samples": stats["fwd"],
                        "diff_samples": stats["diff"], "seconds": time.perf_counter() - t0,
                        "diverged": diverged}


def run_setting(setting, restarts, offset, out_dir, arms=ARMS):
    mu, draw, build_y, loss, evaluate, meta = build(setting)
    zeta = meta["zeta"]
    tape = NoiseTape(seed=9001)
    out = {"setting": setting, "meta": meta, "restarts": restarts, "offset": offset, "arms": {}}
    for arm in arms:
        runs, diag = [], ({} if arm == "kcenter16" else None)
        t0 = time.perf_counter()
        for r in range(offset, offset + restarts):
            if diag is not None:
                diag.pop("prev", None)
            x, info = run_one(arm, mu, draw, build_y, loss, zeta, r, tape, diag)
            ev = evaluate(x, info)
            ev.update({k: info[k] for k in ("fwd_samples", "diff_samples")})
            runs.append(ev)
        wall = time.perf_counter() - t0
        scores = [min(q["L2"], PENALTY) if not q["diverged"] else PENALTY for q in runs]
        fin = [q for q in runs if not q["diverged"]]
        out["arms"][arm] = {
            "scores": scores, "score": sum(scores) / len(scores),
            "success_rate": sum(1 for q in fin if q["abs_err"] < 0.5) / restarts,
            "diverged": len(runs) - len(fin),
            "fwd_samples": sum(q["fwd_samples"] for q in runs) / restarts,
            "diff_samples": sum(q["diff_samples"] for q in runs) / restarts,
            "seconds_per_run": wall / restarts, "peak_mem_mb": peak_rss_mb()}
        if diag is not None:
            out["arms"][arm]["kcenter_diag"] = diag.get("steps", [])
        a = out["arms"][arm]
        print(f"{setting} {arm:<17} score={a['score']:.4f} succ={a['success_rate']:.0%} div={a['diverged']} "
              f"fwd={a['fwd_samples']:.0f} diff={a['diff_samples']:.0f} {a['seconds_per_run']:.2f}s/run "
              f"RSS={a['peak_mem_mb']:.0f}MB", flush=True)
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{setting}.json").write_text(json.dumps(out, default=str))


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def report(out_dir):
    from _common import paired_stats
    data = {p.stem: json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))}
    if not data:
        print("no runs"); return
    lines = ["# B-R6b stage 2 -- end-to-end quality at n = 128 (corrected protocol)\n",
             "score = failure-penalised exact L2 (lower better; 10D: GMM L2 of _guided.evaluate, dimY/nuis: "
             "exp5 population L2), paired over the same restarts; diff = comparator - arm (+ = arm better), "
             "bootstrap 95% CI, permutation p.  fwd = conditional forward samples per run, diff_s = "
             "differentiated samples per run.  NOTE: in dimy16/nuis16 the conditional is the ORACLE with a "
             "common Jacobian, so kcenter aggregation is exact by construction there -- those cells test the "
             "protocol, not the approximation; 10D is the informative setting.\n"]
    rows = []
    for setting, d in data.items():
        arms, meta = d["arms"], d["meta"]
        lines.append(f"\n## {setting} (dim x={meta['dim_x']}, dim y={meta['dim_y']}, zeta={meta['zeta']:.4g}, "
                     f"{meta['protocol']}; R={d['restarts']}, offset {d['offset']})\n")
        lines.append("| arm | score | succ | div | fwd/run | diff_s/run | s/run | RSS MB | vs full128 diff [CI] p | vs fresh-k diff [CI] p |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        full = arms.get("full128")
        for arm, a in arms.items():
            k = "".join(ch for ch in arm.split("_")[-1] if ch.isdigit())
            fresh = arms.get(f"fresh{k}") if not arm.startswith(("full", "fresh")) else None
            cols = [arm, f"{a['score']:.4f}", f"{a['success_rate']:.0%}", str(a["diverged"]),
                    f"{a['fwd_samples']:.0f}", f"{a['diff_samples']:.0f}", f"{a['seconds_per_run']:.2f}",
                    f"{a['peak_mem_mb']:.0f}"]
            rec = {"setting": setting, "arm": arm, "score": a["score"], "success": a["success_rate"],
                   "diverged": a["diverged"], "fwd_samples": a["fwd_samples"], "diff_samples": a["diff_samples"],
                   "seconds_per_run": a["seconds_per_run"], "peak_mem_mb": a["peak_mem_mb"]}
            for comp, tag in ((full, "full"), (fresh, "fresh")):
                if comp is None or comp is a:
                    cols.append("-"); rec[f"diff_vs_{tag}"] = None; continue
                s = paired_stats(comp["scores"], a["scores"])
                cols.append(f"{s['mean_diff']:+.3f} [{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}] p={s['perm_p']:.3f}")
                rec[f"diff_vs_{tag}"], rec[f"ci_vs_{tag}"], rec[f"p_vs_{tag}"] = s["mean_diff"], s["ci95"], s["perm_p"]
            lines.append("| " + " | ".join(cols) + " |")
            rows.append(rec)
        dg = arms.get("kcenter16", {}).get("kcenter_diag", [])
        if dg:
            sizes = [s for r in dg for s in r["sizes"]]
            single = sum(1 for s in sizes if s == 1) / len(sizes)
            maxes = [max(r["sizes"]) for r in dg]
            jac = [r["jaccard"] for r in dg if "jaccard" in r]
            disp = [r["center_disp_rel"] for r in dg if "center_disp_rel" in r]
            lines.append(f"\nkcenter16 diagnostic ({len(dg)} steps): cluster size min/median/max = "
                         f"{min(sizes)}/{q(sizes, .5)}/{max(sizes)}, per-step max size median {q(maxes, .5)}, "
                         f"singleton fraction {single:.2f}; selection stability between consecutive steps: "
                         f"index-Jaccard median {q(jac, .5):.3f} (chance ~0.067, noise is fresh per step), "
                         f"center-set displacement / within-step center spacing median {q(disp, .5):.3f} "
                         f"[p10 {q(disp, .1):.3f}, p90 {q(disp, .9):.3f}].")
    (HERE / "stage2_tables.md").write_text("\n".join(lines) + "\n")
    with open(HERE / "stage2_rows.csv", "w", newline="") as fh:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows -> stage2_rows.csv, stage2_tables.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "report"])
    ap.add_argument("--setting", default="10D", choices=["10D", "dimy16", "nuis16", "10D_z003"])
    ap.add_argument("--restarts", type=int, default=100)
    ap.add_argument("--offset", type=int, default=9000)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--dir", default="stage2_runs")
    a = ap.parse_args()
    out_dir = HERE / a.dir
    if a.mode == "report":
        report(out_dir)
    else:
        run_setting(a.setting, a.restarts, a.offset, out_dir, a.arms)


if __name__ == "__main__":
    main()
