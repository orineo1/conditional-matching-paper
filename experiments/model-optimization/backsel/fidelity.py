"""Agent B, B-R6 stage 1 -- gradient fidelity of subset backprop at n = 128.

    cd simulations
    python ../experiments/model-optimization/backsel/fidelity.py --setting dimy8 --trajectories 2   # smoke
    python ../experiments/model-optimization/backsel/fidelity.py --setting 10D
    python ../experiments/model-optimization/backsel/fidelity.py report          # tables + figure

No end-to-end guidance runs are scored.  Along real trust_noise1 trajectories
(corrected protocol: x_T ~ N(0, I), step_clip=noise tau=1, calibrated zeta,
n = 128 at every step) we probe ~11 timesteps per trajectory.  At each probe,
with 128 fresh conditional samples y_i = S(x_{0|t}, eta_i):

    G     = dL/dx_t            (all 128 differentiated -- the ceiling)
    g_i   = dL/dy_i            (output-space, kernel-only)
    J_i   = dy_i/dx_t          (per-sample Jacobian, batched vjp)
    h_i   = J_i^T g_i          (per-sample contribution; sum_i h_i = G, asserted)

Every subset estimator is then an EXACT linear function of (J_i, g_i), so the
20 selection draws x 4 k x 4 rules cost nothing beyond the per-probe Jacobian:

    uniform / importance : G_hat = sum_{i in S} w_i h_i         (tfg.backsel rules)
    kcenter              : G_hat = sum_c J_{r_c}^T (sum_{i in C_c} g_i)
    kcenter_mean2        : k/2 clusters, two members a_c, b_c differentiated,
                           G_hat = sum_c ((J_{a_c} + J_{b_c})/2)^T (sum_{i in C_c} g_i)

Settings: dimy8 / dimy16 (dimy_benchmark, ORACLE conditional as in Exp 5 --
common Jacobian across samples, see hypotheses/agentB.yaml B-R6) and 10D
(paper setting, trained CM, dim x = 9, per-sample Jacobians).  Metrics per
(setting, t-bucket, rule, k): cos_x (x-space), cos_y (output space,
s_hat = sum w_i g_i vs s = sum g_i), relative norm error, std over draws, ESS
of ||h_i|| and ||g_i||, clusteredness of the y_i (silhouette / within-between
at the k=8 k-center partition, effective rank of cov(y)).
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIM = ROOT / "simulations"
for p in (SIM / "src", SIM / "experiments", HERE.parent / "estimator"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tfg.backsel import (select_importance, select_kcenter, select_stratified,   # noqa: E402
                         select_uniform, soft_aggregate)
from tfg.noise_tape import NoiseTape                                        # noqa: E402

N_GEN = 128
KS = [8, 16, 32, 64]
RULES = ["uniform", "importance", "kcenter", "kcenter_mean2"]
# [B-R7] soft assignment: "soft_<selection>_t<tau_mult>", tau = tau_mult x bandwidth
TAUS = [0.25, 1.0, 4.0]
RULES_R7 = (["uniform", "kcenter", "stratified"]
            + [f"soft_{sel}_t{tm}" for sel in ("uniform", "stratified", "kcenter") for tm in TAUS])
PROBE_T = [98, 90, 80, 70, 60, 50, 40, 30, 20, 10, 3]
ZETA = {"dimy8": None, "dimy16": None, "10D": 0.03125}     # dimY from exp5b v2 json
OUT = HERE / "fidelity_runs"


def bucket(t):
    return "early" if t >= 70 else ("mid" if t >= 30 else "late")


def zeta_dimy(d):
    f = SIM / "results" / "tfg" / "exp5b_zeta_calibration_v2.json"
    z = json.loads(f.read_text())["zeta_star"][str(d)]
    return float(z)


# ---------------------------------------------------------------------------
# settings: (mu, sampler(x0, restart, t, draw) -> y (n, d_y) differentiable, loss, zeta)
# ---------------------------------------------------------------------------

def build(setting):
    from _models import SEEDS, unconditional_model
    if setting.startswith(("dimy", "nuis")):
        from exp5_dimy_scaling import conditional_samples, target_samples
        from tfg.dimy_benchmark import as_params, as_params_nuisance
        from LossFunctions import MMDLoss, RBF
        d = int(setting[4:])
        # nuis<d>: dim y = d with ONE informative coordinate + d-1 pure-noise
        # coordinates (dimy_benchmark.build_nuisance) -- the construction whose
        # MMD gets HARDER with d.  Same X marginal as dimy<d>, so the dimy<d>
        # unconditional checkpoint and zeta* are reused (pre-registered addendum).
        params = as_params_nuisance(d - 1) if setting.startswith("nuis") else as_params(d)
        S_G = target_samples(params, 250, torch.Generator().manual_seed(987654))
        d2 = torch.cdist(S_G, S_G, p=2) ** 2
        bw = float(d2.sum() / (S_G.shape[0] ** 2 - S_G.shape[0]))
        mu = unconditional_model(params, seed=SEEDS[0], tag=f"_dimy{d}")
        mmd = MMDLoss(kernel=RBF(bandwidth=bw, device="cpu"), device="cpu")
        S32 = S_G.float()

        def sampler(x0, restart, t, draw, n=N_GEN):
            gen = torch.Generator().manual_seed(abs(hash(("fid", restart, t, draw, n))) % (2 ** 31))
            return conditional_samples(params, x0.reshape(-1), n, gen).float()

        def loss(y):
            return mmd(y, S32)
        cond = ("oracle nuisance 1+%d (common Jacobian)" % (d - 1) if setting.startswith("nuis")
                else "oracle (common Jacobian)")
        return mu, sampler, loss, zeta_dimy(d), {"dim_x": 1, "dim_y": d, "bw": bw, "conditional": cond}
    if setting == "10D":
        from _common import fixed_bandwidth, load, target_set
        from _models import PAPER_TS, conditional_model
        from tfg.distributional import CMSampler, DistributionalLoss
        params = load("10D")
        S_G = target_set(params)
        bw = fixed_bandwidth(S_G)
        mc = conditional_model(params, seed=SEEDS[0])          # as exp12 (no tag)
        mu = unconditional_model(params, seed=SEEDS[0])
        tape = NoiseTape(seed=777, dtype=torch.float32)
        cms = CMSampler(mc, PAPER_TS, tape, source="tape", dtype=torch.float32)
        dl = DistributionalLoss(S_G, bandwidth="fixed", bandwidth_value=bw, backend="fast")

        def sampler(x0, restart, t, draw, n=N_GEN):
            keys = [("fid", int(restart), int(t), int(draw), i) for i in range(n)]
            return cms(x0.reshape(1, -1), keys)
        return mu, sampler, dl, ZETA["10D"], {"dim_x": 9, "dim_y": 1, "bw": bw,
                                               "conditional": "trained CM (per-sample Jacobian)"}
    raise ValueError(setting)


# ---------------------------------------------------------------------------
# trajectory (trust_noise1, corrected protocol) with probes
# ---------------------------------------------------------------------------

def trajectory(mu, sampler, loss, zeta, restart, probe_t=PROBE_T):
    T = mu.diffusion_steps
    g0 = torch.Generator().manual_seed(0x5EED0000 ^ int(restart))
    x = torch.randn(1, mu.nfeatures, generator=g0, dtype=torch.float32)
    probes = {}
    for t in range(T - 1, 0, -1):
        x = x.detach().clone().requires_grad_(True)
        if t in probe_t:
            probes[t] = x.detach().clone()
        x_prev, pred_x0 = mu.sample_ddim_step(x, t, condition_x=None, device="cpu", eta=0.0)
        y = sampler(pred_x0, restart, t, -1)
        g, = torch.autograd.grad(loss(y), x, allow_unused=True)
        g = torch.zeros_like(x) if g is None else g
        upd = zeta * g.detach()
        with torch.no_grad():
            ref = (1 - mu.baralphas[t]).sqrt()
            upd = upd * torch.clamp(ref / (upd.norm() + 1e-12), max=1.0)
            x = x_prev.detach() - upd
        if not torch.isfinite(x).all() or float(x.abs().max()) > 50.0:
            break
    return probes


# ---------------------------------------------------------------------------
# one probe: G, g_i, J_i, then every estimator in closed form
# ---------------------------------------------------------------------------

def jacobians(y, x):
    """J (n, d_y, d_x) = dy_i/dx via batched vjp (falls back to a loop)."""
    n, d_y = y.shape
    eye = torch.eye(n * d_y).reshape(n * d_y, n, d_y)
    try:
        (J,) = torch.autograd.grad(y, x, grad_outputs=eye, is_grads_batched=True,
                                   retain_graph=True)
        return J.reshape(n, d_y, -1)
    except Exception:
        rows = []
        for i in range(n):
            for j in range(d_y):
                (r,) = torch.autograd.grad(y[i, j], x, retain_graph=True)
                rows.append(r.reshape(-1))
        return torch.stack(rows).reshape(n, d_y, -1)


def kcenter_mean2(Y, g, k, tape, key):
    """k/2 clusters; per cluster the center and one tape-random other member
    are differentiated, each carrying HALF the cluster's summed g."""
    kc = max(1, k // 2)
    centers, g_eff = select_kcenter(Y, g, kc, tape, key)
    Yd = Y.double()
    assign = torch.cdist(Yd, Yd[centers]).argmin(dim=1)
    u = tape.randn(key + ("m2",), (Y.shape[0],), dtype=torch.float64)
    pairs = []                                   # (index, weight-fraction, cluster)
    for c, r in enumerate(centers):
        members = [i for i in torch.nonzero(assign == c).reshape(-1).tolist() if i != r]
        if members:
            other = max(members, key=lambda i: float(u[i]))
            pairs += [(r, 0.5, c), (other, 0.5, c)]
        else:
            pairs.append((r, 1.0, c))
    return pairs, g_eff


def probe(x_t, t, mu, sampler, loss, restart, draws, tape, meta, n=N_GEN, ks=KS, rules=RULES):
    x_t = x_t.detach().clone().requires_grad_(True)
    _, x0 = mu.sample_ddim_step(x_t, t, condition_x=None, device="cpu", eta=0.0)
    y = sampler(x0, restart, t, 0, n)
    n, d_y = y.shape
    L = loss(y)
    (G,) = torch.autograd.grad(L, x_t, retain_graph=True)
    G = G.reshape(-1).double()
    yl = y.detach().clone().requires_grad_(True)
    (g,) = torch.autograd.grad(loss(yl), yl)
    g = g.double()                                            # (n, d_y)
    J = jacobians(y, x_t).double()                            # (n, d_y, d_x)
    h = torch.einsum("ijd,ij->id", J, g)                      # (n, d_x)
    Gsum = h.sum(0)
    rel_check = float((Gsum - G).norm() / (G.norm() + 1e-30))
    s = g.sum(0)
    hn, gn = h.norm(dim=1), g.norm(dim=1)
    ess_h = float(hn.sum() ** 2 / (hn ** 2).sum())
    ess_g = float(gn.sum() ** 2 / (gn ** 2).sum())
    # clusteredness of the y_i
    Yd = y.detach().double()
    c8, _ = select_kcenter(Yd, g, 8, tape, ("cl", restart, t))
    D = torch.cdist(Yd, Yd)
    assign = D[:, c8].argmin(dim=1)
    sil = []
    for i in range(n):
        same = (assign == assign[i]); same[i] = False
        a = float(D[i, same].mean()) if same.any() else 0.0
        b = min(float(D[i, assign == c].mean()) for c in range(len(c8)) if c != assign[i] and (assign == c).any())
        sil.append((b - a) / max(a, b, 1e-12))
    within = float(torch.stack([D[i, c8[assign[i]]] for i in range(n)]).mean())
    between = float(D[c8][:, c8][~torch.eye(len(c8), dtype=bool)].mean())
    ev = torch.linalg.eigvalsh(torch.cov(Yd.T).reshape(d_y, d_y)).clamp_min(0)
    eff_rank = float(ev.sum() ** 2 / ((ev ** 2).sum() + 1e-30))
    diag = {"rel_check": rel_check, "ess_h": ess_h, "ess_g": ess_g, "silhouette": sum(sil) / n,
            "within_between": within / (between + 1e-12), "eff_rank": eff_rank,
            "G_norm": float(G.norm()), "gn_p90_over_med": float(gn.quantile(0.9) / gn.median())}
    rows = []
    for k in ks:
        for rule in rules:
            cos_x, cos_y, rn, ndiff, sign = [], [], [], [], []
            for dr in range(draws):
                key = ("sel", restart, t, k, dr)
                if rule.startswith("soft_"):
                    _, sel_rule, tm = rule.split("_")
                    tau = float(tm[1:]) * meta["bw"]
                    if sel_rule == "uniform":
                        idx, _ = select_uniform(g, k, tape, key)
                    elif sel_rule == "stratified":
                        idx, _ = select_stratified(Yd, g, k, tape, key)
                    else:
                        idx, _ = select_kcenter(Yd, g, k, tape, key)
                    g_eff = soft_aggregate(Yd, g, idx, tau) if len(idx) < n else g[idx]
                    Gh = torch.einsum("ijd,ij->d", J[idx], g_eff)
                    sh = g_eff.sum(0)
                    nd = len(idx)
                elif rule == "kcenter_mean2":
                    pairs, g_eff = kcenter_mean2(Yd, g, k, tape, key)
                    Gh = sum(w * torch.einsum("jd,j->d", J[i], g_eff[c]) for i, w, c in pairs)
                    sh = g_eff.sum(0)
                    nd = len(set(i for i, _, _ in pairs))
                else:
                    if rule == "uniform":
                        idx, g_eff = select_uniform(g, k, tape, key)
                    elif rule == "importance":
                        idx, g_eff = select_importance(g, k, tape, key)
                    elif rule == "stratified":
                        idx, g_eff = select_stratified(Yd, g, k, tape, key)
                    else:
                        idx, g_eff = select_kcenter(Yd, g, k, tape, key)
                    Gh = torch.einsum("ijd,ij->d", J[idx], g_eff)
                    sh = g_eff.sum(0)
                    nd = len(idx)
                cos_x.append(float(Gh @ G / (Gh.norm() * G.norm() + 1e-30)))
                cos_y.append(float(sh @ s / (sh.norm() * s.norm() + 1e-30)))
                rn.append(float(Gh.norm() / (G.norm() + 1e-30)))
                sign.append(float(torch.sign(Gh[0]) == torch.sign(G[0])))
                ndiff.append(nd)
            cx, cy, rn_ = torch.tensor(cos_x), torch.tensor(cos_y), torch.tensor(rn)
            rows.append({"setting": meta["setting"], "restart": restart, "t": t, "bucket": bucket(t),
                         "rule": rule, "k": k, "n": n, "n_diff": sum(ndiff) / len(ndiff),
                         "ratio_draws": rn,
                         "cos_x": float(cx.mean()), "cos_x_std": float(cx.std()),
                         "cos_y": float(cy.mean()), "cos_y_std": float(cy.std()),
                         "sign_agree": sum(sign) / len(sign),
                         "rel_norm_err": float((rn_ - 1).abs().mean()), "rel_norm_std": float(rn_.std()),
                         "cos_x_draws": cos_x, "cos_y_draws": cos_y, **diag})
    return rows


def run_setting(setting, trajectories, draws, out_dir, n=N_GEN, ks=KS, rules=RULES, tag=""):
    mu, sampler, loss, zeta, meta = build(setting)
    meta["setting"], meta["zeta"], meta["n"] = setting + tag, zeta, n
    tape = NoiseTape(seed=4242)
    rows, t0 = [], time.perf_counter()
    for r in range(trajectories):
        probes = trajectory(mu, sampler, loss, zeta, r)     # trajectory itself always at n=128
        for t, x_t in sorted(probes.items(), reverse=True):
            rows += probe(x_t, t, mu, sampler, loss, r, draws, tape, meta, n=n, ks=ks, rules=rules)
        print(f"{setting}{tag} restart {r}: {len(probes)} probes, {time.perf_counter() - t0:.0f}s", flush=True)
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{setting}{tag}.json").write_text(json.dumps({"meta": meta, "rows": rows}, default=str))
    return rows


def report_r7(out_dir):
    """[B-R7] compact table: cos_x med/p10, rel-norm err, NORM INFLATION ||G_hat||/||G|| med/p90."""
    lines = ["# B-R7 -- soft assignment vs hard rules, 10D, corrected protocol (zeta* = 4.0)\n",
             "Pooled over trajectories x probes x 20 draws.  inflation = ||G_hat|| / ||G|| (median / p90; "
             "> 1 = the aggregate is LARGER than the full gradient -- what killed hard kcenter on SD).  "
             "tau = tau_mult x the MMD median bandwidth.\n"]
    out = []
    for p in sorted(out_dir.glob("*.json")):
        d = json.loads(p.read_text()); rows, meta = d["rows"], d["meta"]
        lines.append(f"\n## {meta['setting']}  (n = {meta['n']}, {len(set(r['restart'] for r in rows))} trajectories, "
                     f"{len(set((r['restart'], r['t']) for r in rows))} probes, bw = {meta['bw']:.3g})\n")
        lines.append("| rule | k | n_diff | cos_x med | cos_x p10 | rel_norm med | inflation med | inflation p90 | inflation p99 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for k in sorted(set(r["k"] for r in rows)):
            for rule in RULES_R7:
                sel = [r for r in rows if r["rule"] == rule and r["k"] == k]
                if not sel:
                    continue
                cx = [c for r in sel for c in r["cos_x_draws"]]
                ra = [c for r in sel for c in r["ratio_draws"]]
                rec = {"setting": meta["setting"], "n": meta["n"], "rule": rule, "k": k,
                       "n_diff": sum(r["n_diff"] for r in sel) / len(sel),
                       "cos_x_med": q(cx, .5), "cos_x_p10": q(cx, .1),
                       "rel_norm_med": q([abs(v - 1) for v in ra], .5),
                       "infl_med": q(ra, .5), "infl_p90": q(ra, .9), "infl_p99": q(ra, .99)}
                out.append(rec)
                lines.append(f"| {rule} | {k} | {rec['n_diff']:.0f} | {rec['cos_x_med']:.3f} | {rec['cos_x_p10']:.3f} | "
                             f"{rec['rel_norm_med']:.3f} | {rec['infl_med']:.2f} | {rec['infl_p90']:.2f} | {rec['infl_p99']:.2f} |")
    (HERE / "fidelity_r7_tables.md").write_text("\n".join(lines) + "\n")
    with open(HERE / "fidelity_r7_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"{len(out)} rows -> fidelity_r7_rows.csv, fidelity_r7_tables.md")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def report(out_dir):
    data = {p.stem: json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))}
    if not data:
        print("no runs"); return
    lines = ["# B-R6 stage 1 -- gradient fidelity of subset backprop at n = 128\n",
             "cos_x = cosine to the full 128-sample x-gradient (meaningful when dim x > 1; in the "
             "dimY settings dim x = 1 so cos_x is a sign and `sign` is reported); cos_y = cosine of the "
             "aggregated output gradient s_hat = sum_S w_i g_i to s = sum_i g_i -- with the oracle's common "
             "Jacobian this IS the x-gradient fidelity.  Aggregated over trajectories x probes in the bucket "
             "x 20 selection draws (median / p10).  rel_norm = | ||G_hat||/||G|| - 1 |, std = across draws.  "
             "Decision (pre-registered): median cos >= 0.9 AND p10 cos >= 0.8 in both dimY settings "
             "(cos_y), kcenter additionally median cos_x >= 0.9 in 10D.\n"]
    out_rows, pass_tab = [], {}
    for setting, dset in data.items():
        rows, meta = dset["rows"], dset["meta"]
        lines.append(f"\n## {setting}  (dim x={meta['dim_x']}, dim y={meta['dim_y']}, {meta['conditional']}, "
                     f"zeta={meta['zeta']:.4g}, bw={meta['bw']:.3g}; {len(set(r['restart'] for r in rows))} "
                     f"trajectories, {len(set((r['restart'], r['t']) for r in rows))} probes)\n")
        d = [r for r in rows if r["rule"] == "uniform" and r["k"] == 8]
        lines.append("Redundancy diagnostics (per probe, median [p10, p90]): "
                     + ", ".join(f"{k}={q([r[k] for r in d], .5):.3g} [{q([r[k] for r in d], .1):.3g}, {q([r[k] for r in d], .9):.3g}]"
                                 for k in ("ess_h", "ess_g", "gn_p90_over_med", "silhouette", "within_between", "eff_rank"))
                     + f"; max |sum h_i - G|/|G| = {max(r['rel_check'] for r in d):.1e}\n")
        lines.append("| bucket | rule | k | n_diff | cos_x med | cos_x p10 | sign | cos_y med | cos_y p10 | cos_y std | rel_norm med | rel_norm std | pass(dimY: cos_y; 10D: cos_x) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for b in ("early", "mid", "late", "all"):
            for rule in RULES:
                for k in KS:
                    sel = [r for r in rows if r["rule"] == rule and r["k"] == k and (b == "all" or r["bucket"] == b)]
                    if not sel:
                        continue
                    cx = [c for r in sel for c in r["cos_x_draws"]]
                    cy = [c for r in sel for c in r["cos_y_draws"]]
                    key_cos = cy if setting.startswith(("dimy", "nuis")) else cx
                    ok = q(key_cos, .5) >= 0.9 and q(key_cos, .1) >= 0.8
                    rec = {"setting": setting, "bucket": b, "rule": rule, "k": k,
                           "n_diff": sum(r["n_diff"] for r in sel) / len(sel),
                           "cos_x_med": q(cx, .5), "cos_x_p10": q(cx, .1),
                           "sign_agree": sum(r["sign_agree"] for r in sel) / len(sel),
                           "cos_y_med": q(cy, .5), "cos_y_p10": q(cy, .1),
                           "cos_y_std": sum(r["cos_y_std"] for r in sel) / len(sel),
                           "cos_x_std": sum(r["cos_x_std"] for r in sel) / len(sel),
                           "rel_norm_med": q([r["rel_norm_err"] for r in sel], .5),
                           "rel_norm_std": sum(r["rel_norm_std"] for r in sel) / len(sel),
                           "pass": ok}
                    out_rows.append(rec)
                    if b == "all":
                        pass_tab[(rule, k, setting)] = ok
                    lines.append(f"| {b} | {rule} | {k} | {rec['n_diff']:.0f} | {rec['cos_x_med']:.3f} | {rec['cos_x_p10']:.3f} | "
                                 f"{rec['sign_agree']:.2f} | {rec['cos_y_med']:.3f} | {rec['cos_y_p10']:.3f} | {rec['cos_y_std']:.3f} | "
                                 f"{rec['rel_norm_med']:.3f} | {rec['rel_norm_std']:.3f} | {'PASS' if ok else 'fail'} |")
    lines.append("\n## Stage-2 admission (all buckets pooled)\n")
    lines.append("| rule | k | dimy8 | dimy16 | nuis8 | nuis16 | 10D (cos_x) | admitted (dimY rule) | admitted (nuis addendum) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for rule in RULES:
        for k in KS:
            v = {s: pass_tab.get((rule, k, s)) for s in ("dimy8", "dimy16", "nuis8", "nuis16", "10D")}
            jac_ok = rule not in ("kcenter", "kcenter_mean2") or bool(v["10D"])
            adm = bool(v["dimy8"]) and bool(v["dimy16"]) and jac_ok
            adm_n = bool(v["nuis8"]) and bool(v["nuis16"]) and jac_ok
            lines.append(f"| {rule} | {k} | {v['dimy8']} | {v['dimy16']} | {v['nuis8']} | {v['nuis16']} | {v['10D']} | {'YES' if adm else 'no'} | {'YES' if adm_n else 'no'} |")
    (HERE / "fidelity_tables.md").write_text("\n".join(lines) + "\n")
    with open(HERE / "fidelity_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(data), figsize=(4.5 * len(data), 3.6), squeeze=False)
        for ax, (setting, dset) in zip(axes[0], data.items()):
            key = "cos_y_draws" if setting.startswith(("dimy", "nuis")) else "cos_x_draws"
            for rule in RULES:
                med = [q([c for r in dset["rows"] if r["rule"] == rule and r["k"] == k for c in r[key]], .5) for k in KS]
                p10 = [q([c for r in dset["rows"] if r["rule"] == rule and r["k"] == k for c in r[key]], .1) for k in KS]
                ax.plot(KS, med, marker="o", label=rule)
                ax.plot(KS, p10, ls=":", color=ax.lines[-1].get_color())
            ax.axhline(0.9, color="k", lw=0.6); ax.axhline(0.8, color="k", lw=0.6, ls="--")
            ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS)
            ax.set_ylim(-0.1, 1.05); ax.set_xlabel("k differentiated (n = 128)")
            ax.set_ylabel("cos_y" if setting.startswith(("dimy", "nuis")) else "cos_x")
            ax.set_title(f"{setting}: median (solid) / p10 (dotted)")
        axes[0][0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(HERE / "fidelity_cos_vs_k.png", dpi=130)
    except Exception as e:                       # pragma: no cover
        print("figure skipped:", e)
    print(f"{len(out_rows)} rows -> fidelity_rows.csv, fidelity_tables.md, fidelity_cos_vs_k.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "report", "report_r7"])
    ap.add_argument("--r7", action="store_true", help="[B-R7] soft-assignment rule set, k in {8,16,32}")
    ap.add_argument("--n", type=int, default=N_GEN, help="generated samples per probe (trajectory stays at 128)")
    ap.add_argument("--setting", default="dimy8", choices=["dimy8", "dimy16", "nuis8", "nuis16", "10D"])
    ap.add_argument("--trajectories", type=int, default=10)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--dir", default="fidelity_runs")
    a = ap.parse_args()
    out_dir = HERE / a.dir
    if a.mode == "report":
        report(out_dir)
    elif a.mode == "report_r7":
        report_r7(out_dir)
    elif a.r7:
        ks = [k for k in (8, 16, 32) if k < a.n]
        run_setting(a.setting, a.trajectories, a.draws, out_dir, n=a.n, ks=ks, rules=RULES_R7, tag=f"_n{a.n}")
    else:
        run_setting(a.setting, a.trajectories, a.draws, out_dir)


if __name__ == "__main__":
    main()
