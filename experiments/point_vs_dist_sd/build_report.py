"""
build_report.py — THE deliverable: arms x metrics table + overlaid PC1 histograms.
Pure CPU/numpy (CLIP text features are pre-saved by setup_targets.py).

Reads, for each arm directory: metrics.json (eval_final.py), npy/eval_clip_mlgd_f.npy
(the 2000 fresh eval embeddings) and npy/eval_clip_regular.npy (that arm's own
reference), plus profile.json / metrics_partial.json for cost. Targets and CLIP
text features come from cache/<scenario>/.

COMMON REFERENCE. Arms do NOT share a "regular" path: the PGD arm's reference is
f_phi(x0) (the untouched source scribble, no diffusion at all) while the diffusion
arms' reference is f_phi(unguided DDIM trajectory). Deltas against those are not
comparable, so every arm is ALSO scored against one common reference —
f_phi(x0), the do-nothing baseline every arm starts from (auto-detected as the
reference of the PGD arms, or supplied with --common_ref).

VERIFICATION. A run is EXCLUDED from the table (and listed with its reason) when
its target set does not hash-match the scenario cache or its eval count differs
from the modal one — this is what catches smoke leftovers.

Usage:
  python build_report.py --runs ../../output/pvd_B --scenario B [--with_latent]
  python build_report.py                     # default nested layout runs/<scenario>/<arm>
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXP_DIR, band_for, mean_ci95, p_male, pc1_stats, wilson_ci95

try:
    import torch
except ImportError:
    torch = None


def mmd_np(x, y):
    """The pipeline's compute_mmd (unbiased U-statistic, median heuristic, alpha=1)."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    d2 = lambda a, b: (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    ss = min(1000, len(x), len(y))
    d = d2(x[:ss], y[:ss]); d = d[d > 0]
    bw = np.sqrt(np.median(d) / 2) if len(d) else 1.0
    k = lambda a, b: np.exp(-d2(a, b) / (2 * bw * bw))
    n, m = len(x), len(y)
    Kxx, Kyy, Kxy = k(x, x), k(y, y), k(x, y)
    return float(np.sqrt(abs((Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
                             + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
                             - 2 * Kxy.sum() / (n * m)) + 1e-8))


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def cost_of(run_dir):
    out = {}
    pf = os.path.join(run_dir, "profile.json")
    if os.path.exists(pf) and os.path.getsize(pf) > 0:
        s = json.load(open(pf)).get("summary", {})
        out["s_step"] = s.get("mean_total_s")
        out["vram_gb"] = (s.get("max_memory_allocated_mb") or 0) / 1024 or None
    for f in ("metrics.json", "metrics_partial.json"):
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            m = json.load(open(p))
            out.setdefault("wall_s", m.get("optimization_time_sec"))
            a = m.get("args", {})
            out.setdefault("n_steps_opt", len(m.get("steps") or []))
            out.setdefault("n_cond", a.get("num_variations", 1))
            out.setdefault("args", a)
    return out


def trace_stats(steps, args):
    """P-C1: does the guidance objective trend DOWN over the run?
    delta = mean(last k) - mean(first k); criterion delta <= -0.03 AND corr <= -0.30.
    Window k is fixed by the registration (see below), not chosen per run."""
    if not steps or "mmd_loss" not in steps[0]:
        return None
    L = np.array([s["mmd_loss"] for s in steps])
    if len(L) < 6:
        return None
    # P-C1 window, EXACTLY as registered in HYPOTHESIS.md ("mean(last 10) - mean(first 10)"):
    # k = 10 whenever the run has >= 20 steps; shorter runs (only the smoke) fall back to
    # max(3, n//6) because first-10/last-10 would overlap. Single-valued, never per-run.
    k = 10 if len(L) >= 20 else max(3, len(L) // 6)
    delta = float(L[-k:].mean() - L[:k].mean())
    corr = float(np.corrcoef(np.arange(len(L)), L)[0, 1])
    c = np.array([s.get("correction_norm_raw", np.nan) for s in steps])
    ap = np.array([s.get("correction_norm_applied", np.nan) for s in steps])
    lo, hi = band_for(args)                      # the ARM's own measured band
    sd_step = float(np.diff(L).std() / np.sqrt(2))
    se_delta = sd_step * np.sqrt(2.0 / k)
    return {"n": len(L), "first": float(L[:k].mean()), "last": float(L[-k:].mean()),
            "delta": delta, "corr": corr, "pc1_pass": bool(delta <= -0.03 and corr <= -0.30),
            "corr_med": float(np.nanmedian(c)), "applied_med": float(np.nanmedian(ap)),
            "clipped_frac": float(np.nanmean(ap < c - 1e-6)),
            "band": (lo, hi), "in_band": float(np.nanmean((ap >= lo) & (ap <= hi))),
            "se_delta": se_delta, "thresh_se": (0.03 / se_delta if se_delta > 0 else float("nan"))}


def collect(runs_root, scenario, cache_dir, flat):
    pattern = os.path.join(runs_root, "*") if flat else os.path.join(runs_root, scenario, "*")
    tc = torch.load(os.path.join(cache_dir, "targets_clip.pt"), map_location="cpu")
    targets = tc["all_clip_embeddings"].numpy()
    cache_hash = sha(targets)
    rows, excluded, trend_only = {}, {}, {}
    for d in sorted(glob.glob(pattern)):
        arm = os.path.basename(d)
        npy = os.path.join(d, "npy", "eval_clip_mlgd_f.npy")
        if not os.path.isdir(d) or not os.path.exists(npy):
            continue
        tp = os.path.join(d, "target_clip_embeddings.pt")
        if os.path.exists(tp):
            th = sha(torch.load(tp, map_location="cpu")["all_clip_embeddings"].numpy())
            if th != cache_hash:
                excluded[arm] = f"target set {th} != scenario cache {cache_hash} (stale/smoke run)"
                continue
        rows[arm] = {"dir": d, "gen": np.load(npy),
                     "ref": np.load(os.path.join(d, "npy", "eval_clip_regular.npy")),
                     "metrics": json.load(open(os.path.join(d, "metrics.json"))),
                     "cost": cost_of(d)}
    if rows:
        modal = max({len(r["gen"]) for r in rows.values()},
                    key=lambda n: sum(len(r["gen"]) == n for r in rows.values()))
        for arm in list(rows):
            if len(rows[arm]["gen"]) != modal:
                why = ("gate run (deliberately short eval)" if "gate" in arm
                       else "smoke run" if "smoke" in arm else "incomplete/partial eval")
                excluded[arm] = (f"eval n={len(rows[arm]['gen'])} != {modal} — {why}; "
                                 "not comparable in the metrics table")
                trend_only[arm] = rows[arm]
                rows.pop(arm)
    return rows, excluded, trend_only, targets, tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(EXP_DIR, "runs"))
    ap.add_argument("--scenario", default=None, help="B or A; with --runs pointing at the arms")
    ap.add_argument("--cache", default=os.path.join(EXP_DIR, "cache"))
    ap.add_argument("--out", default=os.path.join(EXP_DIR, "RESULTS.md"))
    ap.add_argument("--common_ref", default=None,
                    help="arm whose eval_clip_regular.npy is THE common reference "
                         "(default: a pgd arm = f_phi(x0))")
    ap.add_argument("--with_latent", action="store_true")
    args = ap.parse_args()
    scen = args.scenario or "B"
    flat = args.scenario is not None
    cache_dir = os.path.join(args.cache, scen)
    os.makedirs(os.path.join(EXP_DIR, "figures"), exist_ok=True)

    rows, excluded, trend_only, targets, tc = collect(args.runs, scen, cache_dir, flat)
    if not rows:
        print("no usable runs"); return
    sizes = tc["group_sizes"]
    anchors = (np.concatenate([targets[:sizes[0]], targets[-sizes[-1]:]])
               if len(sizes) > 1 else targets)
    txt = torch.load(os.path.join(cache_dir, "text_features.pt"), map_location="cpu")["features"].numpy()

    ref_arm = args.common_ref or next((a for a in rows if a.startswith("pgd")), None)
    common_ref = rows[ref_arm]["ref"] if ref_arm else None
    ref_mmd = mmd_np(common_ref, targets) if common_ref is not None else float("nan")

    for arm, r in rows.items():
        a = r["cost"].get("args", {})
        st = None
        for f in ("metrics.json", "metrics_partial.json"):
            fp = os.path.join(r["dir"], f)
            if os.path.exists(fp):
                st = json.load(open(fp)).get("steps") or st
        r["trace"] = trace_stats(st, a)
        r["cfg"] = ("zeta %.0f, tau %s" % (a["base_zeta"], a.get("trust_noise", 0) or "off")
                    if "base_zeta" in a else
                    "eps %s/255, lr %.4g" % (a.get("eps_255", "?"), a.get("lr", float("nan"))))
        pm = p_male(r["gen"], txt)
        m, lo, hi = mean_ci95(pm)
        frac, flo, fhi = wilson_ci95(int((pm > 0.5).sum()), len(pm))
        pc = pc1_stats(targets, r["gen"], anchors)
        r.update({"mmd": mmd_np(r["gen"], targets), "own_ref_mmd": mmd_np(r["ref"], targets),
                  "p_male": (m, lo, hi), "frac_male": (frac, flo, fhi),
                  "extreme": float(((pm < 0.2) | (pm > 0.8)).mean()), "pc": pc,
                  "own_ref_hash": sha(r["ref"])})

    L = [f"# Results — Scenario {scen}: point vs distributional targets on SD", "",
         "*Auto-generated by build_report.py. Pre-registered predictions: HYPOTHESIS.md. "
         "All numbers from 2000 fresh sprinter samples per arm, identical eval seeds, "
         "ControlNet 0.5, neutral prompt.*", "",
         "## Verification", ""]
    seeds = {arm: r["metrics"].get("eval_seed_base") for arm, r in rows.items()}
    cns = {arm: r["metrics"].get("eval_cn_scale") for arm, r in rows.items()}
    ns = {arm: len(r["gen"]) for arm, r in rows.items()}
    L += [f"- eval seed base: {set(seeds.values())} (identical across arms: "
          f"{len(set(seeds.values())) == 1})",
          f"- eval cn_scale: {set(cns.values())}; eval n: {set(ns.values())}",
          f"- target set: all arms hash-match the scenario cache ({sha(targets)}, "
          f"{targets.shape[0]} embeddings, groups {tc['group_names']} {sizes})"]
    if excluded:
        L += ["- **excluded runs:**"] + [f"  - `{a}`: {why}" for a, why in excluded.items()]
    else:
        L += ["- excluded runs: none"]
    L += ["", "## Reference sets (they are NOT the same — see README 'Two references')", "",
          "| arm | its own 'regular' path | MMD(ref -> G) | ref hash |", "|---|---|---|---|"]
    for arm, r in rows.items():
        what = "f_phi(x0), no diffusion" if arm.startswith("pgd") else "f_phi(unguided DDIM scribble)"
        L.append(f"| {arm} | {what} | {r['own_ref_mmd']:.4f} | `{r['own_ref_hash']}` |")
    L += ["", f"**Common reference** = `{ref_arm}`'s regular set = f_phi(x0), MMD -> G = "
          f"**{ref_mmd:.4f}** — the do-nothing baseline every arm starts from.", ""]

    L += ["## Main table", "",
          "| arm | config | MMD -> G | Δ vs common ref | Δ vs own ref | p(male) [95% CI] | frac p<.2 or >.8 | "
          "PC1 mean (target) | PC1 var (target) | **PC1 var ratio** | n_cond | s/step | wall | VRAM |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, r in rows.items():
        c = r["cost"]; pc = r["pc"]
        sstep = f"{c['s_step']:.1f}" if c.get("s_step") else "—"
        vram = f"{c['vram_gb']:.1f} GB" if c.get("vram_gb") else "—"
        wall = f"{c['wall_s']/60:.0f} min" if c.get("wall_s") else "—"
        L.append(f"| {arm} | {r['cfg']} | {r['mmd']:.4f} | {ref_mmd - r['mmd']:+.4f} | "
                 f"{r['own_ref_mmd'] - r['mmd']:+.4f} | "
                 f"{r['p_male'][0]:.3f} [{r['p_male'][1]:.3f}, {r['p_male'][2]:.3f}] | "
                 f"{r['extreme']:.3f} | {pc['gen_mean']:+.3f} ({pc['target_mean']:+.3f}) | "
                 f"{pc['gen_var']:.4f} ({pc['target_var']:.4f}) | **{pc['var_ratio']:.3f}** | "
                 f"{c.get('n_cond','—')} | {sstep} | {wall} | {vram} |")
    L += ["", "Δ > 0 = better than the reference (lower MMD). PC1 is fitted on the target "
          "anchor groups (the pipeline's PCA convention) and both sets are projected onto it.", ""]

    for arm, r in trend_only.items():
        a = r["cost"].get("args", {})
        st = None
        for f in ("metrics.json", "metrics_partial.json"):
            fp = os.path.join(r["dir"], f)
            if os.path.exists(fp):
                st = json.load(open(fp)).get("steps") or st
        r["trace"] = trace_stats(st, a)
        r["cfg"] = ("zeta %.0f, tau %s" % (a["base_zeta"], a.get("trust_noise", 0) or "off")
                    if "base_zeta" in a else "-")
    tr = {a: r["trace"] for a, r in list(rows.items()) + list(trend_only.items()) if r.get("trace")}
    # zeta=0 rows are the DRIFT CONTROL: they take no guidance step, so P-C1 (a criterion on
    # the guidance's effect) does not apply to them; their trend IS the drift baseline.
    if tr:
        L += ["## Guidance-objective trend (P-C1: delta <= -0.03 AND corr(step,loss) <= -0.30)", "",
              "| arm | config | steps | first | last | delta | corr | applied med | arm band | in-band | clipped | -0.03 in SE | **P-C1** |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a, t in tr.items():
            src = rows.get(a) or trend_only[a]
            note = "" if a in rows else " *(trend only)*"
            is_ctrl = "zeta 0," in src["cfg"]
            pw = "" if t["thresh_se"] >= 2 else " *(underpowered)*"
            if is_ctrl:
                pw = ""
            L.append(f"| {a}{note} | {src['cfg']} | {t['n']} | {t['first']:.4f} | {t['last']:.4f} | "
                     f"{t['delta']:+.4f} | {t['corr']:+.3f} | {t['applied_med']:.2f} | "
                     f"{t['band'][0]}-{t['band'][1]} | {t['in_band']:.0%} | {t['clipped_frac']:.0%} | "
                     f"{t['thresh_se']:.1f} | "
                     f"{'*control (n/a)*' if is_ctrl else ('**PASS**' if t['pc1_pass'] else 'FAIL')}{pw} |")
        L.append("")

    # PGD eps sweep
    pgd = {a: r for a, r in rows.items() if a.startswith("pgd")}
    if pgd:
        L += ["## Arm 1 eps sweep (AdvI2I grid) — the attack's OWN objective", "",
              "| arm | eps | final point loss L(x*) | L(x0) fresh-sample | improved? |", "|---|---|---|---|---|"]
        ys = torch.load(os.path.join(cache_dir, "point_target.pt"), map_location="cpu")["y_star"].numpy()
        l0 = float(((rows[ref_arm]["ref"] - ys) ** 2).sum(1).mean())
        for arm, r in pgd.items():
            a = r["cost"].get("args", {})
            fl = r["metrics"].get("final_loss", a.get("final_loss"))
            lstar = float(((r["gen"] - ys) ** 2).sum(1).mean())
            L.append(f"| {arm} | {a.get('eps_255','?')}/255 | "
                     f"{fl:.4f} (train) / {lstar:.4f} (fresh) | {l0:.4f} | "
                     f"{'YES' if lstar < l0 else '**NO**'} |")
        L.append("")

    # histograms
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5.5))
        any_r = next(iter(rows.values()))
        ax.hist(any_r["pc"]["target_proj"], bins=40, density=True, alpha=0.35, color="gray",
                label=f"target G_bal (var {any_r['pc']['target_var']:.3f})")
        if common_ref is not None:
            rp = pc1_stats(targets, common_ref, anchors)
            ax.hist(rp["gen_proj"], bins=40, density=True, histtype="step", lw=2, ls=":",
                    color="black", label=f"common ref f_phi(x0) (ratio {rp['var_ratio']:.2f})")
        for (arm, r), c in zip(rows.items(), ["crimson", "royalblue", "seagreen", "orange",
                                              "mediumpurple", "sienna"]):
            ax.hist(r["pc"]["gen_proj"], bins=40, density=True, histtype="step", lw=2, color=c,
                    label=f"{arm} (ratio {r['pc']['var_ratio']:.2f})")
        ax.set_xlabel("CLIP PC1 (fitted on target anchor groups)"); ax.set_ylabel("density")
        ax.set_title(f"Scenario {scen}: PC1 of 2000 fresh samples vs target")
        ax.legend(fontsize=8)
        fp = os.path.join(EXP_DIR, "figures", f"pc1_{scen}.png")
        fig.savefig(fp, dpi=130, bbox_inches="tight"); plt.close(fig)
        L += [f"![PC1 scenario {scen}](figures/pc1_{scen}.png)", ""]
    except ImportError:
        L += ["(matplotlib unavailable — histograms skipped)", ""]

    for extra in (f"findings_{scen}.md", f"closeout_{scen}.md"):
        fx = os.path.join(EXP_DIR, extra)
        if os.path.exists(fx):
            L += [open(fx).read()]
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    json.dump({a: {k: v for k, v in r.items() if k not in ("gen", "ref", "pc", "metrics")}
               for a, r in rows.items()},
              open(os.path.join(EXP_DIR, f"results_{scen}.json"), "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
