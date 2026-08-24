"""Held-out Pareto (offset 1000, 100 restarts): failure-penalised mean exact L2 vs conditional
calls per setting, from verification/heldout_runs/*.json. Writes pareto.md and pareto.png.

Arms: baseline no-LGD/none n in {4,8,16,32,64,96}; LGD/none n in {8,32};
trust_noise1 / sqrt_floor / relclip2 (no-LGD/none) n in {4,8,16,32}.
Error bars: 95% normal CI of the mean over the 100 restarts (1.96 * sd / sqrt(R)).
Frontier: non-dominated points (no other point with calls <= and score <=, one strictly).

Run: /Users/stolk/miniconda3/bin/python experiments/model-optimization/report_tools/pareto.py
"""
import glob, json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "verification", "heldout_runs")
SETTINGS = ["2D", "5D", "10D"]
ARMS = [  # (label, spatial, temporal, candidate, ns, marker, colour)
    ("baseline no-LGD/none", "no_lgd", "none", "baseline", [4, 8, 16, 32, 64, 96], "o", "#555555"),
    ("LGD/none (baseline)", "lgd", "none", "baseline", [8, 32], "s", "#1f77b4"),
    ("trust_noise1 (promoted)", "no_lgd", "none", "trust_noise1", [4, 8, 16, 32], "^", "#d62728"),
    ("sqrt_floor (conditional)", "no_lgd", "none", "sqrt_floor", [4, 8, 16, 32], "D", "#2ca02c"),
    ("relclip2 (2D-only)", "no_lgd", "none", "relclip2", [4, 8, 16, 32], "v", "#ff7f0e"),
]
PENALTY = 2.0


def load(setting, n, spatial, temporal, cand):
    p = os.path.join(RUNS, f"{setting}_n{n}_{spatial}_{temporal}_{cand}_tape_off1000.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    scores = [min(float(s), PENALTY) if s == s else PENALTY for s in d["scores"]]
    R = len(scores)
    mean = sum(scores) / R
    sd = math.sqrt(sum((s - mean) ** 2 for s in scores) / (R - 1))
    calls = int(round(d["cm_samples_mean"]))
    return dict(setting=setting, n=n, spatial=spatial, temporal=temporal, cand=cand, score=d["score"],
                mean=mean, ci=1.96 * sd / math.sqrt(R), calls=calls, R=R,
                success=d["success_rate"], div=d["diverged"], mmd2=d.get("mmd2_eval_mean"))


def frontier(points):
    front = []
    for p in points:
        dominated = any((q["calls"] <= p["calls"] and q["score"] <= p["score"]) and
                        (q["calls"] < p["calls"] or q["score"] < p["score"]) for q in points)
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p["calls"])


fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
md = ["# Held-out Pareto: failure-penalised mean exact L2 vs conditional calls (Agent 7)",
      "",
      "Source: `verification/heldout_runs/*.json` (Agent 6; offset 1000, 100 restarts per cell, float32, "
      "engine path `estimator/engine_runner.py`, same-node pairs). Score = failure-penalised mean exact GMM L2 "
      "(cap 2.0); CI = 95% normal interval of the mean over restarts (not the paired CI -- paired diffs and "
      "permutation p are in `verification/heldout_tables.md`). Calls = conditional-model samples per restart "
      "(99 steps x M_t x n). Frontier = non-dominated points among the arms plotted. Figure: `pareto.png`.",
      ""]
summary = {}
for ax, setting in zip(axes, SETTINGS):
    pts = []
    for label, sp, tm, cand, ns, mk, col in ARMS:
        arm_pts = [load(setting, n, sp, tm, cand) for n in ns]
        arm_pts = [p for p in arm_pts if p]
        for p in arm_pts:
            p["label"] = label
        pts += arm_pts
        xs = [p["calls"] for p in arm_pts]
        ys = [p["score"] for p in arm_pts]
        es = [p["ci"] for p in arm_pts]
        ax.errorbar(xs, ys, yerr=es, marker=mk, color=col, linestyle="-", linewidth=1.0,
                    markersize=6, capsize=2, label=label, alpha=0.95)
        for p in arm_pts:
            ax.annotate(f"n={p['n']}", (p["calls"], p["score"]), textcoords="offset points",
                        xytext=(4, 4), fontsize=6.5, color=col)
    fr = frontier(pts)
    ax.plot([p["calls"] for p in fr], [p["score"] for p in fr], color="black", linestyle="--",
            linewidth=1.2, zorder=1, label="frontier")
    ax.scatter([p["calls"] for p in fr], [p["score"] for p in fr], s=120, facecolors="none",
               edgecolors="black", linewidths=1.2, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("conditional samples per restart (log)")
    ax.set_ylabel("failure-penalised mean exact L2 (held-out, R=100)")
    ax.set_title(f"{setting}: held-out Pareto (offset 1000)")
    ax.grid(True, which="both", alpha=0.25)
    summary[setting] = (pts, fr)

axes[0].legend(fontsize=7, loc="upper right")
fig.tight_layout()
png = os.path.join(ROOT, "pareto.png")
fig.savefig(png, dpi=150)
print("wrote", png)

for setting in SETTINGS:
    pts, fr = summary[setting]
    md.append(f"## {setting}")
    md.append("")
    md.append("| arm | candidate | n | calls | score | 95% CI of mean | success | diverged | mmd2_eval | frontier |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    fr_ids = {(p["spatial"], p["cand"], p["n"]) for p in fr}
    for p in sorted(pts, key=lambda p: (p["calls"], p["score"])):
        star = "**yes**" if (p["spatial"], p["cand"], p["n"]) in fr_ids else ""
        md.append(f"| {p['spatial']}/{p['temporal']} | {p['cand']} | {p['n']} | {p['calls']} | {p['score']:.3f} | "
                  f"+/-{p['ci']:.3f} | {100*p['success']:.0f}% | {p['div']} | {p['mmd2']:.3f} | {star} |")
    md.append("")
    md.append("Frontier (increasing calls): " + "; ".join(
        f"{p['cand']}@n={p['n']} ({p['spatial']}, {p['calls']} calls, {p['score']:.3f})" for p in fr))
    md.append("")

# compact cross-setting table
md.append("## Compact table (score; calls in header)")
md.append("")
md.append("| setting | arm | n=4 (396) | n=8 (792) | n=16 (1584) | n=32 (3168) | n=64 (6336) | n=96 (9504) | LGD n=8 (2376) | LGD n=32 (9504) |")
md.append("|---|---|---|---|---|---|---|---|---|---|")
for setting in SETTINGS:
    pts, fr = summary[setting]
    fr_ids = {(p["spatial"], p["cand"], p["n"]) for p in fr}
    def cell(sp, cand, n):
        for p in pts:
            if p["spatial"] == sp and p["cand"] == cand and p["n"] == n:
                s = f"{p['score']:.3f}"
                return f"**{s}**" if (sp, cand, n) in fr_ids else s
        return "-"
    for label, sp, tm, cand, ns, mk, col in ARMS:
        if sp == "lgd":
            continue
        r = [cell(sp, cand, n) for n in [4, 8, 16, 32, 64, 96]]
        lgd = [cell("lgd", "baseline", n) for n in [8, 32]] if cand == "baseline" else ["-", "-"]
        md.append(f"| {setting} | {cand} | " + " | ".join(r + lgd) + " |")
md.append("")
md.append("Bold = on the frontier for that setting. The LGD/none columns are shown on the baseline row only.")
md.append("")
md.append("Reading (verifier numbers, VERIFICATION.md 5.3): 2D frontier is relclip2 at n=4/8 (2D-only rule; "
          "trust_noise1 at n=8, 0.167, already beats every baseline point incl. n=96 at 12x the calls and LGD/none "
          "at n=8/32); 5D frontier moves by 0.01-0.04 only (trust_noise1 at n=16 is on it; sqrtfloor_clip0.5, not "
          "plotted, holds n=4/8/32); 10D frontier is trust_noise1 at every n in 4..32, then the baseline at n=64/96 "
          "(trust_noise1 n=32 = baseline n=64 at half the calls).")
with open(os.path.join(ROOT, "pareto.md"), "w") as f:
    f.write("\n".join(md) + "\n")
print("wrote pareto.md")
for setting in SETTINGS:
    print(setting, "frontier:", [(p["cand"], p["n"], p["calls"], round(p["score"], 3)) for p in summary[setting][1]])
