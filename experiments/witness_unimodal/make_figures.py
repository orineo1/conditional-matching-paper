"""
make_figures.py — publication figures for the Round-2 identifiability results.

Reads results/identifiability_*_cell_*.json (cap1_inv convention, seeds 42 and
1042) and writes ~150 dpi PNGs into figures/:

  fig1_landing_map.png     — mean-blindness vs MMD identifiability: jittered
                             x_hat landing strips per arm, colored by which
                             target was optimized, with KS p-values.
  fig2_witness_paired.png  — witness-vs-uniform paired differences in distance
                             to the target's own design point, per seed and
                             pooled, uni vs bi target (the regime asymmetry).

Usage: python make_figures.py
"""
import os, glob, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(_HERE, "figures")
X_BI, X_UNI = -2.0, 2.0

# Okabe-Ito CVD-safe pair; color follows the TARGET entity, fixed order (bi, uni)
C_BI, C_UNI = "#0072B2", "#E69F00"
INK, MUTED = "#333333", "#8a8a8a"


def load(conv="cap1_inv"):
    data = {}   # data[(seed, target, arm)] = {restart: row}
    for p in sorted(glob.glob(os.path.join(_HERE, "results", "identifiability_*_cell_*.json"))):
        d = json.load(open(p))
        if d["meta"]["convention"] != conv:
            continue
        seed = d["meta"]["seed"]
        for r in d["rows"]:
            data.setdefault((seed, r["target"], r["arm"]), {})[r["restart"]] = r
    return data


def t975(df):
    tbl = {7: 2.365, 39: 2.023, 79: 1.990}
    for k in sorted(tbl):
        if df <= k:
            return tbl[k]
    return 1.96


def fig1(data):
    arms = ["mean", "mmd", "mmd_uniform", "mmd_witness"]
    arm_labels = {"mean": "mean matching", "mmd": "MMD (full)",
                  "mmd_uniform": "MMD + uniform backsel (k=8/32)",
                  "mmd_witness": "MMD + witness backsel (k=8/32)"}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(arms), 1, figsize=(8.2, 7.4), sharex=True)
    for ax, arm in zip(axes, arms):
        xs = {}
        for tgt in ("bi", "uni"):
            xs[tgt] = [r["x_hat"] for s in (42, 1042)
                       for r in data.get((s, tgt, arm), {}).values()]
        ks = ks_2samp(xs["bi"], xs["uni"])
        for tgt, y0, c in (("bi", 0.62, C_BI), ("uni", 0.28, C_UNI)):
            v = np.array(xs[tgt])
            ax.scatter(v, y0 + rng.uniform(-0.10, 0.10, len(v)), s=14, color=c,
                       alpha=0.55, linewidths=0, zorder=3)
        for xd, lab in ((X_BI, "$x_{bi}$"), (X_UNI, "$x_{uni}$")):
            ax.axvline(xd, color=MUTED, linestyle="--", linewidth=1, zorder=1)
        ax.set_ylim(0, 0.95)
        ax.set_yticks([0.62, 0.28])
        ax.set_yticklabels(["target: bimodal", "target: unimodal"], fontsize=8.5)
        for t, c in zip(ax.get_yticklabels(), (C_BI, C_UNI)):
            t.set_color(c)
        p_txt = f"p = {ks.pvalue:.2f}" if ks.pvalue >= 0.001 else "p < 10$^{-4}$"
        ax.text(0.99, 0.92, f"{arm_labels[arm]}   (KS {p_txt}, n = 80 + 80)",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        ax.grid(axis="x", alpha=0.18)
    axes[0].text(X_BI, 1.02, "$x_{bi}=-2$", ha="center", fontsize=8.5,
                 color=MUTED, transform=axes[0].get_xaxis_transform())
    axes[0].text(X_UNI, 1.02, "$x_{uni}=+2$", ha="center", fontsize=8.5,
                 color=MUTED, transform=axes[0].get_xaxis_transform())
    axes[-1].set_xlabel(r"landed design point $\hat{x}$", fontsize=10)
    fig.suptitle("Mean matching cannot tell the targets apart; MMD lands on the right design point\n"
                 "dot = one restart's $\\hat{x}$ (both seeds pooled);  KS compares bi- vs uni-target landing sets\n"
                 "convention: step cap $\\tau{=}1$ with $1/\\sqrt{\\alpha_t}$ scaling", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = os.path.join(FIG_DIR, "fig1_landing_map.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig2(data):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.1))  # y NOT shared: two bi-panel
    # outliers (~±4) would otherwise squash the uni panel's ~0.05-scale effect
    rng = np.random.default_rng(1)
    for ax, tgt, metric, ttl in (
            (axes[0], "uni", "dist_uni", "concentrated unimodal target"),
            (axes[1], "bi", "dist_bi", "bimodal target (pre-registered null)")):
        groups = []
        for s in (42, 1042):
            w, u = data[(s, tgt, "mmd_witness")], data[(s, tgt, "mmd_uniform")]
            common = sorted(set(w) & set(u))
            groups.append([w[i][metric] - u[i][metric] for i in common])
        groups.append(groups[0] + groups[1])                       # pooled
        for gx, (label, diffs) in enumerate(zip(["seed 42", "seed 1042",
                                                 "pooled (80 pairs)"], groups)):
            d = np.array(diffs)
            m, sd = d.mean(), d.std(ddof=1)
            ci = t975(len(d) - 1) * sd / math.sqrt(len(d))
            ax.scatter(gx + rng.uniform(-0.16, 0.16, len(d)), d, s=12,
                       color=MUTED, alpha=0.45, linewidths=0, zorder=2)
            ax.errorbar(gx, m, yerr=ci, fmt="o", color=C_UNI if tgt == "uni" else C_BI,
                        markersize=7, capsize=5, linewidth=2, zorder=4,
                        markeredgecolor="white", markeredgewidth=1)
            sig = "*" if abs(m) > ci else ""
            ax.annotate(f"{m:+.3f}{sig}\n±{ci:.3f}", (gx, m + ci), xytext=(10, 6),
                        textcoords="offset points", fontsize=8.5, color=INK)
        ax.axhline(0, color=INK, linewidth=1)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["seed 42", "seed 1042", "pooled"], fontsize=9)
        ax.set_xlim(-0.5, 2.75)
        ax.set_title(ttl, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("paired diff per restart:\n"
                       r"dist$(\hat{x},x^{*})_{witness}$ − dist$(\hat{x},x^{*})_{uniform}$",
                       fontsize=9)
    fig.suptitle("Witness back-selection helps only where off-mode stragglers exist\n"
                 "negative = witness lands closer to the true design point $x^{*}$;  dot = one paired restart\n"
                 "marker = mean, bar = 95% CI, * = CI excludes 0;  note the different y-scales",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(FIG_DIR, "fig2_witness_paired.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    data = load()
    print(fig1(data))
    print(fig2(data))
