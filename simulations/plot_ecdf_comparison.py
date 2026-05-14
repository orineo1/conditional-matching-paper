"""
ECDF comparison plot: MLGD vs MLGD-F (LGD-CM) across 2D, 5D, 10D.
Reads l2_gmm scores from JSON result files and plots empirical CDFs.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

EXPERIMENTS = [
    ("2D",  RESULTS_DIR / "2D_cond_1D"  / "2D_cond_1D_results_seed42.json"),
    ("5D",  RESULTS_DIR / "5D_cond_1D"  / "5D_cond_1D_results_seed42_xt_5_100steps.json"),
    ("10D", RESULTS_DIR / "10D_cond_1D" / "10D_cond_1D_results_seed42_xt_5_100steps.json"),
]

# JSON keys → display labels
METHOD_A = ("LGD",    "MLGD",   "solid",  "#1f77b4")
METHOD_B = ("LGD-CM", "MLGD-F", "dashed", "#ff7f0e")


def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    # prepend 0 so the step starts from the x-axis
    x = np.concatenate([[x[0]], x])
    y = np.concatenate([[0], y])
    return x, y


fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)

for ax, (dim_label, json_path) in zip(axes, EXPERIMENTS):
    with open(json_path) as fh:
        data = json.load(fh)

    for key, label, ls, color in [METHOD_A, METHOD_B]:
        scores = np.array(data[key]["l2_gmm"])
        x, y = ecdf(scores)
        ax.step(x, y, where="post", linestyle=ls, color=color, linewidth=1.8, label=label)

    ax.set_title(f"({dim_label})", fontsize=12)
    ax.set_xlabel(r"$L^2$ GMM", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.05)

axes[0].set_ylabel("Cumulative probability", fontsize=11)
axes[0].legend(fontsize=10)

plt.tight_layout()
out_path = Path(__file__).parent / "ecdf_mlgd_vs_mlgdf.pdf"
plt.savefig(out_path, bbox_inches="tight")
out_png = out_path.with_suffix(".png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
print(f"Saved: {out_png}")
plt.show()
