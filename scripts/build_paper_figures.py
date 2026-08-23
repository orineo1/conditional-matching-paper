"""
Build two appendix figures for the paper.

Figure 1: scribble_change_regions.png
  3 panels: unguided scribble | DPS-guided scribble | pixel diff heatmap

Figure 2: saliency_and_correlation.png
  3 panels: gender saliency | overlay on scribble | binned correlation
"""

import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr, binned_statistic, sem

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "output", "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
UNGUIDED_PNG  = os.path.join(BASE, "autoresearch/output/run2_zeta5/final_scribble_unguided.png")
GUIDED_PNG    = os.path.join(BASE, "autoresearch/output/run2_zeta5/final_scribble_guided.png")
INPUT_SCRIB   = os.path.join(BASE, "autoresearch/output/run2_zeta5/input_scribble.png")
SALIENCY_NPY  = os.path.join(BASE, "output/saliency_44257689/saliency.npy")
DIFF_NPY      = os.path.join(BASE, "output/scribble_diff/scribble_diff.npy")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: scribble_change_regions.png
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Figure 1: scribble_change_regions.png")

unguided = np.array(Image.open(UNGUIDED_PNG).convert("RGB"))
guided   = np.array(Image.open(GUIDED_PNG).convert("RGB"))
diff_raw = np.load(DIFF_NPY)
diff_smooth = gaussian_filter(diff_raw, sigma=2.0)

fig1, axes = plt.subplots(1, 3, figsize=(10, 3.5))

axes[0].imshow(unguided)
axes[0].set_title("Unguided (ζ = 0)", fontsize=11)
axes[0].axis("off")

axes[1].imshow(guided)
axes[1].set_title("DPS-guided (ζ = 5)", fontsize=11)
axes[1].axis("off")

im = axes[2].imshow(diff_smooth, cmap="hot")
axes[2].set_title("Pixel Difference", fontsize=11)
axes[2].axis("off")
cbar = fig1.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, orientation="vertical")
cbar.ax.tick_params(labelsize=8)

plt.tight_layout(pad=0.5)
fig1_path = os.path.join(OUT_DIR, "scribble_change_regions.png")
fig1.savefig(fig1_path, dpi=200, bbox_inches="tight")
plt.close(fig1)
print(f"  Saved: {fig1_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: saliency_and_correlation.png
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Figure 2: saliency_and_correlation.png")

saliency = np.load(SALIENCY_NPY)       # (512, 512)
input_scrib = np.array(Image.open(INPUT_SCRIB).convert("RGB")).astype(np.float32) / 255.0

# Build binned correlation: line pixels only (scribble brightness > 0.3)
line_mask = input_scrib.mean(axis=2) > 0.3
diff_smooth_corr = gaussian_filter(diff_raw, sigma=2.0)
sal_line  = saliency[line_mask]
diff_line = diff_smooth_corr[line_mask]

n_bins = 15
# Equal-count quantile bins (same n per bin, gives uniform/smooth curve)
quantile_edges = np.percentile(sal_line, np.linspace(0, 100, n_bins + 1))
bin_ids = np.digitize(sal_line, quantile_edges[1:-1])  # 0..n_bins-1
bin_means  = np.array([sal_line[bin_ids == i].mean()  for i in range(n_bins)])
diff_means = np.array([diff_line[bin_ids == i].mean() for i in range(n_bins)])
diff_sems  = np.array([sem(diff_line[bin_ids == i])   for i in range(n_bins)])

rho, _ = spearmanr(sal_line, diff_line)
print(f"  Spearman ρ = {rho:.3f}  (n={line_mask.sum()} line pixels)")

# Layout: panels 1+2 square, panel 3 slightly wider
fig2 = plt.figure(figsize=(12, 3.8))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.3], wspace=0.35)

# Panel 1: saliency heatmap
ax1 = fig2.add_subplot(gs[0])
im1 = ax1.imshow(saliency, cmap="hot")
ax1.set_title("Gender Saliency", fontsize=11)
ax1.axis("off")
cb1 = fig2.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, orientation="vertical")
cb1.ax.tick_params(labelsize=7)

# Panel 2: overlay on input scribble
ax2 = fig2.add_subplot(gs[1])
ax2.imshow(input_scrib, cmap="gray")
ax2.imshow(saliency, cmap="hot", alpha=0.6)
ax2.set_title("Overlay on Scribble", fontsize=11)
ax2.axis("off")

# Panel 3: binned correlation
ax3 = fig2.add_subplot(gs[2])
ax3.errorbar(bin_means, diff_means, yerr=diff_sems,
             fmt='o-', color='steelblue', markersize=5,
             capsize=3, linewidth=1.5, elinewidth=1)
ax3.set_xlabel("Gender Saliency (binned mean)", fontsize=9)
ax3.set_ylabel("DPS Scribble Change (mean)", fontsize=9)
ax3.set_title(f"Saliency vs. DPS Change (ρ = {rho:.2f})", fontsize=11)
ax3.tick_params(labelsize=8)
ax3.grid(True, alpha=0.3)
# Annotate rho inside axes
ax3.text(0.97, 0.05, f"Spearman ρ = {rho:.2f}",
         transform=ax3.transAxes, ha="right", va="bottom",
         fontsize=9, color="steelblue",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="steelblue", alpha=0.8))

plt.tight_layout(pad=0.5)
fig2_path = os.path.join(OUT_DIR, "saliency_and_correlation.png")
fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved: {fig2_path}")

print("\nDone.")
