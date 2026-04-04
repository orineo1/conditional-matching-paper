"""
analysis.py — offline plot generation for LGD-CM runs.

Load a completed run from disk (metrics.json + npy/ files) and
regenerate any figure without touching the GPU.

Usage:
    from analysis import load_run, make_all_plots
    run = load_run("output/dps_main_44388132")
    make_all_plots(run, output_dir="output/dps_main_44388132/plots")

Or from CLI:
    python analysis.py --run_dir output/dps_main_44388132
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ── Data loading ──────────────────────────────────────────────────────────────

def load_run(run_dir):
    """
    Load all saved data from a completed LGD-CM run.
    Returns a dict with metrics + numpy arrays + PIL image lists.
    """
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path) as f:
        metrics = json.load(f)

    data = {"metrics": metrics, "run_dir": run_dir}

    # load npy arrays
    npy_keys = metrics.get("npy", {})
    data["npy"] = {}
    for key, rel_path in npy_keys.items():
        full_path = os.path.join(run_dir, rel_path)
        if os.path.exists(full_path):
            data["npy"][key] = np.load(full_path)
        else:
            print(f"  ⚠️  Missing npy: {full_path}")

    # load individual eval photos as PIL lists
    data["photos_lgd_cm"]  = _load_photo_dir(os.path.join(run_dir, "photos_lgd_cm"))
    data["photos_regular"] = _load_photo_dir(os.path.join(run_dir, "photos_regular"))
    data["targets_man"]    = _load_photo_dir(os.path.join(run_dir, "targets_man"))
    data["targets_woman"]  = _load_photo_dir(os.path.join(run_dir, "targets_woman"))

    # load scribble / source if present
    for name in ["scribble.png", "source_portrait.png",
                 "final_scribble_lgd_cm.png", "final_scribble_regular.png"]:
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            data[name.replace(".png", "")] = Image.open(path)

    return data


def _load_photo_dir(photo_dir):
    """Load all photo_NNN.png files from a directory as a list of PIL images."""
    if not os.path.exists(photo_dir):
        return []
    paths = sorted(
        p for p in os.listdir(photo_dir) if p.endswith(".png")
    )
    return [Image.open(os.path.join(photo_dir, p)) for p in paths]


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_pca(run_data, save_path=None):
    """PCA of CLIP embeddings: targets (man/woman) + LGD-CM + regular."""
    npy = run_data["npy"]
    m   = run_data["metrics"]

    clip_man     = npy.get("clip_targets_man")
    clip_woman   = npy.get("clip_targets_woman")
    clip_lgd_cm  = npy.get("clip_lgd_cm")
    clip_regular = npy.get("clip_regular")

    if any(x is None for x in [clip_man, clip_woman, clip_lgd_cm, clip_regular]):
        print("⚠️  Missing CLIP embeddings for PCA — skipping.")
        return

    combined = np.vstack([clip_man, clip_woman, clip_lgd_cm, clip_regular])
    pca      = PCA(n_components=2)
    coords   = pca.fit_transform(combined)

    n_man     = len(clip_man)
    n_woman   = len(clip_woman)
    n_lgd_cm  = len(clip_lgd_cm)

    c_man     = coords[:n_man]
    c_woman   = coords[n_man:n_man + n_woman]
    c_lgd_cm  = coords[n_man + n_woman:n_man + n_woman + n_lgd_cm]
    c_regular = coords[n_man + n_woman + n_lgd_cm:]

    lgd_cm_mmd  = m.get("final_lgd_cm_mmd",  "?")
    regular_mmd = m.get("final_regular_mmd", "?")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(c_man[:, 0],     c_man[:, 1],
               c="royalblue", alpha=0.6, s=60, label="Target (man)")
    ax.scatter(c_woman[:, 0],   c_woman[:, 1],
               c="crimson",   alpha=0.6, s=60, label="Target (woman)")
    ax.scatter(c_regular[:, 0], c_regular[:, 1],
               c="orange",    alpha=0.8, s=80, marker="s",
               label=f"Regular (MMD={regular_mmd:.4f})")
    ax.scatter(c_lgd_cm[:, 0],  c_lgd_cm[:, 1],
               c="limegreen", alpha=0.8, s=80, marker="x",
               label=f"LGD-CM (MMD={lgd_cm_mmd:.4f})")
    ax.set_title(f"CLIP PCA — Target vs Regular vs LGD-CM\n"
                 f"Variance explained: {pca.explained_variance_ratio_.sum():.1%}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_tsne(run_data, save_path=None, random_state=42):
    """t-SNE of CLIP embeddings: same points as PCA, men/women colored differently."""
    npy = run_data["npy"]
    m   = run_data["metrics"]

    clip_man     = npy.get("clip_targets_man")
    clip_woman   = npy.get("clip_targets_woman")
    clip_lgd_cm  = npy.get("clip_lgd_cm")
    clip_regular = npy.get("clip_regular")

    if any(x is None for x in [clip_man, clip_woman, clip_lgd_cm, clip_regular]):
        print("⚠️  Missing CLIP embeddings for t-SNE — skipping.")
        return

    combined = np.vstack([clip_man, clip_woman, clip_lgd_cm, clip_regular])
    print("  Running t-SNE...", flush=True)
    tsne   = TSNE(n_components=2, random_state=random_state,
                  perplexity=min(30, len(combined) - 1))
    coords = tsne.fit_transform(combined)

    n_man     = len(clip_man)
    n_woman   = len(clip_woman)
    n_lgd_cm  = len(clip_lgd_cm)

    c_man     = coords[:n_man]
    c_woman   = coords[n_man:n_man + n_woman]
    c_lgd_cm  = coords[n_man + n_woman:n_man + n_woman + n_lgd_cm]
    c_regular = coords[n_man + n_woman + n_lgd_cm:]

    lgd_cm_mmd  = m.get("final_lgd_cm_mmd",  "?")
    regular_mmd = m.get("final_regular_mmd", "?")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(c_man[:, 0],     c_man[:, 1],
               c="royalblue", alpha=0.6, s=60, label="Target (man)")
    ax.scatter(c_woman[:, 0],   c_woman[:, 1],
               c="crimson",   alpha=0.6, s=60, label="Target (woman)")
    ax.scatter(c_regular[:, 0], c_regular[:, 1],
               c="orange",    alpha=0.8, s=80, marker="s",
               label=f"Regular (MMD={regular_mmd:.4f})")
    ax.scatter(c_lgd_cm[:, 0],  c_lgd_cm[:, 1],
               c="limegreen", alpha=0.8, s=80, marker="x",
               label=f"LGD-CM (MMD={lgd_cm_mmd:.4f})")
    ax.set_title("CLIP t-SNE — Target vs Regular vs LGD-CM")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_kde(run_data, save_path=None):
    """
    Overlaid KDE of CLIP softmax confidence scores.
    One panel: LGD-CM (orange) vs Regular (blue), both conditions.
    X-axis: p(male) score per image.
    """
    from scipy.stats import gaussian_kde

    m = run_data["metrics"]

    lgd_cm_per_image  = m.get("lgd_cm_gender",  {}).get("per_image", [])
    regular_per_image = m.get("regular_gender", {}).get("per_image", [])

    if not lgd_cm_per_image or not regular_per_image:
        print("⚠️  Missing softmax data for KDE — skipping.")
        return

    lgd_cm_p_male  = np.array([x["p_male"] for x in lgd_cm_per_image])
    regular_p_male = np.array([x["p_male"] for x in regular_per_image])

    xs = np.linspace(0, 1, 300)

    fig, ax = plt.subplots(figsize=(7, 4))

    for scores, color, label in [
        (regular_p_male, "steelblue",  "Regular"),
        (lgd_cm_p_male,  "darkorange", "LGD-CM"),
    ]:
        if len(np.unique(scores)) > 1:
            kde = gaussian_kde(scores, bw_method=0.15)
            ax.fill_between(xs, kde(xs), alpha=0.25, color=color)
            ax.plot(xs, kde(xs), color=color, lw=2, label=label)
        else:
            ax.axvline(scores[0], color=color, lw=2, label=label)

    ax.axvline(0.5, color="gray", lw=1, linestyle="--", alpha=0.6,
               label="p=0.5 threshold")
    ax.set_xlabel("p(male) — CLIP softmax")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.set_title("CLIP softmax confidence distribution")
    ax.legend(); ax.grid(True, alpha=0.3)

    # annotate counts
    lgd_cm_stats  = m.get("lgd_cm_gender",  {})
    regular_stats = m.get("regular_gender", {})
    ax.text(0.02, 0.95,
            f"Regular: {regular_stats.get('n_male','?')}M "
            f"/ {regular_stats.get('n_female','?')}F",
            transform=ax.transAxes, fontsize=9,
            color="steelblue", va="top")
    ax.text(0.02, 0.87,
            f"LGD-CM: {lgd_cm_stats.get('n_male','?')}M "
            f"/ {lgd_cm_stats.get('n_female','?')}F",
            transform=ax.transAxes, fontsize=9,
            color="darkorange", va="top")

    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_boxplot(run_data, save_path=None):
    """Boxplot of CLIP softmax p(male) — LGD-CM vs Regular, same y-axis."""
    m = run_data["metrics"]

    lgd_cm_per_image  = m.get("lgd_cm_gender",  {}).get("per_image", [])
    regular_per_image = m.get("regular_gender", {}).get("per_image", [])

    if not lgd_cm_per_image or not regular_per_image:
        print("⚠️  Missing softmax data for boxplot — skipping.")
        return

    lgd_cm_p_male  = [x["p_male"] for x in lgd_cm_per_image]
    regular_p_male = [x["p_male"] for x in regular_per_image]

    fig, ax = plt.subplots(figsize=(5, 5))
    bp = ax.boxplot(
        [regular_p_male, lgd_cm_p_male],
        labels=["Regular", "LGD-CM"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][1].set_facecolor("darkorange")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", lw=1, linestyle="--", alpha=0.6)
    ax.set_ylabel("p(male) — CLIP softmax")
    ax.set_title("CLIP softmax p(male) distribution")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_portrait_grid(run_data, condition="lgd_cm",
                       n_cols=5, save_path=None):
    """
    Reload and display a grid of eval portraits.
    condition: "lgd_cm" or "regular"
    """
    photos = run_data.get(f"photos_{condition}", [])
    if not photos:
        print(f"⚠️  No photos found for condition '{condition}'")
        return

    m         = run_data["metrics"]
    stats_key = f"{condition}_gender"
    stats     = m.get(stats_key, {})
    n_male    = stats.get("n_male",   "?")
    n_female  = stats.get("n_female", "?")

    n       = len(photos)
    n_rows  = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3 * n_cols, 3 * n_rows))
    axes = np.array(axes).flatten()

    per_image = stats.get("per_image", [{}] * n)
    for idx, (photo, ax) in enumerate(zip(photos, axes)):
        ax.imshow(photo)
        if idx < len(per_image):
            p = per_image[idx]
            label = p.get("label", "")
            conf  = p.get("p_male", 0) if label == "male" else p.get("p_female", 0)
            color = "royalblue" if label == "male" else "crimson"
            ax.set_title(f"{label}\n{conf:.2f}", fontsize=8, color=color)
        ax.axis("off")

    # hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    title = (f"{'LGD-CM' if condition == 'lgd_cm' else 'Regular'} portraits "
             f"— {n_male}M / {n_female}F")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def compare_scribbles_heatmap(run_data, save_path=None):
    """
    Pixel-wise absolute difference heatmap between LGD-CM and regular scribbles.
    Moved here from visualization.py.
    """
    lgd_cm_pil  = run_data.get("final_scribble_lgd_cm")
    regular_pil = run_data.get("final_scribble_regular")

    if lgd_cm_pil is None or regular_pil is None:
        print("⚠️  Missing scribble images for heatmap — skipping.")
        return

    lgd_cm_np  = np.array(lgd_cm_pil).astype(float)
    regular_np = np.array(regular_pil).astype(float)
    diff       = np.abs(lgd_cm_np - regular_np).mean(axis=2)
    diff_norm  = diff / (diff.max() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(regular_pil);                  axes[0].set_title("Regular");   axes[0].axis("off")
    axes[1].imshow(lgd_cm_pil);                   axes[1].set_title("LGD-CM");    axes[1].axis("off")
    im = axes[2].imshow(diff_norm, cmap="hot");   axes[2].set_title("Difference"); axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    plt.suptitle(f"Max diff: {diff.max():.1f}  Mean diff: {diff.mean():.2f}",
                 fontsize=12)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ── Make all plots ────────────────────────────────────────────────────────────

def make_all_plots(run_dir, plots_dir=None):
    """
    Load a run and regenerate all plots.
    Saves to <run_dir>/plots/ by default.
    """
    if plots_dir is None:
        plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Loading run from {run_dir}...", flush=True)
    run = load_run(run_dir)

    print("Generating PCA...",       flush=True)
    plot_pca(run,
             save_path=os.path.join(plots_dir, "pca.png"))

    print("Generating t-SNE...",     flush=True)
    plot_tsne(run,
              save_path=os.path.join(plots_dir, "tsne.png"))

    print("Generating KDE...",       flush=True)
    plot_kde(run,
             save_path=os.path.join(plots_dir, "kde.png"))

    print("Generating boxplot...",   flush=True)
    plot_boxplot(run,
                 save_path=os.path.join(plots_dir, "boxplot.png"))

    print("Generating portrait grids...", flush=True)
    plot_portrait_grid(run, condition="lgd_cm",
                       save_path=os.path.join(plots_dir, "portraits_lgd_cm.png"))
    plot_portrait_grid(run, condition="regular",
                       save_path=os.path.join(plots_dir, "portraits_regular.png"))

    print("Generating heatmap...",   flush=True)
    compare_scribbles_heatmap(run,
                              save_path=os.path.join(plots_dir, "scribble_heatmap.png"))

    print(f"✅ All plots saved to {plots_dir}", flush=True)
    return run


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save_or_show(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate all plots from a saved LGD-CM run."
    )
    parser.add_argument("--run_dir",   required=True,
                        help="Path to the run output directory")
    parser.add_argument("--plots_dir", default=None,
                        help="Where to save plots (default: <run_dir>/plots/)")
    args = parser.parse_args()
    make_all_plots(args.run_dir, args.plots_dir)
