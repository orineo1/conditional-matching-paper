"""Build CLIP gender eval PDF for asymmetric target ablation runs.

Usage:
    python scripts/build_asym_pdf.py
"""

import math
import os
import sys

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, "scripts")
from clip_gender_eval import load_clip, classify_folder, make_pdf_page


# All runs to evaluate
RUNS = [
    {
        "dir": "output/dps_asym_30m70f_44317554",
        "label": "Baseline (zeta=5, start=125, seed=1)",
        "dps_mmd": 0.291, "reg_mmd": 0.605,
    },
    {
        "dir": "output/dps_asym_zeta2_44334445_0",
        "label": "Zeta=2 (weaker guidance)",
        "dps_mmd": 0.330, "reg_mmd": 0.605,
    },
    {
        "dir": "output/dps_asym_lessnoise_44334445_1",
        "label": "Less noise (start=150, strength=0.4)",
        "dps_mmd": 0.291, "reg_mmd": 0.605,
    },
]

TARGET_MALE_RATIO = 0.3


def eval_run(run, model, processor, device):
    """Classify photos and return results dict."""
    run_results = {}
    for folder, cond_label in [("photos_dps", "DPS Guided"), ("photos_regular", "Unguided")]:
        photo_dir = os.path.join(run["dir"], folder)
        if not os.path.exists(photo_dir):
            continue
        results = classify_folder(photo_dir, model, processor, device)
        n_male = sum(1 for _, g, _, _, _ in results if g == "Male")
        n_female = sum(1 for _, g, _, _, _ in results if g == "Female")
        n = len(results)
        L_50 = math.sqrt((n_male - n / 2) ** 2 + (n_female - n / 2) ** 2)
        L_target = math.sqrt((n_male - TARGET_MALE_RATIO * n) ** 2 +
                             (n_female - (1 - TARGET_MALE_RATIO) * n) ** 2)
        run_results[folder] = {
            "results": results, "n_male": n_male, "n_female": n_female,
            "L_50": L_50, "L_target": L_target, "cond_label": cond_label,
        }
    return run_results


def add_summary_page(pdf, all_run_data):
    """First page: comparison table + bar chart across all runs."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Asymmetric Target Experiment: 30M / 70F — All Ablations\n"
                 "CLIP Zero-Shot Gender Classification",
                 fontsize=16, fontweight="bold", y=0.98)

    # Bar chart
    ax = fig.add_axes([0.05, 0.45, 0.9, 0.45])
    run_labels = [r["label"] for r in RUNS]
    dps_male = [all_run_data[r["dir"]]["photos_dps"]["n_male"] for r in RUNS]
    dps_female = [all_run_data[r["dir"]]["photos_dps"]["n_female"] for r in RUNS]
    reg_male = [all_run_data[r["dir"]]["photos_regular"]["n_male"] for r in RUNS]
    reg_female = [all_run_data[r["dir"]]["photos_regular"]["n_female"] for r in RUNS]

    x = np.arange(len(RUNS))
    w = 0.2
    ax.bar(x - 1.5 * w, dps_male, w, label="DPS Male", color="#4C72B0", alpha=0.8)
    ax.bar(x - 0.5 * w, dps_female, w, label="DPS Female", color="#DD8452", alpha=0.8)
    ax.bar(x + 0.5 * w, reg_male, w, label="Unguided Male", color="#4C72B0", alpha=0.3)
    ax.bar(x + 1.5 * w, reg_female, w, label="Unguided Female", color="#DD8452", alpha=0.3)
    ax.axhline(y=30, color="#4C72B0", linestyle="--", alpha=0.5, label="Target M (30)")
    ax.axhline(y=70, color="#DD8452", linestyle="--", alpha=0.5, label="Target F (70)")
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, fontsize=9)
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 110)

    for i in range(len(RUNS)):
        ax.text(i - 1.5 * w, dps_male[i] + 1, str(dps_male[i]), ha="center", fontsize=7, fontweight="bold")
        ax.text(i - 0.5 * w, dps_female[i] + 1, str(dps_female[i]), ha="center", fontsize=7, fontweight="bold")

    # Table
    ax2 = fig.add_axes([0.05, 0.02, 0.9, 0.35])
    ax2.axis("off")
    header = ["Run", "DPS M/F", "L_target", "Unguided M/F", "L_target", "DPS MMD", "Reg MMD", "Delta"]
    rows = []
    for r in RUNS:
        d = all_run_data[r["dir"]]
        dp, rg = d["photos_dps"], d["photos_regular"]
        delta = r["reg_mmd"] - r["dps_mmd"]
        rows.append([
            r["label"],
            f"{dp['n_male']}M / {dp['n_female']}F",
            f"{dp['L_target']:.1f}",
            f"{rg['n_male']}M / {rg['n_female']}F",
            f"{rg['L_target']:.1f}",
            f"{r['dps_mmd']:.3f}",
            f"{r['reg_mmd']:.3f}",
            f"{delta:+.3f}",
        ])

    table = ax2.table(cellText=[header] + rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    for j in range(len(header)):
        table[0, j].set_facecolor("#d4e6f1")
        table[0, j].set_text_props(fontweight="bold")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_run_pages(pdf, run, run_data):
    """Add per-run pages: photo grids, boxplots, PCA, curves, heatmap."""
    run_dir = run["dir"]

    # Photo grid pages
    for folder in ["photos_dps", "photos_regular"]:
        if folder not in run_data:
            continue
        d = run_data[folder]
        photo_dir = os.path.join(run_dir, folder)
        title = (f"{run['label']} — {d['cond_label']}\n"
                 f"{d['n_male']}M / {d['n_female']}F  (L_target={d['L_target']:.1f})")
        make_pdf_page(pdf, title, d["results"], photo_dir)

    # Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{run['label']} — Confidence", fontsize=13, fontweight="bold")
    for idx, folder in enumerate(["photos_dps", "photos_regular"]):
        if folder not in run_data:
            continue
        d = run_data[folder]
        male_confs = [c for _, g, c, _, _ in d["results"] if g == "Male"]
        female_confs = [c for _, g, c, _, _ in d["results"] if g == "Female"]
        box_data, box_labels = [], []
        if male_confs:
            box_data.append(male_confs)
            box_labels.append(f"Male (n={len(male_confs)})")
        if female_confs:
            box_data.append(female_confs)
            box_labels.append(f"Female (n={len(female_confs)})")
        if box_data:
            bp = axes[idx].boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                                   widths=0.5, showmeans=True,
                                   meanprops=dict(marker="D", markerfacecolor="red", markersize=6))
            colors = ["#4C72B0", "#DD8452"]
            for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        axes[idx].set_ylabel("Confidence")
        axes[idx].set_title(d["cond_label"])
        axes[idx].set_ylim(0.45, 1.02)
        axes[idx].grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # PCA, training curves, scribble heatmap (embed existing PNGs)
    for fname, title in [
        ("final_pca_comparison.png", "CLIP PCA"),
        ("training_curves.png", "Training Curves"),
        ("scribble_heatmap.png", "Scribble Heatmap"),
    ]:
        fpath = os.path.join(run_dir, fname)
        if not os.path.exists(fpath):
            continue
        img = Image.open(fpath)
        w, h = img.size
        fig_w = 14
        fig_h = fig_w * h / w
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{run['label']} — {title}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    model, processor = load_clip(device)

    # Evaluate all runs
    all_run_data = {}
    for run in RUNS:
        print(f"\n=== {run['label']} ===")
        all_run_data[run["dir"]] = eval_run(run, model, processor, device)
        for folder in ["photos_dps", "photos_regular"]:
            d = all_run_data[run["dir"]][folder]
            print(f"  {d['cond_label']}: {d['n_male']}M/{d['n_female']}F, "
                  f"L_target={d['L_target']:.1f}")

    # Build PDF
    pdf_path = "output/clip_gender_asym_all_ablations.pdf"
    with PdfPages(pdf_path) as pdf:
        add_summary_page(pdf, all_run_data)
        for run in RUNS:
            add_run_pages(pdf, run, all_run_data[run["dir"]])

    print(f"\nPDF saved to: {pdf_path}")


if __name__ == "__main__":
    main()
