"""
Validate PCA structure of age-based target distributions before running guided generation.

Loads CLIP embeddings for age_continuous, gender_bimodal (anchor-based), and
age_gender_combined distributions, then:
  - Prints PC1/PC2 variance explained for each
  - For combined: prints Pearson r of PC1/PC2 with age and gender
  - Saves PCA scatter plots to validation_plots/

Usage:
    python validate_pc1.py \
        --reference_images_dir reference_images \
        --anchor_a_path manly_man.png \
        --anchor_b_path feminine_woman.png \
        --output_dir validation_plots
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from clip_utils import encode_images_clip, load_clip_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reference_images_dir", type=str, default="reference_images")
    p.add_argument("--anchor_a_path", type=str, default="manly_man.png")
    p.add_argument("--anchor_b_path", type=str, default="feminine_woman.png")
    p.add_argument("--n_binary", type=int, default=100,
                   help="Number of embeddings for gender_bimodal (50 each anchor)")
    p.add_argument("--output_dir", type=str, default="validation_plots")
    return p.parse_args()


def load_ref_dir(ref_dir, device, clip_model, clip_processor):
    """Load PNG images from ref_dir, encode to CLIP. Returns (embeddings, ages)."""
    pngs = sorted([f for f in os.listdir(ref_dir) if f.endswith(".png")])
    pil_imgs, ages = [], []
    for fname in pngs:
        m = re.match(r"age_(\d+)_", fname)
        if m is None:
            continue
        pil_imgs.append(Image.open(os.path.join(ref_dir, fname)).convert("RGB").resize((512, 512)))
        ages.append(int(m.group(1)))
    if not pil_imgs:
        raise RuntimeError(f"No age_*.png images in {ref_dir}")
    tensors = torch.cat([TF.to_tensor(img).unsqueeze(0) for img in pil_imgs], dim=0).to(device)
    with torch.no_grad():
        embs = encode_images_clip(tensors, clip_model, clip_processor)
    return embs.cpu().numpy(), ages


def load_era_dir(ref_dir, device, clip_model, clip_processor):
    """Load era PNG images, encode to CLIP. Returns (embeddings, years).
    Filename format: year_p01700_idx_0003.png (p=AD, n=BC → negative)."""
    pngs = sorted([f for f in os.listdir(ref_dir) if f.endswith(".png")])
    pil_imgs, years = [], []
    for fname in pngs:
        m = re.match(r"year_([pn])(\d+)_", fname)
        if m is None:
            continue
        year = int(m.group(2)) * (1 if m.group(1) == "p" else -1)
        pil_imgs.append(Image.open(os.path.join(ref_dir, fname)).convert("RGB").resize((512, 512)))
        years.append(year)
    if not pil_imgs:
        raise RuntimeError(f"No year_*.png images in {ref_dir}")
    tensors = torch.cat([TF.to_tensor(img).unsqueeze(0) for img in pil_imgs], dim=0).to(device)
    with torch.no_grad():
        embs = encode_images_clip(tensors, clip_model, clip_processor)
    return embs.cpu().numpy(), years


def pca_scatter(ax, coords, color_vals, cmap, title, label, marker='o', vmin=None, vmax=None):
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=color_vals, cmap=cmap, alpha=0.7, s=40,
                    marker=marker, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    return sc


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    print("Loading CLIP...", flush=True)
    clip_model, clip_processor = load_clip_model(device)

    # ── 1. age_continuous (woman) ──────────────────────────────────────────────
    ref_dir_w = os.path.join(args.reference_images_dir, "age_woman")
    ref_dir_m = os.path.join(args.reference_images_dir, "age_man")

    print(f"\nLoading age_woman from {ref_dir_w}...", flush=True)
    emb_w, ages_w = load_ref_dir(ref_dir_w, device, clip_model, clip_processor)
    print(f"  Loaded {len(ages_w)} images  age range [{min(ages_w)}, {max(ages_w)}]")

    print(f"Loading age_man from {ref_dir_m}...", flush=True)
    emb_m, ages_m = load_ref_dir(ref_dir_m, device, clip_model, clip_processor)
    print(f"  Loaded {len(ages_m)} images  age range [{min(ages_m)}, {max(ages_m)}]")

    # ── 2. gender_bimodal (anchor-based) ──────────────────────────────────────
    print(f"\nEncoding anchors...", flush=True)
    anc_a = Image.open(args.anchor_a_path).convert("RGB").resize((512, 512))
    anc_b = Image.open(args.anchor_b_path).convert("RGB").resize((512, 512))
    with torch.no_grad():
        e_a = encode_images_clip(TF.to_tensor(anc_a).unsqueeze(0).to(device),
                                 clip_model, clip_processor).cpu().numpy()
        e_b = encode_images_clip(TF.to_tensor(anc_b).unsqueeze(0).to(device),
                                 clip_model, clip_processor).cpu().numpy()
    half = args.n_binary // 2
    emb_binary = np.vstack([np.repeat(e_a, half, axis=0),
                             np.repeat(e_b, args.n_binary - half, axis=0)])
    gender_binary = np.array([0] * half + [1] * (args.n_binary - half))
    print(f"  gender_bimodal: {half}× anchor_A + {args.n_binary - half}× anchor_B")

    # ── 3. age_gender_combined ─────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    idx_w = rng.choice(len(emb_w), size=min(half, len(emb_w)), replace=False)
    idx_m = rng.choice(len(emb_m), size=min(half, len(emb_m)), replace=False)
    emb_comb  = np.vstack([emb_w[idx_w], emb_m[idx_m]])
    ages_comb = [ages_w[i] for i in idx_w] + [ages_m[i] for i in idx_m]
    gender_comb = np.array([0] * len(idx_w) + [1] * len(idx_m))
    print(f"  age_gender_combined: {len(idx_w)} women + {len(idx_m)} men")

    # ── PCA and report ─────────────────────────────────────────────────────────
    distributions = {
        "age_continuous_woman": (emb_w,    np.array(ages_w),  None),
        "gender_bimodal":       (emb_binary, gender_binary,   None),
        "age_gender_combined":  (emb_comb,  np.array(ages_comb), gender_comb),
    }

    print("\n" + "="*60)
    print("PCA REPORT")
    print("="*60)

    for name, (emb, ages_arr, gender_arr) in distributions.items():
        pca = PCA(n_components=2)
        coords = pca.fit_transform(emb)
        var1, var2 = pca.explained_variance_ratio_

        print(f"\n[{name}]  N={len(emb)}")
        print(f"  PC1 var explained: {var1:.3f} ({var1*100:.1f}%)")
        print(f"  PC2 var explained: {var2:.3f} ({var2*100:.1f}%)")
        print(f"  PC1+PC2 total:     {(var1+var2)*100:.1f}%")

        if gender_arr is not None:
            r1_age,  p1_age  = pearsonr(coords[:, 0], ages_arr)
            r2_age,  p2_age  = pearsonr(coords[:, 1], ages_arr)
            r1_gen,  p1_gen  = pearsonr(coords[:, 0], gender_arr)
            r2_gen,  p2_gen  = pearsonr(coords[:, 1], gender_arr)
            print(f"  PC1 ↔ age:    r={r1_age:+.3f}  p={p1_age:.3e}")
            print(f"  PC2 ↔ age:    r={r2_age:+.3f}  p={p2_age:.3e}")
            print(f"  PC1 ↔ gender: r={r1_gen:+.3f}  p={p1_gen:.3e}")
            print(f"  PC2 ↔ gender: r={r2_gen:+.3f}  p={p2_gen:.3e}")

            dominant = "age" if abs(r1_age) > abs(r1_gen) else "gender"
            print(f"  => PC1 correlates primarily with: {dominant}")
            if dominant != "age":
                print("  WARNING: PC1 does NOT primarily correlate with age.")
                print("           Consider increasing age_std or age range and regenerating.")
        else:
            r1, p1 = pearsonr(coords[:, 0], ages_arr)
            r2, p2 = pearsonr(coords[:, 1], ages_arr)
            print(f"  PC1 ↔ age: r={r1:+.3f}  p={p1:.3e}")
            print(f"  PC2 ↔ age: r={r2:+.3f}  p={p2:.3e}")

        # Save scatter plot
        fig, ax = plt.subplots(figsize=(7, 6))
        if name == "gender_bimodal":
            ax.scatter(coords[:half, 0], coords[:half, 1],
                       c='dodgerblue', alpha=0.7, s=40, label='Anchor A (man)')
            ax.scatter(coords[half:, 0], coords[half:, 1],
                       c='crimson', alpha=0.7, s=40, label='Anchor B (woman)')
            ax.legend()
        elif name == "age_continuous_woman":
            sc = ax.scatter(coords[:, 0], coords[:, 1],
                            c=ages_arr, cmap='plasma', alpha=0.7, s=40)
            plt.colorbar(sc, ax=ax, label='Age')
        elif name == "age_gender_combined":
            n_w = len(idx_w)
            sc_w = ax.scatter(coords[:n_w, 0], coords[:n_w, 1],
                              c=ages_arr[:n_w], cmap='Blues', alpha=0.7, s=40,
                              marker='o', label='Woman', vmin=20, vmax=80)
            sc_m = ax.scatter(coords[n_w:, 0], coords[n_w:, 1],
                              c=ages_arr[n_w:], cmap='Reds', alpha=0.7, s=40,
                              marker='^', label='Man', vmin=20, vmax=80)
            plt.colorbar(sc_m, ax=ax, label='Age')
            ax.legend()

        ax.set_title(f"{name}\nPC1={var1*100:.1f}%  PC2={var2*100:.1f}%")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, f"pca_{name}.png")
        fig.savefig(plot_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Plot saved: {plot_path}")

    # ── 4. era (gender-neutral) ────────────────────────────────────────────────
    era_dir = os.path.join(args.reference_images_dir, "era")
    if os.path.isdir(era_dir):
        print(f"\nLoading era from {era_dir}...", flush=True)
        emb_era, years_era = load_era_dir(era_dir, device, clip_model, clip_processor)
        print(f"  Loaded {len(years_era)} images  year range [{min(years_era):,}, {max(years_era):,}]")

        pca_era = PCA(n_components=2)
        coords_era = pca_era.fit_transform(emb_era)
        var1e, var2e = pca_era.explained_variance_ratio_

        r1, p1 = pearsonr(coords_era[:, 0], years_era)
        r2, p2 = pearsonr(coords_era[:, 1], years_era)

        print(f"\n[era_distribution]  N={len(years_era)}")
        print(f"  PC1 var explained: {var1e:.3f} ({var1e*100:.1f}%)")
        print(f"  PC2 var explained: {var2e:.3f} ({var2e*100:.1f}%)")
        print(f"  PC1+PC2 total:     {(var1e+var2e)*100:.1f}%")
        print(f"  PC1 ↔ year: r={r1:+.3f}  p={p1:.3e}")
        print(f"  PC2 ↔ year: r={r2:+.3f}  p={p2:.3e}")
        if abs(r1) > 0.5:
            print(f"  => PC1 correlates with year (r={r1:+.3f}) ✓ GOOD")
        else:
            print(f"  WARNING: PC1 does NOT strongly correlate with year (r={r1:+.3f})")

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(coords_era[:, 0], coords_era[:, 1],
                        c=years_era, cmap='RdYlGn', alpha=0.8, s=50)
        plt.colorbar(sc, ax=ax, label='Year (negative = BC)')
        ax.set_title(f"era_distribution  PC1={var1e*100:.1f}%  PC2={var2e*100:.1f}%\n"
                     f"PC1↔year r={r1:+.3f}  PC2↔year r={r2:+.3f}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "pca_era_distribution.png")
        fig.savefig(plot_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Plot saved: {plot_path}")

    # ── 5. era_continuous_woman ────────────────────────────────────────────────
    era_w_dir = os.path.join(args.reference_images_dir, "era_woman")
    era_m_dir = os.path.join(args.reference_images_dir, "era_man")

    if os.path.isdir(era_w_dir) and os.path.isdir(era_m_dir):
        print(f"\nLoading era_woman...", flush=True)
        emb_ew, years_ew = load_era_dir(era_w_dir, device, clip_model, clip_processor)
        print(f"  {len(years_ew)} images  year range [{min(years_ew):,}, {max(years_ew):,}]")
        print(f"Loading era_man...", flush=True)
        emb_em, years_em = load_era_dir(era_m_dir, device, clip_model, clip_processor)
        print(f"  {len(years_em)} images  year range [{min(years_em):,}, {max(years_em):,}]")

        # era_continuous_woman
        pca_ew = PCA(n_components=2); coords_ew = pca_ew.fit_transform(emb_ew)
        v1, v2 = pca_ew.explained_variance_ratio_
        r1, p1 = pearsonr(coords_ew[:, 0], years_ew)
        r2, p2 = pearsonr(coords_ew[:, 1], years_ew)
        print(f"\n[era_continuous_woman]  N={len(years_ew)}")
        print(f"  PC1={v1*100:.1f}%  PC2={v2*100:.1f}%")
        print(f"  PC1 ↔ year: r={r1:+.3f}  p={p1:.3e}")
        print(f"  PC2 ↔ year: r={r2:+.3f}  p={p2:.3e}")
        print(f"  => PC1 {'✓ correlates with year' if abs(r1)>0.5 else 'WARNING: weak year correlation'}")

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(coords_ew[:, 0], coords_ew[:, 1], c=years_ew, cmap='RdYlGn', alpha=0.8, s=50)
        plt.colorbar(sc, ax=ax, label='Year (negative=BC)')
        ax.set_title(f"era_continuous_woman  PC1={v1*100:.1f}%  PC2={v2*100:.1f}%\nPC1↔year r={r1:+.3f}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(args.output_dir, "pca_era_continuous_woman.png")
        fig.savefig(p, dpi=120, bbox_inches='tight'); plt.close(fig)
        print(f"  Plot saved: {p}")

        # era_gender_combined
        rng_era = np.random.default_rng(42)
        hw = len(emb_ew) // 2; hm = len(emb_em) // 2
        idx_ew = rng_era.choice(len(emb_ew), size=min(hw, len(emb_ew)), replace=False)
        idx_em = rng_era.choice(len(emb_em), size=min(hm, len(emb_em)), replace=False)
        emb_ecomb  = np.vstack([emb_ew[idx_ew], emb_em[idx_em]])
        years_ecomb = [years_ew[i] for i in idx_ew] + [years_em[i] for i in idx_em]
        gender_ecomb = np.array([0]*len(idx_ew) + [1]*len(idx_em))

        pca_ec = PCA(n_components=2); coords_ec = pca_ec.fit_transform(emb_ecomb)
        v1, v2 = pca_ec.explained_variance_ratio_
        r1y, p1y = pearsonr(coords_ec[:, 0], years_ecomb)
        r2y, p2y = pearsonr(coords_ec[:, 1], years_ecomb)
        r1g, p1g = pearsonr(coords_ec[:, 0], gender_ecomb)
        r2g, p2g = pearsonr(coords_ec[:, 1], gender_ecomb)
        dominant = "year" if abs(r1y) > abs(r1g) else "gender"
        print(f"\n[era_gender_combined]  N={len(emb_ecomb)}")
        print(f"  PC1={v1*100:.1f}%  PC2={v2*100:.1f}%")
        print(f"  PC1 ↔ year:   r={r1y:+.3f}  p={p1y:.3e}")
        print(f"  PC2 ↔ year:   r={r2y:+.3f}  p={p2y:.3e}")
        print(f"  PC1 ↔ gender: r={r1g:+.3f}  p={p1g:.3e}")
        print(f"  PC2 ↔ gender: r={r2g:+.3f}  p={p2g:.3e}")
        print(f"  => PC1 correlates primarily with: {dominant}")

        fig, ax = plt.subplots(figsize=(8, 6))
        nw = len(idx_ew)
        sc_w = ax.scatter(coords_ec[:nw, 0], coords_ec[:nw, 1],
                          c=years_ecomb[:nw], cmap='Blues', alpha=0.7, s=40,
                          marker='o', label='Woman', vmin=-8000, vmax=2200)
        sc_m = ax.scatter(coords_ec[nw:, 0], coords_ec[nw:, 1],
                          c=years_ecomb[nw:], cmap='Reds', alpha=0.7, s=40,
                          marker='^', label='Man', vmin=-8000, vmax=2200)
        plt.colorbar(sc_m, ax=ax, label='Year')
        ax.legend(); ax.set_title(f"era_gender_combined  PC1={v1*100:.1f}%  PC2={v2*100:.1f}%\nPC1↔year r={r1y:+.3f}  PC1↔gender r={r1g:+.3f}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(args.output_dir, "pca_era_gender_combined.png")
        fig.savefig(p, dpi=120, bbox_inches='tight'); plt.close(fig)
        print(f"  Plot saved: {p}")

    print("\n" + "="*60)
    print("Validation complete.")


if __name__ == "__main__":
    main()
