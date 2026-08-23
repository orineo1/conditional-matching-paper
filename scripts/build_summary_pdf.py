"""
Build a summary PDF for 4 DPS runs.
Each run gets one page with:
  - Final DPS image + Final regular image
  - Starting scribble (HED)
  - Target CLIP PCA
  - CLIP gender classification confidence (bar chart)
  - Delta MMD
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

# ── CLIP gender eval ──────────────────────────────────────────────────────────
def clip_gender_scores(photos_dir, clip_model, clip_processor, device):
    import torch
    import torchvision.transforms.functional as TF

    prompts = ["a photo of a man", "a photo of a woman"]
    text_inputs = clip_processor.tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_feats = clip_model.get_text_features(**text_inputs)
    if not isinstance(text_feats, torch.Tensor):
        text_feats = text_feats.pooler_output
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    files = sorted([f for f in os.listdir(photos_dir) if f.endswith(".png")])[:100]
    man_scores = []
    for fname in files:
        img = Image.open(os.path.join(photos_dir, fname)).convert("RGB")
        tensor = TF.to_tensor(img).unsqueeze(0).to(device)
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            img_feats = clip_model.get_image_features(**inputs)
        if not isinstance(img_feats, torch.Tensor):
            img_feats = img_feats.pooler_output
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        sims = (img_feats @ text_feats.T).squeeze(0).cpu().numpy()
        # softmax
        exp = np.exp(sims - sims.max())
        probs = exp / exp.sum()
        man_scores.append(float(probs[0]))
    return np.array(man_scores)


RUNS = [
    {
        "dir":   "output/dps_binary_man_hed_44379913",
        "label": "Gender Binary\n(man HED scribble, 250 steps, 40% noise)",
        "mode":  "binary",
    },
    {
        "dir":   "output/dps_interp_man_hed_44380149",
        "label": "Gender Interpolated\n(man HED scribble, 250 steps, 40% noise)",
        "mode":  "interpolated",
    },
    {
        "dir":   "output/dps_age_cont_44380525",
        "label": "Age Continuous — Women\n(age-72 woman HED scribble, 500 steps, 40% noise)",
        "mode":  "age_continuous",
    },
    {
        "dir":   "output/dps_age_gen_44380509",
        "label": "Age + Gender Combined\n(age-72 man HED scribble, 500 steps, 40% noise)",
        "mode":  "age_gender_combined",
    },
]

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PDF = os.path.join(BASE, "output", "summary_4runs.pdf")


def load_img(path):
    return Image.open(path).convert("RGB")


def main():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading CLIP...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()

    os.makedirs(os.path.join(BASE, "output"), exist_ok=True)

    with PdfPages(OUTPUT_PDF) as pdf:
        for run in RUNS:
            run_dir = os.path.join(BASE, run["dir"])
            print(f"\nProcessing: {run['label'].split(chr(10))[0]}")

            with open(os.path.join(run_dir, "metrics.json")) as f:
                metrics = json.load(f)
            dps_mmd = metrics["final_dps_mmd"]
            reg_mmd = metrics["final_regular_mmd"]
            delta   = metrics["mmd_delta"]

            scribble_img = load_img(os.path.join(run_dir, "scribble.png"))
            pca_img      = load_img(os.path.join(run_dir, "final_pca_comparison.png"))

            # Load individual photos (up to 25 each)
            def load_photos(folder, n=25):
                files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])[:n]
                return [load_img(os.path.join(folder, f)) for f in files]

            dps_photos = load_photos(os.path.join(run_dir, "photos_dps"))
            reg_photos = load_photos(os.path.join(run_dir, "photos_regular"))

            print("  Computing gender scores...")
            dps_scores = clip_gender_scores(
                os.path.join(run_dir, "photos_dps"), clip_model, clip_processor, device)
            reg_scores = clip_gender_scores(
                os.path.join(run_dir, "photos_regular"), clip_model, clip_processor, device)
            dps_mean = dps_scores.mean()
            reg_mean = reg_scores.mean()

            # ── Page 1: photo grids ─────────────────────────────────────────
            fig = plt.figure(figsize=(16, 14))
            fig.suptitle(run["label"] + f"\n  ΔMMD={delta:+.4f}   DPS MMD={dps_mmd:.3f}   Reg MMD={reg_mmd:.3f}   P(man): DPS={dps_mean:.2f} Reg={reg_mean:.2f}",
                         fontsize=11, fontweight="bold", y=0.99)

            gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.08, top=0.93, bottom=0.01)
            gs_dps = gridspec.GridSpecFromSubplotSpec(5, 5, subplot_spec=gs[0], hspace=0.02, wspace=0.02)
            gs_reg = gridspec.GridSpecFromSubplotSpec(5, 5, subplot_spec=gs[1], hspace=0.02, wspace=0.02)

            for grid, photos, label, scores in [
                (gs_dps, dps_photos, "DPS guided", dps_scores),
                (gs_reg, reg_photos, "Unguided", reg_scores),
            ]:
                for i, (img, score) in enumerate(zip(photos, scores)):
                    ax = fig.add_subplot(grid[i // 5, i % 5])
                    ax.imshow(img)
                    ax.set_title(f"P(m)={score:.2f}", fontsize=5, pad=1)
                    ax.axis("off")
                # label the block
                ax0 = fig.add_subplot(grid[:, :])
                ax0.set_visible(False)
                fig.text(0.01, ax0.get_position().y1 - 0.005, label,
                         fontsize=10, fontweight="bold", va="top",
                         color="steelblue" if "DPS" in label else "gray")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── Page 2: scribble + PCA + gender histogram ───────────────────
            fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig2.suptitle(run["label"].split("\n")[0] + " — diagnostics", fontsize=11)

            axes[0].imshow(scribble_img); axes[0].set_title("Starting scribble"); axes[0].axis("off")
            axes[1].imshow(pca_img);      axes[1].set_title("CLIP PCA");          axes[1].axis("off")

            bins = np.linspace(0, 1, 25)
            axes[2].hist(dps_scores, bins=bins, alpha=0.6, color="steelblue", label=f"DPS (μ={dps_mean:.2f})")
            axes[2].hist(reg_scores, bins=bins, alpha=0.6, color="salmon",    label=f"Unguided (μ={reg_mean:.2f})")
            axes[2].axvline(0.5, color="gray", lw=1, ls="--")
            axes[2].set_xlabel("P(man)"); axes[2].set_ylabel("Count")
            axes[2].set_title("Gender distribution"); axes[2].legend()

            plt.tight_layout()
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

            print(f"  Done. ΔMMD={delta:+.4f}  P(man) DPS={dps_mean:.3f} Reg={reg_mean:.3f}")

    print(f"\nSaved: {OUTPUT_PDF}")
    os.system(f"open '{OUTPUT_PDF}'")


if __name__ == "__main__":
    main()
