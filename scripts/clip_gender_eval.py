"""CLIP zero-shot gender classification on experiment images.

For each image, computes cosine similarity between the image embedding
and two text prompts ("a photo of a man" / "a photo of a woman").
Classifies by higher score. Outputs summary + per-image results + PDF grid.

Usage:
    python scripts/clip_gender_eval.py --image_dir autoresearch/output/run2_zeta5
    python scripts/clip_gender_eval.py --image_dir autoresearch/output/run2_zeta5 --device cuda
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


MALE_PROMPT = "a photo of a man"
FEMALE_PROMPT = "a photo of a woman"


def load_clip(device):
    model_id = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    return model, processor


def classify_folder(image_dir, model, processor, device, batch_size=32):
    """Classify all PNGs in a folder. Returns list of (filename, gender, confidence)."""
    png_files = sorted([f for f in os.listdir(image_dir)
                        if f.endswith(".png") and not f.startswith("._")
                        and not f.startswith("gender")])
    if not png_files:
        return []

    # Precompute text embeddings
    text_inputs = processor(text=[MALE_PROMPT, FEMALE_PROMPT],
                            return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_out = model.get_text_features(**text_inputs)
        # Handle both old (tensor) and new (BaseModelOutput) transformers versions
        text_embs = text_out if isinstance(text_out, torch.Tensor) else text_out.pooler_output
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)  # [2, 768]

    results = []
    for i in range(0, len(png_files), batch_size):
        batch_files = png_files[i:i + batch_size]
        images = [Image.open(os.path.join(image_dir, f)).convert("RGB") for f in batch_files]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            img_out = model.get_image_features(**inputs)
            img_embs = img_out if isinstance(img_out, torch.Tensor) else img_out.pooler_output
            img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)  # [B, 768]

            # Cosine similarity with both prompts
            sims = img_embs @ text_embs.T  # [B, 2] — col 0=male, col 1=female
            probs = torch.softmax(sims * 100, dim=1)  # temperature-scaled softmax (CLIP logit_scale ~ 100)

        for fname, prob, sim in zip(batch_files, probs, sims):
            male_prob = prob[0].item()
            female_prob = prob[1].item()
            gender = "Male" if male_prob > female_prob else "Female"
            confidence = max(male_prob, female_prob)
            # Raw cosine similarity margin: positive = more male, negative = more female
            margin = sim[0].item() - sim[1].item()
            results.append((fname, gender, round(confidence, 4), round(margin, 4),
                            round(male_prob, 4)))

    return results


def make_pdf_page(pdf, title, results, image_dir, cols=10):
    """Add a page with Male/Female grids."""
    males = [(f, c, m) for f, g, c, m, _ in results if g == "Male"]
    females = [(f, c, m) for f, g, c, m, _ in results if g == "Female"]

    male_rows = math.ceil(len(males) / cols) if males else 0
    female_rows = math.ceil(len(females) / cols) if females else 0
    total_rows = male_rows + female_rows + 2
    row_height = 1.0
    fig_height = max(4, total_rows * row_height + 1.5)

    fig = plt.figure(figsize=(cols * 1.2, fig_height))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)

    def draw_grid(items, start_row, label, color):
        y_header = 1.0 - (start_row * row_height + 0.3) / fig_height
        fig.text(0.02, y_header, f"{label} ({len(items)})",
                 fontsize=11, fontweight='bold', color=color, va='top')
        for i, (fname, conf, margin) in enumerate(items):
            row = i // cols
            col = i % cols
            left = 0.02 + col * (0.96 / cols)
            bottom = 1.0 - ((start_row + 1 + row) * row_height + row_height) / fig_height
            width = 0.9 / cols
            height = (row_height * 0.85) / fig_height
            ax = fig.add_axes([left, bottom, width, height])
            img = Image.open(os.path.join(image_dir, fname))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"{conf:.2f} ({margin:+.3f})", fontsize=4, pad=1)
        return start_row + 1 + (math.ceil(len(items) / cols) if items else 0)

    current_row = 0
    current_row = draw_grid(males, current_row, "Male", '#2166ac')
    draw_grid(females, current_row, "Female", '#b2182b')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def save_boxplot(results, title, save_path):
    """Save a boxplot of CLIP male/female softmax probabilities."""
    male_confs = [c for _, g, c, _, _ in results if g == "Male"]
    female_confs = [c for _, g, c, _, _ in results if g == "Female"]

    fig, ax = plt.subplots(figsize=(6, 5))
    box_data, box_labels = [], []
    if male_confs:
        box_data.append(male_confs)
        box_labels.append(f"Male (n={len(male_confs)})")
    if female_confs:
        box_data.append(female_confs)
        box_labels.append(f"Female (n={len(female_confs)})")
    if box_data:
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                        widths=0.5, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
        colors = ['#4C72B0', '#DD8452']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.set_ylabel("CLIP softmax confidence (dominant class)")
    ax.set_title(f"CLIP Gender Confidence\n{title}")
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Boxplot saved to: {save_path}", flush=True)


def evaluate_gender_balance(pil_images, device, save_dir=None,
                             model=None, processor=None):
    """Classify a list of PIL images by gender using CLIP zero-shot.

    Args:
        pil_images: list of PIL Images
        device: torch device string
        save_dir: if provided, save images as PNGs here before classifying
        model: preloaded CLIPModel (loaded if None)
        processor: preloaded CLIPProcessor (loaded if None)

    Returns:
        dict with n_male, n_female, gender_L, gender_conf_mean
    """
    import shutil
    import tempfile

    if model is None or processor is None:
        model, processor = load_clip(device)

    # Save images to a temp dir (or save_dir) so classify_folder can read them
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        img_dir = save_dir
    else:
        img_dir = tempfile.mkdtemp()

    for idx, img in enumerate(pil_images):
        img.save(os.path.join(img_dir, f"photo_{idx:04d}.png"))

    results = classify_folder(img_dir, model, processor, device)

    if not save_dir:
        shutil.rmtree(img_dir)

    n_male = sum(1 for _, g, _, _, _ in results if g == "Male")
    n_female = sum(1 for _, g, _, _, _ in results if g == "Female")
    n_total = len(results)
    L = math.sqrt((n_male - n_total / 2) ** 2 + (n_female - n_total / 2) ** 2)
    conf_mean = sum(c for _, _, c, _, _ in results) / max(n_total, 1)

    return {
        "n_male": n_male,
        "n_female": n_female,
        "gender_L": round(L, 2),
        "gender_conf_mean": round(conf_mean, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="CLIP zero-shot gender classification")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Run directory with eval_photos_guided/ and eval_photos_unguided/")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}", flush=True)
    model, processor = load_clip(device)

    run_dir = Path(args.image_dir)
    run_name = run_dir.name
    pdf_path = run_dir / f"clip_gender_{run_name}.pdf"

    with PdfPages(str(pdf_path)) as pdf:
        for condition in ["eval_photos_guided", "eval_photos_unguided"]:
            photo_dir = run_dir / condition
            if not photo_dir.exists():
                continue

            cond_label = "DPS Guided" if condition == "eval_photos_guided" else "Unguided"
            print(f"\n=== {run_name} / {cond_label} ===", flush=True)

            results = classify_folder(str(photo_dir), model, processor, device)
            n_male = sum(1 for _, g, _, _, _ in results if g == "Male")
            n_female = sum(1 for _, g, _, _, _ in results if g == "Female")
            L = math.sqrt((n_male - len(results)/2)**2 + (n_female - len(results)/2)**2)

            print(f"  → {n_male}M/{n_female}F, L={L:.1f}", flush=True)

            # Save per-image JSON
            json_path = photo_dir / "clip_gender_results.json"
            per_image = [{"filename": f, "gender": g, "confidence": c, "margin": m,
                          "male_prob": mp} for f, g, c, m, mp in results]
            with open(json_path, "w") as fp:
                json.dump(per_image, fp, indent=2)

            # Sort images into clip_male/ and clip_female/ subdirs
            import shutil
            clip_male_dir = photo_dir / "clip_male"
            clip_female_dir = photo_dir / "clip_female"
            for d in [clip_male_dir, clip_female_dir]:
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir()
            for fname, gender, conf, margin, _ in results:
                src = photo_dir / fname
                dst_dir = clip_male_dir if gender == "Male" else clip_female_dir
                dst = dst_dir / f"{Path(fname).stem}_conf{conf:.2f}_margin{margin:+.3f}.png"
                shutil.copy2(str(src), str(dst))

            # Boxplot
            boxplot_path = photo_dir / "clip_gender_boxplot.png"
            save_boxplot(results, f"{run_name} — {cond_label} ({n_male}M / {n_female}F)",
                         boxplot_path)

            # PDF page
            title = f"CLIP Zero-Shot — {run_name} — {cond_label} ({n_male}M / {n_female}F)"
            make_pdf_page(pdf, title, results, str(photo_dir))

    print(f"\nPDF saved to: {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
