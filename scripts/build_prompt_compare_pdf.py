"""Build PDF comparing two prompt conditions from the same scribble.

Usage:
    conda run -n grassy_dit python scripts/build_prompt_compare_pdf.py \
        --run_dir SD_cond_SD_controlnet/output/prompt_compare_XXXXX
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Add scripts/ to path for clip_gender_eval
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from clip_gender_eval import load_clip, classify_folder, make_pdf_page


def encode_folder_clip(image_dir, model, processor, device, batch_size=32):
    """Encode all PNGs in a folder to CLIP embeddings."""
    from PIL import Image

    png_files = sorted([f for f in os.listdir(image_dir)
                        if f.endswith(".png") and not f.startswith("._")])
    all_embs = []
    text_inputs = processor(text=["a photo of a man", "a photo of a woman"],
                            return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_out = model.get_text_features(**text_inputs)
        text_embs = text_out if isinstance(text_out, torch.Tensor) else text_out.pooler_output
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

    for i in range(0, len(png_files), batch_size):
        batch_files = png_files[i:i + batch_size]
        images = [Image.open(os.path.join(image_dir, f)).convert("RGB") for f in batch_files]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            img_out = model.get_image_features(**inputs)
            img_embs = img_out if isinstance(img_out, torch.Tensor) else img_out.pooler_output
            img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
            all_embs.append(img_embs.cpu().numpy())

    return np.vstack(all_embs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
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

    run_dir = Path(args.run_dir)
    prompt_a_dir = run_dir / "prompt_a"
    prompt_b_dir = run_dir / "prompt_b"

    assert prompt_a_dir.exists(), f"Not found: {prompt_a_dir}"
    assert prompt_b_dir.exists(), f"Not found: {prompt_b_dir}"

    print(f"Device: {device}", flush=True)
    model, processor = load_clip(device)

    # Classify both sets
    print("\n=== Prompt A ===", flush=True)
    results_a = classify_folder(str(prompt_a_dir), model, processor, device)
    n_male_a = sum(1 for _, g, _, _, _ in results_a if g == "Male")
    n_female_a = len(results_a) - n_male_a
    print(f"  → {n_male_a}M/{n_female_a}F", flush=True)

    print("\n=== Prompt B ===", flush=True)
    results_b = classify_folder(str(prompt_b_dir), model, processor, device)
    n_male_b = sum(1 for _, g, _, _, _ in results_b if g == "Male")
    n_female_b = len(results_b) - n_male_b
    print(f"  → {n_male_b}M/{n_female_b}F", flush=True)

    # CLIP embeddings for PCA
    print("\nEncoding for PCA...", flush=True)
    embs_a = encode_folder_clip(str(prompt_a_dir), model, processor, device)
    embs_b = encode_folder_clip(str(prompt_b_dir), model, processor, device)

    all_embs = np.vstack([embs_a, embs_b])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_embs)
    coords_a = coords[:len(embs_a)]
    coords_b = coords[len(embs_a):]

    # Build PDF
    pdf_path = run_dir / "prompt_compare.pdf"
    print(f"\nBuilding PDF: {pdf_path}", flush=True)

    with PdfPages(str(pdf_path)) as pdf:
        # Page 1: Prompt A grid
        title_a = f"Prompt A: original ({n_male_a}M / {n_female_a}F)"
        make_pdf_page(pdf, title_a, results_a, str(prompt_a_dir))

        # Page 2: Prompt B grid
        title_b = f"Prompt B: gender-neutral ({n_male_b}M / {n_female_b}F)"
        make_pdf_page(pdf, title_b, results_b, str(prompt_b_dir))

        # Page 3: PCA
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(coords_a[:, 0], coords_a[:, 1],
                   c="dodgerblue", alpha=0.6, s=50,
                   label=f"Prompt A: original ({n_male_a}M/{n_female_a}F)")
        ax.scatter(coords_b[:, 0], coords_b[:, 1],
                   c="crimson", alpha=0.6, s=50,
                   label=f"Prompt B: gender-neutral ({n_male_b}M/{n_female_b}F)")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title("CLIP Embedding PCA — Prompt Comparison")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"\nPDF saved: {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
