"""
CLIP gender classification + boxplot for DPS vs unguided photo sets.
Usage: python scripts/gender_boxplot.py --runs_json <config>
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import open_clip
    USE_OPEN_CLIP = True
except ImportError:
    USE_OPEN_CLIP = False

from transformers import CLIPModel, CLIPProcessor


def load_clip(device):
    if USE_OPEN_CLIP:
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        model = model.to(device).eval()
        return model, preprocess, tokenizer, "open_clip"
    else:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        return model, processor, None, "hf"


def get_text_features(model, tokenizer_or_processor, prompts, device, mode):
    if mode == "open_clip":
        tokens = tokenizer_or_processor(prompts).to(device)
        with torch.no_grad():
            tf = model.encode_text(tokens)
        return F.normalize(tf, dim=-1)
    else:
        inputs = tokenizer_or_processor(text=prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            tf = model.get_text_features(**inputs)
        # handle both tensor and dataclass outputs
        if not isinstance(tf, torch.Tensor):
            tf = tf.pooler_output if hasattr(tf, 'pooler_output') else tf.last_hidden_state[:, 0]
        return F.normalize(tf, dim=-1)


def get_image_features(model, processor_or_preprocess, img_path, device, mode):
    img = Image.open(img_path).convert("RGB")
    if mode == "open_clip":
        x = processor_or_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            f = model.encode_image(x)
    else:
        inputs = processor_or_preprocess(images=img, return_tensors="pt")
        with torch.no_grad():
            f = model.get_image_features(**inputs)
        if not isinstance(f, torch.Tensor):
            f = f.pooler_output if hasattr(f, 'pooler_output') else f.last_hidden_state[:, 0]
    return F.normalize(f, dim=-1)


def classify_folder(folder, model, proc, tokenizer, text_feats, device, mode):
    photos = sorted(glob.glob(os.path.join(folder, "*.png")))
    probs = []
    for p in photos:
        img_feat = get_image_features(model, proc, p, device, mode)
        sims = (img_feat @ text_feats.T * 100).squeeze(0)
        p_male = torch.softmax(sims, dim=0)[0].item()
        probs.append(p_male)
    return np.array(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, default="CLIP Gender Classification")
    parser.add_argument("--output", type=str, default="output/gender_boxplot.pdf")
    parser.add_argument("--sets", nargs="+",
                        help="label:dps_dir:unguided_dir triplets",
                        default=[
                            "Binary:output/dps_binary_n25/photos_dps:output/dps_binary_n25/photos_regular",
                            "Interpolated:output/dps_interpolated_n25/photos_dps:output/dps_interpolated_n25/photos_regular",
                        ])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, open_clip: {USE_OPEN_CLIP}")

    model, proc, tokenizer, mode = load_clip(device)
    text_feats = get_text_features(model, tokenizer or proc,
                                   ["a photo of a man", "a photo of a woman"], device, mode)

    parsed = [s.split(":") for s in args.sets]

    all_results = {}
    for label, dps_dir, ung_dir in parsed:
        print(f"Classifying {label} DPS ({dps_dir})...")
        all_results[f"{label} DPS"] = classify_folder(dps_dir, model, proc, tokenizer, text_feats, device, mode)
        print(f"Classifying {label} Unguided ({ung_dir})...")
        all_results[f"{label} Unguided"] = classify_folder(ung_dir, model, proc, tokenizer, text_feats, device, mode)

    for k, v in all_results.items():
        print(f"  {k}: {v.mean()*100:.1f}% male (n={len(v)})")

    # Plot
    n_groups = len(parsed)
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 6), sharey=True)
    if n_groups == 1:
        axes = [axes]
    fig.suptitle(args.title, fontsize=13, fontweight="bold")

    colors = {"DPS": "steelblue", "Unguided": "lightgray"}

    for ax, (label, dps_dir, ung_dir) in zip(axes, parsed):
        keys = [f"{label} DPS", f"{label} Unguided"]
        data = [all_results[k] for k in keys]
        bp = ax.boxplot(data, labels=["DPS", "Unguided"], patch_artist=True, widths=0.5)
        for box, key in zip(bp["boxes"], keys):
            box.set_facecolor(colors["DPS"] if "DPS" in key else colors["Unguided"])
        for i, d in enumerate(data):
            ax.scatter(np.random.normal(i + 1, 0.04, len(d)), d,
                       alpha=0.25, s=12, color="black", zorder=3)
            ax.text(i + 1, -0.08, f"{d.mean()*100:.1f}% M",
                    ha="center", fontsize=9, color="dimgray")
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="50% male")
        ax.set_title(label, fontsize=12)
        ax.set_ylabel("P(male)")
        ax.set_ylim(-0.12, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
