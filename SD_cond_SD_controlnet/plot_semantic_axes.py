"""
Project age_gender_combined reference images onto semantic CLIP axes:
  x-axis: gender axis  (man − woman text embeddings)
  y-axis: age axis     (old − young text embeddings, orthogonalized against gender)
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from clip_utils import load_clip_model, encode_images_clip

OUTPUT_PATH = "validation_plots/semantic_axes_age_gender.png"
REF_DIR     = "reference_images"
N_SAMPLE    = 50  # per gender


def encode_text(texts, model, processor, device):
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


def load_images_and_ages(folder, n, seed=42):
    ages_path = os.path.join(folder, "ages.json")
    with open(ages_path) as f:
        ages_data = json.load(f)   # list of {filename, age}
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ages_data), size=min(n, len(ages_data)), replace=False)
    selected = [ages_data[i] for i in idx]
    imgs, ages = [], []
    for item in selected:
        img = Image.open(os.path.join(folder, item["filename"])).convert("RGB")
        imgs.append(img)
        ages.append(item["age"])
    return imgs, np.array(ages)


def encode_pil_list(pil_imgs, model, processor, device, batch=16):
    all_embs = []
    for i in range(0, len(pil_imgs), batch):
        batch_imgs = pil_imgs[i:i+batch]
        tensors = torch.cat([TF.to_tensor(img).unsqueeze(0) for img in batch_imgs]).to(device)
        with torch.no_grad():
            emb = encode_images_clip(tensors, model, processor)
        all_embs.append(emb.cpu().float().numpy())
    return np.vstack(all_embs)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs("validation_plots", exist_ok=True)

    model, processor = load_clip_model(device)

    # ── Define semantic axes via text ────────────────────────────────────────
    print("Encoding text axes...")
    man_emb   = encode_text(["a photo of a man"],    model, processor, device)[0]
    woman_emb = encode_text(["a photo of a woman"],  model, processor, device)[0]
    old_emb   = encode_text(["a photo of an old person"],   model, processor, device)[0]
    young_emb = encode_text(["a photo of a young person"],  model, processor, device)[0]

    gender_axis = man_emb - woman_emb
    gender_axis = gender_axis / np.linalg.norm(gender_axis)

    age_raw = old_emb - young_emb
    # Orthogonalize age axis against gender axis (Gram-Schmidt)
    age_axis = age_raw - np.dot(age_raw, gender_axis) * gender_axis
    age_axis = age_axis / np.linalg.norm(age_axis)

    print(f"  Gender axis norm: {np.linalg.norm(man_emb - woman_emb):.4f}")
    print(f"  Age-gender dot (after ortho): {np.dot(age_axis, gender_axis):.6f}")

    # ── Load reference images ────────────────────────────────────────────────
    print("Loading reference images...")
    woman_dir = os.path.join(REF_DIR, "age_woman")
    man_dir   = os.path.join(REF_DIR, "age_man")

    imgs_w, ages_w = load_images_and_ages(woman_dir, N_SAMPLE)
    imgs_m, ages_m = load_images_and_ages(man_dir,   N_SAMPLE)
    print(f"  Women: {len(imgs_w)}  Men: {len(imgs_m)}")

    # ── Encode images ────────────────────────────────────────────────────────
    print("Encoding images...")
    emb_w = encode_pil_list(imgs_w, model, processor, device)
    emb_m = encode_pil_list(imgs_m, model, processor, device)

    # L2-normalize
    emb_w = emb_w / np.linalg.norm(emb_w, axis=1, keepdims=True)
    emb_m = emb_m / np.linalg.norm(emb_m, axis=1, keepdims=True)

    all_emb  = np.vstack([emb_w, emb_m])
    all_ages = np.concatenate([ages_w, ages_m])
    all_gender = np.array([0]*len(imgs_w) + [1]*len(imgs_m))  # 0=woman, 1=man

    # ── Project ─────────────────────────────────────────────────────────────
    x = all_emb @ gender_axis   # gender score
    y = all_emb @ age_axis      # age score

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))

    cmap = plt.cm.RdPu
    for i in range(len(all_emb)):
        color = cmap((all_ages[i] - 1) / 98)
        marker = "^" if all_gender[i] == 1 else "o"
        ax.scatter(x[i], y[i], c=[color], marker=marker, s=60, alpha=0.85,
                   edgecolors="none")

    # Legend: gender markers
    ax.scatter([], [], marker="^", c="gray", s=60, label="Man")
    ax.scatter([], [], marker="o", c="gray", s=60, label="Woman")
    ax.legend(fontsize=10)

    # Colorbar for age
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 99))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Age", fontsize=11)

    ax.set_xlabel("← Female          Gender axis (man − woman)          Male →", fontsize=10)
    ax.set_ylabel("← Young           Age axis (old − young)             Old →", fontsize=10)
    ax.set_title("Reference images projected onto semantic CLIP axes\n(age axis orthogonalized against gender axis)", fontsize=11)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.4)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
