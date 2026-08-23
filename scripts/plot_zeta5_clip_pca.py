"""
Plot eval_photos_guided and eval_photos_unguided from run2_zeta5
in CLIP PC1/PC2 space, colored by CLIP gender score (P(man)).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(BASE, "autoresearch/output/run2_zeta5")
OUT_PATH = os.path.join(BASE, "autoresearch/output/run2_zeta5/clip_pca_gender.png")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("Loading CLIP...")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model.eval()

# Gender text embeddings
prompts = ["a photo of a man", "a photo of a woman"]
text_inputs = processor.tokenizer(prompts, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_feats = model.get_text_features(**text_inputs)
    if not isinstance(text_feats, torch.Tensor):
        text_feats = text_feats.pooler_output
text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)


def encode_dir(img_dir):
    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    embeddings = []
    gender_scores = []
    for fname in files:
        img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            img_feats = model.get_image_features(**inputs)
            if not isinstance(img_feats, torch.Tensor):
                img_feats = img_feats.pooler_output
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        embeddings.append(img_feats.squeeze(0).cpu().float().numpy())
        # gender score: softmax of [man_sim, woman_sim]
        sims = (img_feats @ text_feats.T * 100).squeeze(0).cpu().float()
        probs = torch.softmax(sims, dim=0).numpy()
        gender_scores.append(float(probs[0]))  # P(man)
    return np.array(embeddings), np.array(gender_scores)


print("Encoding guided images...")
guided_embs, guided_scores = encode_dir(os.path.join(RUN_DIR, "eval_photos_guided"))
print(f"  {len(guided_embs)} images, mean P(man)={guided_scores.mean():.3f}")

print("Encoding unguided images...")
unguided_embs, unguided_scores = encode_dir(os.path.join(RUN_DIR, "eval_photos_unguided"))
print(f"  {len(unguided_embs)} images, mean P(man)={unguided_scores.mean():.3f}")

# PCA on combined set
from sklearn.decomposition import PCA
all_embs = np.vstack([guided_embs, unguided_embs])
pca = PCA(n_components=2)
all_proj = pca.fit_transform(all_embs)
n_guided = len(guided_embs)
guided_proj = all_proj[:n_guided]
unguided_proj = all_proj[n_guided:]

print(f"PC1 var: {pca.explained_variance_ratio_[0]:.3f}, PC2 var: {pca.explained_variance_ratio_[1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("run2_zeta5 — CLIP PCA (colored by P(man))", fontsize=13)

vmin, vmax = 0.0, 1.0
cmap = "bwr"  # blue=woman, red=man

for ax, proj, scores, label in [
    (axes[0], guided_proj, guided_scores, f"DPS guided (n={n_guided}, μ={guided_scores.mean():.2f})"),
    (axes[1], unguided_proj, unguided_scores, f"Unguided (n={len(unguided_embs)}, μ={unguided_scores.mean():.2f})"),
]:
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=scores, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=60, alpha=0.85, edgecolors='none')
    ax.set_title(label, fontsize=11)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

plt.colorbar(sc, ax=axes, label="P(man)", fraction=0.02, pad=0.04)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
