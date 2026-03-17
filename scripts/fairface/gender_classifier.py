"""Gender classifier using FairFace (dchen236/FairFace).

Uses ResNet-34 with 18-output multi-task head:
  - neurons 0-6: race (7 classes)
  - neurons 7-8: gender (Male/Female)
  - neurons 9-17: age (9 classes)

Images are resized to 224x224 and classified directly (no face detection).
Fine-tuned weights are used automatically if available.

Weights: res34_fair_align_multi_7_20190809.pt
Download from: https://drive.google.com/drive/folders/1F_pXfbzWvG-bhCpNsRj6F_xsdjpesiFu
"""

import json
import math
import os
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

GENDER_LABELS = ["Male", "Female"]
GENDER_SLICE = slice(7, 9)  # neurons 7-8 in the 18-output head

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent.parent / "models" / "fairface" / "res34_fair_align_multi_7_20190809.pt"
FINETUNED_WEIGHTS = Path(__file__).resolve().parent.parent.parent / "models" / "fairface" / "res34_fairface_finetuned.pt"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(weights_path: Optional[Union[str, Path]] = None, device: str = "cpu") -> nn.Module:
    """Load FairFace ResNet-34 with pretrained weights.

    If no weights_path is given, uses fine-tuned weights if available,
    otherwise falls back to original FairFace weights.
    """
    if weights_path is not None:
        weights_path = Path(weights_path)
    elif FINETUNED_WEIGHTS.exists():
        weights_path = FINETUNED_WEIGHTS
        print(f"Using fine-tuned weights: {weights_path}", flush=True)
    else:
        weights_path = DEFAULT_WEIGHTS
    if not weights_path.exists():
        raise FileNotFoundError(
            f"FairFace weights not found at {weights_path}. "
            "Download from: https://drive.google.com/drive/folders/1F_pXfbzWvG-bhCpNsRj6F_xsdjpesiFu"
        )

    model = torchvision.models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 18)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def _classify_batch(pil_images, model, device):
    """Classify a list of PIL images. Returns (predictions, confidences) tensors."""
    tensors = [TRANSFORM(img) for img in pil_images]
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        outputs = model(batch)
        gender_logits = outputs[:, GENDER_SLICE]
        probs = torch.softmax(gender_logits, dim=1)
        confidences, predictions = probs.max(dim=1)
    return predictions, confidences


def classify_gender(
    image_paths: List[str],
    model: Optional[nn.Module] = None,
    weights_path: Optional[str] = None,
    device: str = "cpu",
    batch_size: int = 32,
) -> List[dict]:
    """Classify gender for a list of image paths.

    Returns list of dicts: {path, predicted_gender, confidence}
    """
    if model is None:
        model = load_model(weights_path, device)

    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        tensors = []
        valid_paths = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(TRANSFORM(img))
                valid_paths.append(p)
            except Exception:
                results.append({"path": str(p), "predicted_gender": "Error", "confidence": 0.0})
                continue

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            outputs = model(batch)
            gender_logits = outputs[:, GENDER_SLICE]
            probs = torch.softmax(gender_logits, dim=1)
            confidences, predictions = probs.max(dim=1)

        for path, pred, conf in zip(valid_paths, predictions, confidences):
            results.append({
                "path": str(path),
                "predicted_gender": GENDER_LABELS[pred.item()],
                "confidence": round(conf.item(), 4),
            })

    return results


def evaluate_gender_balance(pil_images, device, save_dir=None, target_ratio=(50, 50),
                            weights_path=None):
    """Classify gender of PIL images and compute balance metrics.

    If save_dir is provided:
      - Saves images into male/ and female/ subdirectories
      - Saves per-image gender_results.json
      - Saves gender_confidence_boxplot.png

    Returns dict with gender_L, n_male, n_female, gender_conf_mean, male_confs, female_confs.
    """
    model = load_model(weights_path, device)
    predictions, confidences = _classify_batch(pil_images, model, device)

    n = len(pil_images)
    n_male = (predictions == 0).sum().item()
    n_female = n - n_male
    conf_mean = confidences.mean().item()

    # Build per-image results
    per_image = []
    male_confs = []
    female_confs = []
    for idx, (pred, conf) in enumerate(zip(predictions, confidences)):
        label = GENDER_LABELS[pred.item()]
        c = round(conf.item(), 4)
        per_image.append({"image_idx": idx, "gender": label, "confidence": c})
        if pred.item() == 0:
            male_confs.append(c)
        else:
            female_confs.append(c)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        # Save images sorted by gender (clear old results first)
        male_dir = os.path.join(save_dir, "male")
        female_dir = os.path.join(save_dir, "female")
        import shutil
        for d in [male_dir, female_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)
        for idx, (img, pred, conf) in enumerate(zip(pil_images, predictions, confidences)):
            out_dir = male_dir if pred.item() == 0 else female_dir
            img.save(os.path.join(out_dir, f"{idx:03d}_conf{conf.item():.2f}.png"))

        # Save per-image JSON
        with open(os.path.join(save_dir, "gender_results.json"), "w") as f:
            json.dump(per_image, f, indent=2)

        # Save confidence boxplot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        box_data = []
        box_labels = []
        if male_confs:
            box_data.append(male_confs)
            box_labels.append(f"Male (n={len(male_confs)})")
        if female_confs:
            box_data.append(female_confs)
            box_labels.append(f"Female (n={len(female_confs)})")
        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                            widths=0.5, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
            colors = ['#4C72B0', '#DD8452']
            for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        ax.set_ylabel("Confidence")
        ax.set_title(f"Gender Classification Confidence\n{n_male}M / {n_female}F")
        ax.set_ylim(0.45, 1.02)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "gender_confidence_boxplot.png"), dpi=120, bbox_inches='tight')
        plt.close(fig)

    # Gender balance loss: L2 distance from target
    ratio_sum = target_ratio[0] + target_ratio[1]
    t_male = n * target_ratio[0] / ratio_sum
    t_female = n * target_ratio[1] / ratio_sum
    L = math.sqrt((n_male - t_male) ** 2 + (n_female - t_female) ** 2)

    return {
        "gender_L": round(L, 4),
        "n_male": n_male,
        "n_female": n_female,
        "gender_conf_mean": round(conf_mean, 4),
        "male_confs": male_confs,
        "female_confs": female_confs,
    }
