"""Minimal gender classifier extracted from FairFace (dchen236/FairFace).

Uses ResNet-34 with 18-output multi-task head:
  - neurons 0-6: race (7 classes)
  - neurons 7-8: gender (Male/Female)
  - neurons 9-17: age (9 classes)

Weights: res34_fair_align_multi_7_20190809.pt
Download from: https://drive.google.com/drive/folders/1F_pXfbzWvG-bhCpNsRj6F_xsdjpesiFu
"""

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

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(weights_path: Optional[Union[str, Path]] = None, device: str = "cpu") -> nn.Module:
    """Load FairFace ResNet-34 with pretrained weights."""
    weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
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
            except Exception as e:
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
