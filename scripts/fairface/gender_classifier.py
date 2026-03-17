"""Minimal gender classifier extracted from FairFace (dchen236/FairFace).

Uses ResNet-34 with 18-output multi-task head:
  - neurons 0-6: race (7 classes)
  - neurons 7-8: gender (Male/Female)
  - neurons 9-17: age (9 classes)

IMPORTANT: FairFace was trained on aligned face crops (res34_fair_align_multi_7).
This module uses MTCNN face detection to crop faces before classification,
matching the original FairFace pipeline (which uses dlib alignment).

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

# Padding factor around detected face box (0.25 matches FairFace's dlib padding)
FACE_PADDING = 0.25


def _load_mtcnn(device: str):
    """Load MTCNN face detector (from facenet-pytorch)."""
    from facenet_pytorch import MTCNN
    return MTCNN(keep_all=True, device=device, min_face_size=50)


def _crop_face(img: Image.Image, mtcnn) -> Optional[Image.Image]:
    """Detect and crop the largest face with padding, mimicking dlib's get_face_chip."""
    boxes, probs = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        return None

    # Pick the largest face (by area)
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    idx = max(range(len(areas)), key=lambda i: areas[i])
    x1, y1, x2, y2 = boxes[idx]

    # Add padding (matching FairFace's dlib padding=0.25)
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = w * FACE_PADDING, h * FACE_PADDING
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(img.width, x2 + pad_w)
    y2 = min(img.height, y2 + pad_h)

    return img.crop((int(x1), int(y1), int(x2), int(y2)))


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

    Uses MTCNN to detect and crop faces before classification.
    Returns list of dicts: {path, predicted_gender, confidence, face_detected}
    """
    if model is None:
        model = load_model(weights_path, device)

    mtcnn = _load_mtcnn(device)
    n_detected = 0
    n_fallback = 0

    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        tensors = []
        valid_paths = []
        face_flags = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                face_crop = _crop_face(img, mtcnn)
                if face_crop is not None:
                    tensors.append(TRANSFORM(face_crop))
                    n_detected += 1
                    face_flags.append(True)
                else:
                    # Fallback: use full image if no face detected
                    tensors.append(TRANSFORM(img))
                    n_fallback += 1
                    face_flags.append(False)
                valid_paths.append(p)
            except Exception as e:
                results.append({"path": str(p), "predicted_gender": "Error", "confidence": 0.0, "face_detected": False})
                continue

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            outputs = model(batch)
            gender_logits = outputs[:, GENDER_SLICE]
            probs = torch.softmax(gender_logits, dim=1)
            confidences, predictions = probs.max(dim=1)

        for path, pred, conf, face_ok in zip(valid_paths, predictions, confidences, face_flags):
            results.append({
                "path": str(path),
                "predicted_gender": GENDER_LABELS[pred.item()],
                "confidence": round(conf.item(), 4),
                "face_detected": face_ok,
            })

    print(f"Face detection: {n_detected} detected, {n_fallback} fallback (no face found)", flush=True)
    return results
