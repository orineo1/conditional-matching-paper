"""Fine-tune FairFace ResNet-34 gender head on generated/stylized portraits.

Only the final FC layer (512→18) is trained; all conv layers are frozen.
Loss is computed on gender neurons only (outputs[:, 7:9]).

Data layout:
  data_dir/man/   — male portrait PNGs
  data_dir/woman/ — female portrait PNGs
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from gender_classifier import GENDER_SLICE, TRANSFORM


class GenderDataset(Dataset):
    """Portrait images with binary gender labels (0=Male, 1=Female).

    Skips MTCNN face detection — portraits are already tightly cropped.
    Just resizes to 224x224 with the same normalization as inference.
    """

    def __init__(self, paths_and_labels):
        self.items = paths_and_labels

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return TRANSFORM(img), label


def main():
    parser = argparse.ArgumentParser(description="Fine-tune FairFace gender head")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with man/ and woman/ subdirs")
    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to original FairFace weights")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save fine-tuned weights")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    device = args.device
    data_dir = Path(args.data_dir)

    # Collect paths and labels (skip macOS ._ resource forks)
    all_items = []
    for img_path in sorted((data_dir / "man").glob("*.png")):
        if img_path.name.startswith("._"):
            continue
        all_items.append((img_path, 0))  # Male
    for img_path in sorted((data_dir / "woman").glob("*.png")):
        if img_path.name.startswith("._"):
            continue
        all_items.append((img_path, 1))  # Female
    print(f"Found {len(all_items)} images ({sum(1 for _, l in all_items if l == 0)} man, "
          f"{sum(1 for _, l in all_items if l == 1)} woman)", flush=True)

    # Train/val split
    random.seed(42)
    random.shuffle(all_items)
    n_val = int(len(all_items) * args.val_split)
    val_items = all_items[:n_val]
    train_items = all_items[n_val:]
    print(f"Train: {len(train_items)}, Val: {len(val_items)}", flush=True)

    train_ds = GenderDataset(train_items)
    val_ds = GenderDataset(val_items)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load model, freeze everything except fc
    model = torchvision.models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 18)
    model.load_state_dict(torch.load(args.weights_path, map_location=device, weights_only=True))
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable} (fc layer only)", flush=True)

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            gender_logits = outputs[:, GENDER_SLICE]
            loss = criterion(gender_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = gender_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        train_acc = correct / total
        train_loss = total_loss / total

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        male_correct, male_total, female_correct, female_total = 0, 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                gender_logits = outputs[:, GENDER_SLICE]
                preds = gender_logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
                # Per-class accuracy
                male_mask = labels == 0
                female_mask = labels == 1
                male_correct += (preds[male_mask] == labels[male_mask]).sum().item()
                male_total += male_mask.sum().item()
                female_correct += (preds[female_mask] == labels[female_mask]).sum().item()
                female_total += female_mask.sum().item()

        val_acc = val_correct / val_total
        male_acc = male_correct / male_total if male_total > 0 else 0
        female_acc = female_correct / female_total if female_total > 0 else 0

        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"Loss {train_loss:.4f} | Train {train_acc:.3f} | "
              f"Val {val_acc:.3f} (M:{male_acc:.3f} F:{female_acc:.3f})",
              flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output_path)
            print(f"  → Saved best model (val_acc={val_acc:.4f})", flush=True)

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}", flush=True)
    print(f"Saved to: {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
