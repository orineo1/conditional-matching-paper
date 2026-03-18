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

import torchvision.transforms as transforms

from gender_classifier import GENDER_SLICE, TRANSFORM

# Training augmentation: color jitter, grayscale, and aspect ratio distortion
# to handle B&W/desaturated outputs and prevent face-shape bias
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class GenderDataset(Dataset):
    """Portrait images with binary gender labels (0=Male, 1=Female)."""

    def __init__(self, paths_and_labels, augment=False):
        self.items = paths_and_labels
        self.transform = TRAIN_TRANSFORM if augment else TRANSFORM

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


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

    train_ds = GenderDataset(train_items, augment=True)
    val_clean_ds = GenderDataset(val_items, augment=False)
    val_aug_ds = GenderDataset(val_items, augment=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_clean_loader = DataLoader(val_clean_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_aug_loader = DataLoader(val_aug_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

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

    def evaluate(loader, label=""):
        model.eval()
        correct, total = 0, 0
        mc, mt, fc, ft = 0, 0, 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs[:, GENDER_SLICE].argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)
                male_mask = labels == 0
                female_mask = labels == 1
                mc += (preds[male_mask] == labels[male_mask]).sum().item()
                mt += male_mask.sum().item()
                fc += (preds[female_mask] == labels[female_mask]).sum().item()
                ft += female_mask.sum().item()
        acc = correct / total
        m_acc = mc / mt if mt > 0 else 0
        f_acc = fc / ft if ft > 0 else 0
        return acc, m_acc, f_acc

    # Output paths for 3 checkpoints
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    path_best_clean = out.parent / out.name.replace(".pt", "_best_clean.pt")
    path_best_aug = out.parent / out.name.replace(".pt", "_best_aug.pt")
    path_latest = out  # latest always overwrites the main path

    best_clean_acc = 0.0
    best_aug_acc = 0.0

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

        # Two val passes: clean and augmented
        clean_acc, clean_m, clean_f = evaluate(val_clean_loader, "clean")
        aug_acc, aug_m, aug_f = evaluate(val_aug_loader, "aug")

        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"Loss {train_loss:.4f} | Train {train_acc:.3f} | "
              f"Val_clean {clean_acc:.3f} (M:{clean_m:.3f} F:{clean_f:.3f}) | "
              f"Val_aug {aug_acc:.3f} (M:{aug_m:.3f} F:{aug_f:.3f})",
              flush=True)

        # Save best on clean val
        if clean_acc > best_clean_acc:
            best_clean_acc = clean_acc
            torch.save(model.state_dict(), path_best_clean)
            print(f"  → Saved best_clean (val_clean={clean_acc:.4f})", flush=True)

        # Save best on augmented val
        if aug_acc > best_aug_acc:
            best_aug_acc = aug_acc
            torch.save(model.state_dict(), path_best_aug)
            print(f"  → Saved best_aug (val_aug={aug_acc:.4f})", flush=True)

        # Always save latest
        torch.save(model.state_dict(), path_latest)

    print(f"\nDone.", flush=True)
    print(f"  Best clean val: {best_clean_acc:.4f} → {path_best_clean}", flush=True)
    print(f"  Best aug val:   {best_aug_acc:.4f} → {path_best_aug}", flush=True)
    print(f"  Latest:         → {path_latest}", flush=True)


if __name__ == "__main__":
    main()
