"""
train_conditional.py
====================
Entry point for the conditional angle consistency model pipeline.

Steps:
  1. Train MNIST classifier
  2. Build AugmentedMNISTDataset
  3. Train CircularAngleConsistencyModel (iCT)

Usage:
    python train_conditional.py
    python train_conditional.py --resume checkpoints/circular_ict_epoch_25.pt
    python train_conditional.py --epochs 500 --threshold 0.999
"""

import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import train_classifier, AugmentedMNISTDataset
from model import CircularAngleConsistencyModel


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clf_epochs", type=int,   default=15,     help="Classifier training epochs")
    parser.add_argument("--clf_batch",  type=int,   default=128)
    parser.add_argument("--threshold",  type=float, default=0.9999, help="Confidence threshold for augmentation")
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--epochs",     type=int,   default=300,    help="Consistency model epochs")
    parser.add_argument("--ckpt_dir",   type=str,   default="checkpoints")
    parser.add_argument("--resume",     type=str,   default=None,   help="Checkpoint path to resume from")
    args = parser.parse_args()

    print(f"Device: {device}")

    # 1. Train classifier
    classifier = train_classifier(epochs=args.clf_epochs, batch_size=args.clf_batch, device=device)

    # 2. Build dataset
    dataset    = AugmentedMNISTDataset(classifier, confidence_threshold=args.threshold, device=device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)} samples")

    # 3. Train consistency model
    model = CircularAngleConsistencyModel(
        nfeatures=2, img_features=784, eps=0.002, nunits=128, depth=5, device=device
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    losses = model.train_model(
        dataloader,
        nepochs=args.epochs,
        device=device,
        use_improved_training=True,
        save_dir=args.ckpt_dir,
        checkpoint_path=args.resume,
    )

    # 4. Plot loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title("iCT Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.show()
