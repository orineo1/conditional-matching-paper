"""
classifier.py
=============
Noise-robust MNIST digit classifier.

Exports
-------
ImprovedCNN            – model class
AddGaussianNoise       – transform
make_deterministic     – seed helper
train_robust_classifier – training function
load_or_train_classifier – convenience: load from disk or train
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Constants ─────────────────────────────────────────────────────────────
NORM_MEAN = 0.1307
NORM_STD  = 0.3081


# ── 1. Determinism ────────────────────────────────────────────────────────
def make_deterministic(seed: int = 42) -> None:
    """Fix all random seeds and force cuDNN determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


# ── 2. Gaussian noise augmentation ────────────────────────────────────────
class AddGaussianNoise:
    """Add i.i.d. Gaussian noise to a tensor (used in training transforms)."""

    def __init__(self, mean: float = 0.0, std: float = 0.15):
        self.mean = mean
        self.std  = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


# ── 3. Architecture ───────────────────────────────────────────────────────
class ImprovedCNN(nn.Module):
    """
    Three-block CNN classifier for MNIST digits (noise-robust variant).

    Changes vs. the original ImprovedCNN
    ─────────────────────────────────────
    • input_dropout  : Dropout2d(0.1) on raw pixels
    • dropout_mid1/2 : Dropout2d(0.1) between the two convs in each block
    • label_smoothing in the loss (set during training, not here)
    """

    def __init__(self):
        super().__init__()
        self.input_dropout = nn.Dropout2d(0.1)

        # Block 1  (28×28 → 14×14)
        self.conv1        = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1          = nn.BatchNorm2d(32)
        self.dropout_mid1 = nn.Dropout2d(0.1)
        self.conv2        = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2          = nn.BatchNorm2d(32)
        self.pool1        = nn.MaxPool2d(2)
        self.dropout1     = nn.Dropout2d(0.25)

        # Block 2  (14×14 → 7×7)
        self.conv3        = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3          = nn.BatchNorm2d(64)
        self.dropout_mid2 = nn.Dropout2d(0.1)
        self.conv4        = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4          = nn.BatchNorm2d(64)
        self.pool2        = nn.MaxPool2d(2)
        self.dropout2     = nn.Dropout2d(0.25)

        # Block 3  (7×7 → 3×3)
        self.conv5    = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5      = nn.BatchNorm2d(128)
        self.pool3    = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout2d(0.25)

        # Fully connected
        self.fc1         = nn.Linear(128 * 3 * 3, 256)
        self.bn_fc1      = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2         = nn.Linear(256, 128)
        self.bn_fc2      = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(0.5)
        self.fc3         = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)

        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout_mid1(x)
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout_mid2(x)
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout2(x)

        # Block 3
        x = self.pool3(F.relu(self.bn5(self.conv5(x))))
        x = self.dropout3(x)

        # FC
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout_fc1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout_fc2(F.relu(self.bn_fc2(self.fc2(x))))
        return self.fc3(x)


# ── 4. Training ───────────────────────────────────────────────────────────
def train_robust_classifier(
    save_path: str,
    epochs:     int   = 10,
    batch_size: int   = 128,
    lr:         float = 1e-3,
    device:     str   = 'cpu',
    seed:       int   = 42,
) -> ImprovedCNN:
    """
    Train ImprovedCNN on MNIST with Gaussian noise augmentation.

    Parameters
    ----------
    save_path  : where to write the best checkpoint (.pth)
    epochs     : number of training epochs (10 is enough for noise stability)
    batch_size : mini-batch size
    lr         : Adam learning rate
    device     : 'cuda' or 'cpu'
    seed       : random seed for full reproducibility

    Returns
    -------
    ImprovedCNN in eval mode, loaded with the best weights.
    """
    make_deterministic(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_tf = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.15),
        transforms.Normalize((NORM_MEAN,), (NORM_STD,)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((NORM_MEAN,), (NORM_STD,)),
    ])

    train_loader = DataLoader(
        datasets.MNIST('./data', train=True,  download=True, transform=train_tf),
        batch_size=batch_size, shuffle=True, num_workers=0, generator=g,
    )
    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=test_tf),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    model     = ImprovedCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc  = 0.0

    n_params = sum(p.numel() for p in model.parameters())
    print(f'[Classifier] Training — {n_params:,} parameters, device={device}')

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            criterion(model(data), target).backward()
            optimizer.step()

        # ── eval ──
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                correct += model(data).argmax(1).eq(target).sum().item()

        acc = 100.0 * correct / len(test_loader.dataset)
        print(f'  Epoch {epoch:2d}/{epochs} | Test accuracy: {acc:.2f}%', end='')

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print(f'  ← new best', end='')
        print()

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    print(f'[Classifier] Done — best test accuracy: {best_acc:.2f}%')
    return model


# ── 5. Convenience loader ─────────────────────────────────────────────────
def load_or_train_classifier(
    save_path:  str,
    device:     str   = 'cpu',
    epochs:     int   = 10,
    batch_size: int   = 128,
    lr:         float = 1e-3,
    seed:       int   = 42,
) -> ImprovedCNN:
    """
    Load classifier weights from *save_path* if the file exists,
    otherwise train from scratch and save to *save_path*.

    Returns ImprovedCNN in eval mode.
    """
    if os.path.exists(save_path):
        print(f'[Classifier] Loading weights from {save_path}')
        model = ImprovedCNN().to(device)
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()
        print('[Classifier] Loaded ✓')
        return model

    print(f'[Classifier] No checkpoint found at {save_path} — training from scratch...')
    return train_robust_classifier(
        save_path=save_path, epochs=epochs,
        batch_size=batch_size, lr=lr,
        device=device, seed=seed,
    )
