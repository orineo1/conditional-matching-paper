"""
MNIST Conditional Consistency Model Training with WandB Logging
================================================================
Trains ConsistencyModeliCT to predict P(angle | image) = P((cos,sin) | MNIST image).
Uses a CNN image encoder as the conditioning backbone.

Based on thesis Appendix 6.2.1:
  - Input:  2D circular angle (cos θ, sin θ)
  - Cond:   784-dim flattened MNIST image → CNN → 128-dim embedding
  - Hidden: 128 units, depth=5, add_input_norm=True, add_output_norm=True
  - Output normalized to unit circle
  - 500 epochs, batch_size=256, lr=1e-4, CosineAnnealingLR(T_max=500, eta_min=1e-7)
  - Condition augmentation: Gaussian noise (0.05) + pixel dropout (0.1)

Model variants:
  Baseline  → exact thesis reproduction (no EMA)
  --use_ema → EMA target for consistency distillation

Usage:
    python train_mnist_conditional.py                # Model A: baseline (exact thesis)
    python train_mnist_conditional.py --use_ema      # Model B: with EMA target
    python train_mnist_conditional.py --run_name X   # custom WandB run name
"""

import argparse
import copy
import math
import os
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

# ── make sure your repo is on the path ─────────────────────────
import sys
# adjust this to your actual path on the cluster
REPO_PATH = os.environ.get("REPO_PATH", "/content/GlobalConditional")
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from ConsistencyModels import ConsistencyModeliCT, kerras_boundaries, ict_discretization_schedule, smooth_huber_loss

# ─────────────────────────── CLI ────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--use_ema",        action="store_true",
                    help="Use EMA target model during consistency training (Model B)")
parser.add_argument("--ema_decay",      type=float, default=0.9999)
parser.add_argument("--nepochs",        type=int,   default=500)
parser.add_argument("--batch_size",     type=int,   default=256)
parser.add_argument("--lr",             type=float, default=1e-4)
parser.add_argument("--weight_decay",   type=float, default=1e-4)
parser.add_argument("--run_name",       type=str,   default=None)
parser.add_argument("--save_dir",       type=str,   default="./checkpoints_cond")
parser.add_argument("--data_dir",       type=str,   default="./data")
parser.add_argument("--wandb_project",  type=str,   default="mnist-conditional-cm")
parser.add_argument("--log_every",      type=int,   default=100)
parser.add_argument("--cond_noise",     type=float, default=0.05,
                    help="Gaussian noise std for image conditioning augmentation")
parser.add_argument("--pixel_dropout",  type=float, default=0.1,
                    help="Pixel dropout probability during conditioning")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.save_dir, exist_ok=True)

# ─────────────────────────── WandB ──────────────────────────
run_name = args.run_name or (
    "cond_cm_EMA" if args.use_ema else "cond_cm_baseline"
)
wandb.init(
    project=args.wandb_project,
    name=run_name,
    config={
        "model":            "ConsistencyModeliCT",
        "nfeatures":        2,
        "condition_on":     784,
        "nunits":           128,
        "depth":            5,
        "cond_embed":       "CNN_custom",
        "add_input_norm":   True,
        "add_output_norm":  True,
        "eps":              0.002,
        "nepochs":          args.nepochs,
        "batch_size":       args.batch_size,
        "lr":               args.lr,
        "weight_decay":     args.weight_decay,
        "use_ema":          args.use_ema,
        "ema_decay":        args.ema_decay if args.use_ema else None,
        "optimizer":        "AdamW",
        "scheduler":        f"CosineAnnealingLR(T_max={args.nepochs}, eta_min=1e-7)",
        "cond_noise":       args.cond_noise,
        "pixel_dropout":    args.pixel_dropout,
        "loss":             "smooth_huber + lambda_t weighting (iCT)",
        "ict_s0":           10,
        "ict_s1":           1280,
        "dataset":          "MNIST-rotation-augmented",
    }
)

# ─────────────────── Dataset ──────────────────────────────────
class MNISTAngleDataset(Dataset):
    """
    Returns (flattened_image [784], angle_vec [2]) pairs.
    angle_vec = (cos(θ), sin(θ)) where θ ∈ [0, 360) is a uniformly sampled rotation.

    NOTE: For full thesis replication, load the augmented dataset that includes
    classifier-filtered rotated duplicates (thesis Table 4.5). Here we show the
    straightforward augmentation approach: uniform random rotation of all images.
    """
    def __init__(self, root, train=True):
        self.base = datasets.MNIST(
            root, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]                        # [1, 28, 28]

        # Uniformly sample a rotation angle
        theta_deg = float(torch.rand(1) * 360.0)
        theta_rad = math.radians(theta_deg)

        # Rotate the image
        from torchvision.transforms.functional import rotate
        img_rotated = rotate(img, theta_deg)

        # Circular angle encoding (thesis eq.)
        angle_vec = torch.tensor(
            [math.cos(theta_rad), math.sin(theta_rad)], dtype=torch.float32
        )

        img_flat = img_rotated.view(-1)                # [784]
        return img_flat, angle_vec


train_dataset = MNISTAngleDataset(args.data_dir, train=True)
train_loader  = DataLoader(
    train_dataset, batch_size=args.batch_size,
    shuffle=True, num_workers=2, pin_memory=True
)


# ──────────────── Conditioning augmentation ──────────────────
def augment_conditioning(img_flat: torch.Tensor,
                          noise_std: float = 0.05,
                          dropout_p: float = 0.1) -> torch.Tensor:
    """
    Gaussian noise + pixel dropout applied to flat image conditioning vector.
    Thesis Appendix 6.2.1: σ=0.05 noise, 0.1 pixel dropout.
    """
    x = img_flat + noise_std * torch.randn_like(img_flat)
    mask = (torch.rand_like(img_flat) > dropout_p).float()
    return x * mask


# ─────────────────── CNN image encoder ──────────────────────
# Exact architecture from thesis Figure 6.11 / Appendix 6.2.1
custom_cond = nn.Sequential(
    nn.Unflatten(1, (1, 28, 28)),

    # Block 1
    nn.Conv2d(1, 32, 3, padding=1),
    nn.GroupNorm(8, 32),
    nn.SiLU(),
    nn.Dropout2d(0.1),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.GroupNorm(8, 32),
    nn.SiLU(),
    nn.MaxPool2d(2),                       # 28→14

    # Block 2
    nn.Conv2d(32, 64, 3, padding=1),
    nn.GroupNorm(8, 64),
    nn.SiLU(),
    nn.Dropout2d(0.15),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.GroupNorm(8, 64),
    nn.SiLU(),
    nn.MaxPool2d(2),                       # 14→7

    nn.Dropout2d(0.2),
    nn.Flatten(),                          # 64*7*7 = 3136
    nn.Dropout(0.3),
    nn.Linear(64 * 7 * 7, 128),
    nn.SiLU(),
).to(device)

# ─────────────────────── Model ──────────────────────────────
model = ConsistencyModeliCT(
    nfeatures=2,          # (cos θ, sin θ)
    condition_on=784,     # flattened 28×28 image
    eps=0.002,
    nunits=128,
    depth=5,
    cond_embed_type="linear",     # overridden by cond_embed_model
    cond_embed_model=custom_cond,
    add_input_norm=True,
    add_output_norm=True,
    device=device,
)
model.c_huber = 0.00054 * math.sqrt(2)    # iCT recommended c = 0.00054*sqrt(d)

# ────────────── Optional EMA target model ───────────────────
# With use_ema=True, the target in the consistency loss uses a
# slowly-updated shadow copy (mu ≠ 0), similar to original CD.
# With use_ema=False (baseline), mu=0 so target == current model (iCT).
if args.use_ema:
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    def update_ema(model, ema_model, decay):
        with torch.no_grad():
            for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                ema_p.mul_(decay).add_(p.data, alpha=1.0 - decay)
else:
    ema_model = model  # mu=0: target IS the model (iCT default)

# ─────────────────────── Optimizer ──────────────────────────
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.nepochs, eta_min=1e-7
)

wandb.watch(model, log="gradients", log_freq=500)

# ─────────────────────── Training Loop ──────────────────────
global_step = 0
for epoch in range(1, args.nepochs + 1):
    model.train()
    epoch_loss = 0.0
    n_batches  = 0

    # iCT discretization curriculum N(k)
    N = ict_discretization_schedule(epoch, args.nepochs, s0=10, s1=1280)
    boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.nepochs}  N={N}", leave=False)
    for images_flat, angle_vecs in pbar:
        images_flat = images_flat.to(device)     # [B, 784]
        angle_vecs  = angle_vecs.to(device)      # [B, 2]
        B = angle_vecs.shape[0]

        # Condition augmentation (Gaussian noise + pixel dropout)
        cond = augment_conditioning(images_flat, args.cond_noise, args.pixel_dropout)

        # iCT noise-level sampling via erf distribution
        t_idx = model.ict_noise_sampling(boundaries, B, P_mean=-1.1, P_std=2.0, device=device)
        t0 = boundaries[t_idx]          # [B, 1]
        t1 = boundaries[t_idx + 1]      # [B, 1]

        z = torch.randn_like(angle_vecs)

        # ── Consistency training loss ───────────────────────
        # target  = stop_grad(f_ema(x + z*t0, t0, cond))
        # student = f_theta(x + z*t1, t1, cond)
        with torch.no_grad():
            x_t0 = angle_vecs + z * t0
            x1_target = ema_model(x_t0, t0, cond=cond)

        x_t1    = angle_vecs + z * t1
        x2_pred = model(x_t1, t1, cond=cond)

        # iCT: λ(t) = 1/(t1-t0) weighting + smooth Huber loss
        lambda_t    = 1.0 / (t1 - t0 + 1e-8)
        huber       = smooth_huber_loss(x1_target, x2_pred, c=model.c_huber)
        loss        = (lambda_t * huber).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if args.use_ema:
            update_ema(model, ema_model, args.ema_decay)

        epoch_loss += loss.item()
        n_batches  += 1
        global_step += 1

        if global_step % args.log_every == 0:
            wandb.log({
                "train/loss_step":  loss.item(),
                "train/global_step": global_step,
                "train/N_boundaries": N,
            }, step=global_step)

        pbar.set_postfix(loss=f"{loss.item():.5f}", N=N)

    scheduler.step()
    avg_loss = epoch_loss / n_batches

    wandb.log({
        "train/loss_epoch":  avg_loss,
        "train/epoch":       epoch,
        "train/lr":          scheduler.get_last_lr()[0],
        "train/N_curriculum": N,
    }, step=global_step)
    print(f"Epoch {epoch:4d} | avg_loss={avg_loss:.6f} | N={N} | lr={scheduler.get_last_lr()[0]:.2e}")

    # Checkpoint every 100 epochs
    if epoch % 100 == 0 or epoch == args.nepochs:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": model.get_config(),
            "loss": avg_loss,
            "args": vars(args),
        }
        if args.use_ema:
            ckpt["ema_state_dict"] = ema_model.state_dict()
        path = os.path.join(args.save_dir, f"{run_name}_epoch{epoch:04d}.pt")
        torch.save(ckpt, path)
        wandb.save(path)
        print(f"  → Saved checkpoint: {path}")

wandb.finish()
print("Training complete.")

# ═══════════════════════════════════════════════════════════
# NOTE: to use the REAL augmented dataset instead of on-the-fly
# generation, replace the DataLoader block above with:
#
#   from create_mnist_rotation_dataset import load_dataset_as_loaders
#   train_loader, val_loader, _ = load_dataset_as_loaders(
#       "data/mnist_rotation/mnist_rotation_thesis.pt",    # or every10
#       batch_size=args.batch_size)
#
# The loader already yields (images_flat [B,784], angle_vec [B,2])
# which is exactly what this training loop expects.
# ═══════════════════════════════════════════════════════════
