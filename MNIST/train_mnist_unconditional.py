"""
MNIST Unconditional UNet Training with WandB Logging
=====================================================
Trains a UNet2DModel (DDPM/DDIM) on augmented MNIST rotation dataset.

Based on thesis Appendix 6.2.1:
  - UNet2DModel with channels (32, 64, 128), attention in deeper blocks
  - DDPMScheduler with squaredcos_cap_v2 beta schedule, T=1000
  - AdamW, lr=1e-3, 100 epochs, batch_size=256

Usage:
    python train_mnist_unconditional.py
    python train_mnist_unconditional.py --use_ema          # Model B: with EMA
    python train_mnist_unconditional.py --run_name my_run  # custom run name
"""

import argparse
import math
import os
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel
from tqdm import tqdm
import numpy as np

# ─────────────────────────── CLI ────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--use_ema",      action="store_true",
                    help="Enable EMA on model weights (Model B variant)")
parser.add_argument("--ema_decay",    type=float, default=0.9999)
parser.add_argument("--nepochs",      type=int,   default=100)
parser.add_argument("--batch_size",   type=int,   default=256)
parser.add_argument("--lr",           type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--run_name",     type=str,   default=None)
parser.add_argument("--save_dir",     type=str,   default="./checkpoints_uncond")
parser.add_argument("--data_dir",     type=str,   default="./data")
parser.add_argument("--wandb_project",type=str,   default="mnist-unconditional")
parser.add_argument("--log_every",    type=int,   default=50,
                    help="Log loss every N batches")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.save_dir, exist_ok=True)

# ─────────────────────────── WandB ──────────────────────────
run_name = args.run_name or (
    "uncond_unet_EMA" if args.use_ema else "uncond_unet_baseline"
)
wandb.init(
    project=args.wandb_project,
    name=run_name,
    config={
        "model":        "UNet2DModel",
        "architecture": "channels=(32,64,128), attn_in_deep_blocks",
        "noise_schedule": "squaredcos_cap_v2",
        "diffusion_steps": 1000,
        "nepochs":      args.nepochs,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "weight_decay": args.weight_decay,
        "use_ema":      args.use_ema,
        "ema_decay":    args.ema_decay if args.use_ema else None,
        "optimizer":    "AdamW",
        "dataset":      "MNIST-rotation-augmented",
    }
)

# ──────────────────── Dataset ─────────────────────────────────
class MNISTRotationDataset(Dataset):
    """
    Augmented MNIST rotation dataset (thesis Section 4.2.1).
    Loads the base MNIST and applies random rotations + augmentations.
    For full replication, replace with the augmented dataset that includes
    classifier-filtered 90/180/270 degree duplicate entries.
    """
    def __init__(self, root, train=True):
        self.base = datasets.MNIST(
            root, train=train, download=True,
            transform=transforms.Compose([
                transforms.RandomRotation(360),        # Uniform rotation ∈ [0, 360)
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),    # → [-1, 1]
            ])
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img  # only the image; this is unconditional

train_dataset = MNISTRotationDataset(args.data_dir, train=True)
train_loader  = DataLoader(train_dataset, batch_size=args.batch_size,
                           shuffle=True, num_workers=2, pin_memory=True)

# ─────────────────────────── Model ──────────────────────────
# Exact architecture from thesis Appendix 6.2.1
model = UNet2DModel(
    sample_size=28,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(32, 64, 128),
    down_block_types=(
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
    ),
    dropout=0.1,
).to(device)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",
)

# ─────────────────── Optional EMA shadow ────────────────────
# EMA keeps a running exponential average of weights, which often
# improves sample quality at inference without changing training loss.
if args.use_ema:
    ema_model = UNet2DModel(
        sample_size=28, in_channels=1, out_channels=1,
        layers_per_block=2, block_out_channels=(32, 64, 128),
        down_block_types=("DownBlock2D","AttnDownBlock2D","AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D","AttnUpBlock2D","UpBlock2D"),
        dropout=0.1,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    def update_ema(model, ema_model, decay):
        with torch.no_grad():
            for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                ema_p.mul_(decay).add_(p.data, alpha=1.0 - decay)

# ─────────────────────── Optimizer ──────────────────────────
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
)
loss_fn = nn.MSELoss()

wandb.watch(model, log="gradients", log_freq=500)

# ─────────────────────── Training Loop ──────────────────────
global_step = 0
for epoch in range(1, args.nepochs + 1):
    model.train()
    epoch_loss = 0.0
    n_batches  = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.nepochs}", leave=False)
    for batch_idx, images in enumerate(pbar):
        images = images.to(device)                         # [B, 1, 28, 28]
        B = images.shape[0]

        # Sample random timesteps
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (B,), device=device, dtype=torch.long
        )

        # Forward diffusion: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps
        noise  = torch.randn_like(images)
        noisy  = noise_scheduler.add_noise(images, noise, timesteps)

        # Predict noise
        pred_noise = model(noisy, timesteps).sample
        loss = loss_fn(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if args.use_ema:
            update_ema(model, ema_model, args.ema_decay)

        epoch_loss += loss.item()
        n_batches  += 1
        global_step += 1

        # Batch-level logging
        if global_step % args.log_every == 0:
            wandb.log({
                "train/loss_step": loss.item(),
                "train/global_step": global_step,
            }, step=global_step)

        pbar.set_postfix(loss=f"{loss.item():.5f}")

    # Epoch-level logging
    avg_loss = epoch_loss / n_batches
    log_dict = {
        "train/loss_epoch":  avg_loss,
        "train/epoch":       epoch,
    }
    wandb.log(log_dict, step=global_step)
    print(f"Epoch {epoch:4d} | avg_loss={avg_loss:.6f}")

    # Checkpoint every 25 epochs
    if epoch % 25 == 0 or epoch == args.nepochs:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "config": dict(vars(args)),
        }
        if args.use_ema:
            ckpt["ema_state_dict"] = ema_model.state_dict()
        path = os.path.join(args.save_dir, f"{run_name}_epoch{epoch:04d}.pth")
        torch.save(ckpt, path)
        wandb.save(path)
        print(f"  → Saved checkpoint: {path}")

wandb.finish()
print("Training complete.")
