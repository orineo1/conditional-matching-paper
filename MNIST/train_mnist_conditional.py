"""
MNIST Conditional Consistency Model Training with WandB Logging
================================================================
Matches notebook training exactly, but uses the every10_balanced dataset.

Key differences from notebook:
  - Uses pre-built every10_balanced rotation dataset instead of on-the-fly
  - kerras_boundaries(10.0, 0.002, N, 7.0) matches notebook exactly
  - scheduler.step() per batch, not per epoch (matches notebook)
  - No conditioning augmentation (was causing posterior collapse)
  - depth=6 matches notebook checkpoint

Usage:
    python train_mnist_conditional.py                          # default
    python train_mnist_conditional.py --use_ema               # with EMA
    python train_mnist_conditional.py --dataset every10        # full dataset
"""

import argparse
import copy
import math
import os
import sys
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import rotate
from tqdm import tqdm

REPO_PATH = os.environ.get("REPO_PATH", "/sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST")
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from ConsistencyModels import ConsistencyModeliCT, kerras_boundaries, ict_discretization_schedule, smooth_huber_loss

# ─────────────────────────── CLI ────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",        type=str,   default="every10_balanced",
                    choices=["every10_balanced", "every10",
                             "every45_balanced", "every45",
                             "thesis_balanced", "thesis",
                             "vanilla"])
parser.add_argument("--use_ema",        action="store_true")
parser.add_argument("--ema_decay",      type=float, default=0.9999)
parser.add_argument("--nepochs",        type=int,   default=500)
parser.add_argument("--batch_size",     type=int,   default=256)
parser.add_argument("--lr",             type=float, default=1e-4)
parser.add_argument("--weight_decay",   type=float, default=1e-4)
parser.add_argument("--depth",          type=int,   default=6,
                    help="Depth of consistency model MLP (default: 6, matches notebook)")
parser.add_argument("--run_name",       type=str,   default=None)
parser.add_argument("--save_dir",       type=str,   default="./checkpoints_cond")
parser.add_argument("--data_dir",       type=str,   default="./data")
parser.add_argument("--wandb_project",  type=str,   default="mnist-conditional-cm")
parser.add_argument("--log_every",      type=int,   default=100)
parser.add_argument("--ict_s0",         type=int,   default=10)
parser.add_argument("--ict_s1",         type=int,   default=1280)
parser.add_argument("--scheduler",      type=str,   default="cosine",
                    choices=["cosine", "cosine_restarts", "constant"])
parser.add_argument("--restart_period", type=int,   default=100)
parser.add_argument("--eta_min",        type=float, default=1e-7)
# ── Notebook-matching flags ──────────────────────────────────
parser.add_argument("--cond_augment",   action="store_true",
                    help="Enable conditioning augmentation (noise+dropout). Default: off.")
parser.add_argument("--cond_noise",     type=float, default=0.05)
parser.add_argument("--pixel_dropout",  type=float, default=0.1)
parser.add_argument("--scheduler_per_batch", action="store_true",
                    help="Step scheduler per batch instead of per epoch (matches notebook). Default: off.")
parser.add_argument("--kerras_sigma_max", type=float, default=10.0,
                    help="kerras_boundaries sigma_max (notebook=10.0, old script=7.0)")
parser.add_argument("--kerras_rho",     type=float, default=7.0,
                    help="kerras_boundaries rho (notebook=7.0, old script=80.0)")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.save_dir, exist_ok=True)

# ─────────────────────────── WandB ──────────────────────────
run_name = args.run_name or f"cond_cm_{'EMA' if args.use_ema else 'baseline'}_{args.dataset}"
wandb.init(
    project=args.wandb_project,
    name=run_name,
    config={
        "model":                "ConsistencyModeliCT",
        "dataset":              args.dataset,
        "nfeatures":            2,
        "condition_on":         784,
        "nunits":               128,
        "depth":                args.depth,
        "cond_embed":           "CNN_custom",
        "add_input_norm":       True,
        "add_output_norm":      True,
        "eps":                  0.002,
        "nepochs":              args.nepochs,
        "batch_size":           args.batch_size,
        "lr":                   args.lr,
        "weight_decay":         args.weight_decay,
        "use_ema":              args.use_ema,
        "ema_decay":            args.ema_decay if args.use_ema else None,
        "optimizer":            "AdamW",
        "scheduler":            args.scheduler,
        "scheduler_per_batch":  args.scheduler_per_batch,
        "restart_period":       args.restart_period,
        "eta_min":              args.eta_min,
        "cond_augment":         args.cond_augment,
        "cond_noise":           args.cond_noise if args.cond_augment else None,
        "pixel_dropout":        args.pixel_dropout if args.cond_augment else None,
        "loss":                 "smooth_huber + lambda_t weighting (iCT)",
        "ict_s0":               args.ict_s0,
        "ict_s1":               args.ict_s1,
        "kerras_sigma_max":     args.kerras_sigma_max,
        "kerras_rho":           args.kerras_rho,
    }
)

# ─────────────────────────── Datasets ───────────────────────
DATASET_FILES = {
    "every10_balanced": "mnist_rotation_every10_balanced.pt",
    "every10":          "mnist_rotation_every10.pt",
    "every45_balanced": "mnist_rotation_every45_balanced.pt",
    "every45":          "mnist_rotation_every45.pt",
    "thesis_balanced":  "mnist_rotation_thesis_balanced.pt",
    "thesis":           "mnist_rotation_thesis.pt",
}


class RotationPtDataset(Dataset):
    def __init__(self, pt_path):
        print(f"Loading dataset: {pt_path}")
        data = torch.load(pt_path, map_location="cpu")
        self.images_flat = data["images_flat"]  # [N, 784]
        self.angle_vec   = data["angle_vec"]    # [N, 2]
        print(f"  {len(self.images_flat):,} samples loaded.")

    def __len__(self):
        return len(self.images_flat)

    def __getitem__(self, idx):
        return self.images_flat[idx], self.angle_vec[idx]


class VanillaMNISTDataset(Dataset):
    def __init__(self, root):
        self.base = datasets.MNIST(
            root, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        theta_deg = float(torch.rand(1) * 360.0)
        theta_rad = math.radians(theta_deg)
        img_rotated = rotate(img, theta_deg)
        angle_vec = torch.tensor(
            [math.cos(theta_rad), math.sin(theta_rad)], dtype=torch.float32
        )
        return img_rotated.view(-1), angle_vec


if args.dataset == "vanilla":
    train_dataset = VanillaMNISTDataset(args.data_dir)
else:
    pt_path = os.path.join(args.data_dir, "mnist_rotation", DATASET_FILES[args.dataset])
    if not os.path.exists(pt_path):
        raise FileNotFoundError(
            f"Dataset file not found: {pt_path}\n"
            f"Files available: {os.listdir(os.path.dirname(pt_path))}"
        )
    train_dataset = RotationPtDataset(pt_path)

train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size,
    shuffle=True, num_workers=2, pin_memory=True
)
print(f"Dataset '{args.dataset}': {len(train_dataset):,} samples, {len(train_loader)} batches/epoch")

# ──────────────── Conditioning augmentation ──────────────────
def augment_conditioning(img_flat, noise_std=0.05, dropout_p=0.1):
    x    = img_flat + noise_std * torch.randn_like(img_flat)
    mask = (torch.rand_like(img_flat) > dropout_p).float()
    return x * mask

# ─────────────────── CNN image encoder ──────────────────────
custom_cond = nn.Sequential(
    nn.Unflatten(1, (1, 28, 28)),
    nn.Conv2d(1, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(), nn.Dropout2d(0.1),
    nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(), nn.Dropout2d(0.15),
    nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(), nn.MaxPool2d(2),
    nn.Dropout2d(0.2), nn.Flatten(), nn.Dropout(0.3),
    nn.Linear(64 * 7 * 7, 128), nn.SiLU(),
).to(device)

# ─────────────────────── Model ──────────────────────────────
model = ConsistencyModeliCT(
    nfeatures=2, condition_on=784, eps=0.002,
    nunits=128, depth=args.depth,
    cond_embed_type="linear",
    cond_embed_model=custom_cond,
    add_input_norm=True, add_output_norm=True,
    device=device,
)
model.c_huber = 0.00054 * math.sqrt(2)

# ────────────── Optional EMA target model ───────────────────
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
    ema_model = model

# ─────────────────────── Optimizer + Scheduler ──────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Total scheduler steps depends on whether we step per batch or per epoch
total_steps = args.nepochs * len(train_loader) if args.scheduler_per_batch else args.nepochs

if args.scheduler == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.eta_min)
elif args.scheduler == "cosine_restarts":
    T_0 = args.restart_period * len(train_loader) if args.scheduler_per_batch else args.restart_period
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, eta_min=args.eta_min)
elif args.scheduler == "constant":
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

wandb.watch(model, log="gradients", log_freq=500)

# ─────────────────────── Training Loop ──────────────────────
global_step = 0
for epoch in range(1, args.nepochs + 1):
    model.train()
    epoch_loss = 0.0
    n_batches  = 0

    N          = ict_discretization_schedule(epoch, args.nepochs, s0=args.ict_s0, s1=args.ict_s1)
    # Matches notebook: kerras_boundaries(10.0, 0.002, N, 7.0)
    boundaries = kerras_boundaries(args.kerras_sigma_max, 0.002, N, args.kerras_rho).to(device)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.nepochs}  N={N}", leave=False)
    for images_flat, angle_vecs in pbar:
        images_flat = images_flat.to(device)
        angle_vecs  = angle_vecs.to(device)
        B = angle_vecs.shape[0]

        # Conditioning: augment if requested, otherwise use image directly
        if args.cond_augment:
            cond = augment_conditioning(images_flat, args.cond_noise, args.pixel_dropout)
        else:
            cond = images_flat

        t_idx = model.ict_noise_sampling(boundaries, B, P_mean=-1.1, P_std=2.0, device=device)
        t0    = boundaries[t_idx]
        t1    = boundaries[t_idx + 1]
        z     = torch.randn_like(angle_vecs)

        with torch.no_grad():
            x1_target = ema_model(angle_vecs + z * t0, t0, cond=cond)

        x2_pred  = model(angle_vecs + z * t1, t1, cond=cond)
        lambda_t = 1.0 / (t1 - t0 + 1e-8)
        loss     = (lambda_t * smooth_huber_loss(x1_target, x2_pred, c=model.c_huber)).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Step scheduler per batch if requested (matches notebook behavior)
        if args.scheduler_per_batch:
            scheduler.step()

        if args.use_ema:
            update_ema(model, ema_model, args.ema_decay)

        epoch_loss  += loss.item()
        n_batches   += 1
        global_step += 1

        if global_step % args.log_every == 0:
            wandb.log({"train/loss_step":    loss.item(),
                       "train/global_step":  global_step,
                       "train/N_boundaries": N,
                       "train/lr":          optimizer.param_groups[0]['lr']},
                      step=global_step)
        pbar.set_postfix(loss=f"{loss.item():.5f}", N=N)

    # Step scheduler per epoch (default)
    if not args.scheduler_per_batch:
        scheduler.step()

    avg_loss = epoch_loss / n_batches
    wandb.log({"train/loss_epoch":    avg_loss,
               "train/epoch":         epoch,
               "train/lr":            optimizer.param_groups[0]['lr'],
               "train/N_curriculum":  N}, step=global_step)
    print(f"Epoch {epoch:4d} | avg_loss={avg_loss:.6f} | N={N} | lr={optimizer.param_groups[0]['lr']:.2e}")

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
        print(f"  -> Saved checkpoint: {path}")

wandb.finish()
print("Training complete.")