"""
MNIST Unconditional UNet Training with WandB Logging
=====================================================
Usage:
    python train_mnist_unconditional.py --dataset every10_balanced   # default
    python train_mnist_unconditional.py --dataset every10
    python train_mnist_unconditional.py --dataset every45_balanced
    python train_mnist_unconditional.py --dataset every45
    python train_mnist_unconditional.py --dataset thesis_balanced
    python train_mnist_unconditional.py --dataset thesis
    python train_mnist_unconditional.py --dataset vanilla            # raw MNIST on-the-fly
    python train_mnist_unconditional.py --use_ema                    # with EMA
"""

import argparse
import os
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from diffusers import DDPMScheduler, UNet2DModel
from tqdm import tqdm

# ─────────────────────────── CLI ────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="every10_balanced",
                    choices=["every10_balanced", "every10",
                             "every45_balanced", "every45",
                             "thesis_balanced", "thesis",
                             "vanilla"],
                    help="Which dataset to use for training.")
parser.add_argument("--use_ema",       action="store_true")
parser.add_argument("--ema_decay",     type=float, default=0.9999)
parser.add_argument("--nepochs",       type=int,   default=100)
parser.add_argument("--batch_size",    type=int,   default=256)
parser.add_argument("--lr",            type=float, default=1e-3)
parser.add_argument("--weight_decay",  type=float, default=1e-4)
parser.add_argument("--run_name",      type=str,   default=None)
parser.add_argument("--save_dir",      type=str,   default="./checkpoints_uncond")
parser.add_argument("--data_dir",      type=str,   default="./data")
parser.add_argument("--wandb_project", type=str,   default="mnist-unconditional")
parser.add_argument("--log_every",     type=int,   default=50)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.save_dir, exist_ok=True)

# ─────────────────────────── WandB ──────────────────────────
run_name = args.run_name or f"uncond_unet_{'EMA' if args.use_ema else 'baseline'}_{args.dataset}"
wandb.init(
    project=args.wandb_project,
    name=run_name,
    config={
        "model":           "UNet2DModel",
        "architecture":    "channels=(32,64,128), attn_in_deep_blocks",
        "noise_schedule":  "squaredcos_cap_v2",
        "diffusion_steps": 1000,
        "dataset":         args.dataset,
        "nepochs":         args.nepochs,
        "batch_size":      args.batch_size,
        "lr":              args.lr,
        "weight_decay":    args.weight_decay,
        "use_ema":         args.use_ema,
        "ema_decay":       args.ema_decay if args.use_ema else None,
        "optimizer":       "AdamW",
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
    """
    Loads a pre-built .pt rotation dataset.
    Expects key: 'images' [N, 1, 28, 28] in [-1, 1].
    """
    def __init__(self, pt_path):
        print(f"Loading dataset: {pt_path}")
        data = torch.load(pt_path, map_location="cpu")
        self.images = data["images"]   # [N, 1, 28, 28]
        print(f"  {len(self.images):,} samples loaded.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


class VanillaMNISTDataset(Dataset):
    """Raw MNIST with uniform on-the-fly rotation."""
    def __init__(self, root):
        self.base = datasets.MNIST(
            root, train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomRotation(180),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return img


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

# ─────────────────────────── Model ──────────────────────────
model = UNet2DModel(
    sample_size=28,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(32, 64, 128),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    dropout=0.1,
).to(device)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",
)

# ─────────────────── Optional EMA shadow ────────────────────
if args.use_ema:
    ema_model = UNet2DModel(
        sample_size=28, in_channels=1, out_channels=1,
        layers_per_block=2, block_out_channels=(32, 64, 128),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
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
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss_fn   = nn.MSELoss()
wandb.watch(model, log="gradients", log_freq=500)

# ─────────────────────── Training Loop ──────────────────────
global_step = 0
for epoch in range(1, args.nepochs + 1):
    model.train()
    epoch_loss = 0.0
    n_batches  = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.nepochs}", leave=False)
    for images in pbar:
        images = images.to(device)   # [B, 1, 28, 28]
        B = images.shape[0]

        timesteps  = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                   (B,), device=device, dtype=torch.long)
        noise      = torch.randn_like(images)
        noisy      = noise_scheduler.add_noise(images, noise, timesteps)
        pred_noise = model(noisy, timesteps).sample
        loss       = loss_fn(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if args.use_ema:
            update_ema(model, ema_model, args.ema_decay)

        epoch_loss  += loss.item()
        n_batches   += 1
        global_step += 1

        if global_step % args.log_every == 0:
            wandb.log({"train/loss_step": loss.item(),
                       "train/global_step": global_step}, step=global_step)
        pbar.set_postfix(loss=f"{loss.item():.5f}")

    avg_loss = epoch_loss / n_batches
    wandb.log({"train/loss_epoch": avg_loss, "train/epoch": epoch}, step=global_step)
    print(f"Epoch {epoch:4d} | avg_loss={avg_loss:.6f}")

    if epoch % 25 == 0 or epoch == args.nepochs:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "args": vars(args),
        }
        if args.use_ema:
            ckpt["ema_state_dict"] = ema_model.state_dict()
        path = os.path.join(args.save_dir, f"{run_name}_epoch{epoch:04d}.pth")
        torch.save(ckpt, path)
        wandb.save(path)
        print(f"  -> Saved checkpoint: {path}")

wandb.finish()
print("Training complete.")