"""
Unconditional DDPM on Rotated MNIST
====================================
Trains an unconditional UNet diffusion model on MNIST images that have been
randomly rotated. Checkpoints and DDIM samples are saved every 25 epochs.

Usage:
    python train_uncond.py
    python train_uncond.py --resume 100   # resume from epoch 100 checkpoint
"""

import os
import argparse
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.datasets import MNIST
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RotatedMNISTDataset(torch.utils.data.Dataset):
    """MNIST (train + test) with a random rotation sampled per access."""

    def __init__(self):
        train_data = MNIST("./data", train=True,  download=True, transform=transforms.ToTensor())
        test_data  = MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
        self.data  = torch.utils.data.ConcatDataset([train_data, test_data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        rotation = np.random.uniform(0, 360)
        image    = TF.rotate(image.float(), rotation, fill=0)
        return image, label, torch.tensor(rotation, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class UnconditionalUnet(nn.Module):
    """Thin wrapper around HuggingFace UNet2DModel (no class conditioning)."""

    def __init__(self):
        super().__init__()
        self.model = UNet2DModel(
            sample_size=28,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(32, 64, 128),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            dropout=0.1
        )

    def forward(self, x, t):
        return self.model(x, t).sample


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize_samples(model, noise_scheduler, epoch, device, n=16, plots_dir='plots'):
    """Generate samples with DDIM and save a grid."""
    model.eval()
    with torch.no_grad():
        ddim = DDIMScheduler.from_config(noise_scheduler.config)
        ddim.set_timesteps(num_inference_steps=50)
        x = torch.randn(n, 1, 28, 28, device=device)
        for t in ddim.timesteps:
            x = ddim.step(model(x, t), t, x).prev_sample

    grid = torchvision.utils.make_grid(x.cpu().clip(-1, 1), nrow=8)[0]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(grid, cmap="gray")
    ax.set_title(f"Generated Samples — Epoch {epoch} (DDIM, unconditional)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/samples_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved samples plot: {plots_dir}/samples_epoch_{epoch}.png")
    model.train()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(net, dataloader, noise_scheduler, n_epochs, lr, device,
          checkpoint_dir, resume_from_epoch, plots_dir='plots'):

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    loss_fn = nn.MSELoss()
    opt     = torch.optim.AdamW(net.parameters(), lr=lr)
    losses  = []
    start   = 0

    # ---- optional resume ----
    if resume_from_epoch is not None:
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{resume_from_epoch}.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            net.load_state_dict(ckpt["model_state_dict"])
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            losses = ckpt["losses"]
            start  = resume_from_epoch
            print(f"Resumed from epoch {start}")
        else:
            print(f"Checkpoint not found: {ckpt_path}")

    # ---- training loop ----
    net.train()
    for epoch in range(start, n_epochs):
        epoch_loss, n_batches = 0.0, 0

        for x, _, __ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            x         = x.to(device) * 2 - 1          # map [0,1] → [-1,1]
            noise     = torch.randn_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],), device=device).long()
            noisy_x   = noise_scheduler.add_noise(x, noise, timesteps)

            pred = net(noisy_x, timesteps)
            loss = loss_fn(pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            epoch_loss += loss.item()
            n_batches  += 1

        avg = epoch_loss / n_batches
        recent_avg = sum(losses[-100:]) / min(100, len(losses))
        print(f"Epoch {epoch+1:4d} | avg loss {avg:.5f} | recent-100 avg {recent_avg:.5f}")

        # ---- checkpoint + visualize every 25 epochs ----
        if (epoch + 1) % 25 == 0:
            ckpt = {
                "epoch":                epoch,
                "model_state_dict":     net.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "losses":               losses,
                "avg_loss":             avg,
            }
            path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(ckpt, path)
            print(f"Checkpoint saved: {path}")
            visualize_samples(net, noise_scheduler, epoch + 1, device, plots_dir=plots_dir)

    return net, losses


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--resume",     type=int,   default=None,
                        help="Resume from this epoch number")
    parser.add_argument("--ckpt_dir",   type=str,   default="checkpoints")
    parser.add_argument("--plots_dir",  type=str,   default="plots")
    args = parser.parse_args()

    print(f"Device: {device}")

    dataset    = RotatedMNISTDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    net             = UnconditionalUnet().to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000,
                                    beta_schedule="squaredcos_cap_v2")

    print(f"Model parameters: {sum(p.numel() for p in net.parameters()):,}")

    net, losses = train(
        net               = net,
        dataloader        = dataloader,
        noise_scheduler   = noise_scheduler,
        n_epochs          = args.epochs,
        lr                = args.lr,
        device            = device,
        checkpoint_dir    = args.ckpt_dir,
        resume_from_epoch = args.resume,
        plots_dir         = args.plots_dir,
    )

    # Final loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig(f'{args.plots_dir}/loss_curve.png', dpi=150)
    plt.close()
    print(f"Loss curve saved: {args.plots_dir}/loss_curve.png")