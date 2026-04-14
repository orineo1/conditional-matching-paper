"""
cond_model.py
========
Circular Angle Consistency Model (iCT) conditioned on MNIST images.

Exports:
    - CircularAngleConsistencyModel
    - angles_to_circular()
    - circular_to_angles()
"""

import os
import math
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Angle utilities
# ---------------------------------------------------------------------------

def angles_to_circular(angles_deg):
    """Convert angles (degrees) → (cos, sin) circular representation."""
    rad = torch.deg2rad(angles_deg.float())
    return torch.stack([torch.cos(rad), torch.sin(rad)], dim=-1)


def circular_to_angles(circular):
    """Convert (cos, sin) circular representation → angles in [0°, 360°)."""
    return torch.rad2deg(torch.atan2(circular[..., 1], circular[..., 0])) % 360


# ---------------------------------------------------------------------------
# iCT helpers
# ---------------------------------------------------------------------------

def smooth_huber_loss(x, y, c=0.00054):
    """iCT smooth Huber loss: √(‖x−y‖₂² + c²) − c"""
    return torch.sqrt(torch.sum((x - y) ** 2, dim=-1, keepdim=True) + c**2) - c


def kerras_boundaries(sigma_min, sigma_max, N, rho, device):
    steps = torch.arange(N + 1, dtype=torch.float32, device=device)
    return (
        sigma_max ** (-1 / rho)
        + steps / N * (sigma_min ** (-1 / rho) - sigma_max ** (-1 / rho))
    ) ** (-rho)


def ict_discretization_schedule(epoch, total_epochs, s0=10, s1=1280):
    """iCT curriculum: N(k) = min(s₀ · 2^⌊k/K'⌋, s₁) + 1"""
    K_prime = math.floor(math.log2(s1 / s0))
    if K_prime <= 0:
        return s0 + 1
    exponent = math.floor(epoch / total_epochs * K_prime)
    return int(min(s0 * (2 ** exponent), s1)) + 1


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=1)


class GenericNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t_emb=None):
        if t_emb is not None and t_emb.dim() == 2 and x.dim() == 2:
            x = x + t_emb
        return self.net(x)


class ConditionAugmentation:
    """Lightweight augmentation applied to flattened condition images during training."""

    def __init__(self, noise_level=0.05, dropout_prob=0.1):
        self.noise_level  = noise_level
        self.dropout_prob = dropout_prob

    def __call__(self, condition_imgs):
        if not torch.is_grad_enabled():
            return condition_imgs
        imgs = condition_imgs.clone()
        if self.noise_level > 0:
            imgs = imgs + torch.randn_like(imgs) * self.noise_level
        if self.dropout_prob > 0:
            mask = torch.bernoulli(torch.ones_like(imgs) * (1 - self.dropout_prob))
            imgs = imgs * mask
        return imgs


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CircularAngleConsistencyModel(nn.Module):
    """
    Improved Consistency Training (iCT) model that predicts the rotation angle
    of an MNIST image as a circular (cos, sin) vector, conditioned on the image.
    """

    def __init__(self, nfeatures=2, img_features=784, eps=0.002,
                 nunits=128, depth=6, device=None):
        super().__init__()
        self.eps          = eps
        self.nfeatures    = nfeatures
        self.img_features = img_features
        self.nunits       = nunits
        self.device       = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.betas        = None

        # Image conditioning encoder
        self.cond_embed = nn.Sequential(
            nn.Unflatten(1, (1, 28, 28)),
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(), nn.Dropout2d(0.1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.MaxPool2d(2),                           # 14×14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(), nn.Dropout2d(0.15),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),        # 7×7
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 7 * 7, nunits),
            nn.SiLU(),
        )

        self.input_layer = nn.Linear(nfeatures, nunits)
        self.input_norm  = nn.LayerNorm(nunits)
        self.output_norm = nn.LayerNorm(nunits)
        self.time_embed  = TimeEmbedding(nunits)
        self.net         = GenericNN(nunits, [nunits] * depth, nunits)
        self.out         = nn.Linear(nunits, nfeatures)
        self.c_huber     = 0.00054 * math.sqrt(nfeatures)

        self.to(self.device)

    # ---- forward -----------------------------------------------------------

    def forward(self, x, t, cond=None):
        x_ori = x
        x     = self.input_norm(self.input_layer(x))

        if cond is not None:
            if cond.dim() == 4:
                cond = cond.view(cond.size(0), -1)
            elif cond.dim() == 3:
                cond = cond.view(cond.size(0), -1)
            x = x + self.cond_embed(cond)

        if isinstance(t, (float, int)):
            t = torch.tensor([t] * x.shape[0], dtype=torch.float32, device=x.device).unsqueeze(1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        elif t.dim() == 3:
            t = t.squeeze(-1)

        x = x + self.time_embed(t.squeeze(-1))
        x = self.output_norm(self.net(x))
        x = self.out(x)

        t_w    = t - self.eps
        c_skip = 0.25 / (t_w.pow(2) + 0.25)
        c_out  = 0.25 * t_w / ((t_w + self.eps).pow(2) + 0.25).sqrt()
        result = c_skip * x_ori + c_out * x
        return result / (torch.norm(result, dim=1, keepdim=True) + 1e-8)

    # ---- loss --------------------------------------------------------------

    def loss(self, x, z, t0, t1, ema_model, cond=None):
        with torch.no_grad():
            x1 = ema_model(x + z * t0, t0, cond=cond)
        x2       = self(x + z * t1, t1, cond=cond)
        lambda_t = 1.0 / (t1 - t0 + 1e-8)
        return (lambda_t * smooth_huber_loss(x1, x2, c=self.c_huber)).mean()

    # ---- noise sampling ----------------------------------------------------

    def ict_noise_sampling(self, boundaries, batch_size, P_mean=-1.1, P_std=2.0, device='cuda'):
        N          = len(boundaries) - 1
        log_s      = torch.log(boundaries[:-1])
        log_s_next = torch.log(boundaries[1:])
        sqrt_2_std = math.sqrt(2) * P_std
        probs      = (torch.erf((log_s_next - P_mean) / sqrt_2_std)
                    - torch.erf((log_s      - P_mean) / sqrt_2_std))
        probs      = torch.clamp(probs, min=1e-8)
        probs      = probs / probs.sum()
        return torch.multinomial(probs, batch_size, replacement=True).unsqueeze(1).to(device)

    # ---- sampling ----------------------------------------------------------

    def sample(self, nsamples=250, condition_x=None,
               ts: List[float] = [150.0, 50.0, 20.0, 10.0, 5.0, 1.], device=None):
        device = self.device if device is None else device
        if condition_x is not None:
            condition_x = condition_x.to(device)
            if condition_x.dim() > 2:
                condition_x = condition_x.view(condition_x.size(0), -1)

        x = torch.randn(nsamples, self.nfeatures, device=device) * ts[0]
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)

        for t in ts[1:]:
            z = torch.randn_like(x)
            x = x + math.sqrt(t ** 2 - self.eps ** 2) * z
            x = self(x, t, cond=condition_x)

        return x, None, None

    # ---- visualization -----------------------------------------------------

    def generate_sample_predictions(self, dataloader, device, epoch,
                                    num_samples=10, samples_per_condition=250,
                                    plots_dir='plots'):
        print(f"\n--- Epoch {epoch}: conditional angle histograms ---")
        self.eval()
        for batch_data in dataloader:
            if len(batch_data) == 2:
                images, true_angles = batch_data
                labels = None
            else:
                images, labels, true_angles = batch_data
            images      = images[:num_samples].to(device)
            true_angles = true_angles[:num_samples]
            break

        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        for i in range(5):
            true_angle = true_angles[i].item()
            axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
            title = f'Image {i+1}\nAngle: {true_angle:.1f}°'
            if labels is not None:
                title += f', Label: {labels[i].item()}'
            axes[0, i].set_title(title)
            axes[0, i].axis('off')

            circular_pred, _, _ = self.sample(
                nsamples=samples_per_condition,
                condition_x=images[i:i+1],
                device=device,
            )
            angles_np = circular_to_angles(circular_pred).detach().cpu().numpy()
            axes[1, i].hist(angles_np, bins=30, alpha=0.7, color='skyblue',
                            edgecolor='black', range=(0, 360))
            axes[1, i].axvline(true_angle, color='red', linestyle='--', linewidth=2,
                               label=f'True: {true_angle:.1f}°')
            axes[1, i].set_title('Predicted Angles')
            axes[1, i].set_xlabel('Angles (°)')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].set_xlim(0, 360)
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].legend()

        plt.tight_layout()
        plt.suptitle(f'Images (top) vs Angle Predictions (bottom) — Epoch {epoch}',
                     y=1.02, fontsize=16)
        plt.tight_layout()
        plt.suptitle(f'Images (top) vs Angle Predictions (bottom) — Epoch {epoch}',
                     y=1.02, fontsize=16)
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(f'{plots_dir}/predictions_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved predictions plot: {plots_dir}/predictions_epoch_{epoch}.png")
        self.train()
        self.train()

    # ---- training loop -----------------------------------------------------

    def train_model(self, dataloader, nepochs=100, device='cpu',
                    use_improved_training=True, save_dir='checkpoints',
                    checkpoint_path=None, start_epoch=None,plots_dir='plots'):

        os.makedirs(save_dir, exist_ok=True)
        self.to(device)
        aug = ConditionAugmentation(noise_level=0.05, dropout_prob=0.1)

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=nepochs, eta_min=1e-7
        )

        ema_model = self if use_improved_training else self._clone_to(device)

        losses          = []
        start_epoch_num = 1

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            self.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch_num = ckpt['epoch'] + 1
            losses          = ckpt.get('losses_history', [])
            print(f"Resuming from epoch {start_epoch_num}, loss={ckpt['loss']:.6f}")

        if start_epoch is not None:
            start_epoch_num = start_epoch

        pbar = tqdm(range(start_epoch_num, nepochs + 1))
        for epoch in pbar:
            N          = ict_discretization_schedule(epoch, nepochs, s0=10, s1=1280)
            boundaries = kerras_boundaries(10.0, 0.002, N, 7.0, device)
            self.betas = boundaries
            loss_ema   = None
            mu         = 0.0

            for batch_data in dataloader:
                if len(batch_data) == 2:
                    images, angles = batch_data
                elif len(batch_data) == 3:
                    images, _, angles = batch_data
                else:
                    images, _, angles, _ = batch_data

                images = images.to(device)
                angles = angles.to(device)
                x      = angles_to_circular(angles)
                cond_x = aug(images.view(images.size(0), -1))
                z      = torch.randn_like(x)

                t_idx = self.ict_noise_sampling(boundaries, x.shape[0],
                                                P_mean=-1.1, P_std=2.0, device=device)
                t0 = boundaries[t_idx]
                t1 = boundaries[t_idx + 1]

                optimizer.zero_grad()

                if use_improved_training:
                    with torch.no_grad():
                        x1_target = ema_model(x + z * t0, t0, cond=cond_x)
                    x2_pred  = self(x + z * t1, t1, cond=cond_x)
                    lambda_t = 1.0 / (t1 - t0 + 1e-8)
                    loss     = (lambda_t * smooth_huber_loss(x1_target, x2_pred,
                                                              c=self.c_huber)).mean()
                else:
                    loss = self.loss(x, z, t0, t1, ema_model=ema_model, cond=cond_x)

                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())

                loss_ema = loss.item() if loss_ema is None \
                    else 0.9 * loss_ema + 0.1 * loss.item()

                if not use_improved_training:
                    with torch.no_grad():
                        mu = math.exp(2 * math.log(0.95) / N)
                        for p, ema_p in zip(self.parameters(), ema_model.parameters()):
                            ema_p.mul_(mu).add_(p, alpha=1 - mu)

            pbar.set_description(f"loss: {loss_ema:.6f}, mu: {mu:.4f}, N: {N}")

            if epoch % 25 == 0:
                print(f"\nEpoch {epoch}/{nepochs}")
                self.generate_sample_predictions(dataloader, device, epoch, plots_dir=plots_dir)
                ckpt = {
                    'epoch':                epoch,
                    'model_state_dict':     self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss':                 loss_ema,
                    'losses_history':       losses,
                    'nfeatures':            self.nfeatures,
                    'img_features':         self.img_features,
                }
                ckpt_path = os.path.join(save_dir, f'circular_ict_epoch_{epoch}.pt')
                torch.save(ckpt, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

        return losses

    # ---- helpers -----------------------------------------------------------

    def _clone_to(self, device):
        clone = CircularAngleConsistencyModel(
            nfeatures=self.nfeatures, img_features=self.img_features, nunits=self.nunits
        ).to(device)
        clone.load_state_dict(self.state_dict())
        return clone

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device=None):
        ckpt  = torch.load(checkpoint_path, map_location=device)
        model = cls(nfeatures=ckpt['nfeatures'], img_features=ckpt['img_features'], device=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded {checkpoint_path} — epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f}")
        return model, ckpt
