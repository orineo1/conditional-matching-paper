# improved_autoencoders_with_training.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from functools import partial
from sklearn.preprocessing import StandardScaler

# -------------------------
# Feature-Gating "Attention"
# -------------------------
class FeatureGating(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        hidden = max(hidden, dim * 2)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        gating = self.net(x)
        return x * (1.0 + gating)  # residual gated scaling

# -------------------------
# Self-Attention Block
# -------------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, hidden_gate=64, dropout=0.1):
        super().__init__()
        self.gating = FeatureGating(dim, hidden=hidden_gate)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        g = self.gating(self.norm1(x))
        x = x + self.dropout(g)
        f = self.ffn(self.norm2(x))
        x = x + f
        return x

# -------------------------
# Encoder
# -------------------------
class ImprovedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.1, use_attention=True, device=None):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_norm  = nn.LayerNorm(hidden_dims[0])
        self.dropout     = nn.Dropout(dropout)
        self.use_attention = use_attention

        self.hidden_layers = nn.ModuleList()
        self.hidden_norms  = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.hidden_norms.append(nn.LayerNorm(hidden_dims[i+1]))
            if use_attention and i >= max(0, len(hidden_dims)//2 - 1):
                self.attention_blocks.append(SelfAttentionBlock(hidden_dims[i+1], hidden_gate=max(32, hidden_dims[i+1]//2), dropout=dropout))
            else:
                self.attention_blocks.append(None)

        self.mu_head     = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], latent_dim)
        nn.init.constant_(self.logvar_head.weight, 0.0)
        nn.init.constant_(self.logvar_head.bias, -4.0)

        self._last_mu = None
        self._last_logvar = None

        if device is None: device = "cpu"
        self.to(device)

    def _backbone(self, x):
        h = F.gelu(self.input_norm(self.input_layer(x)))
        h = self.dropout(h)
        for layer, norm, attn_block in zip(self.hidden_layers, self.hidden_norms, self.attention_blocks):
            residual = h
            h = F.gelu(norm(layer(h)))
            h = self.dropout(h)
            if attn_block is not None:
                h = attn_block(h)
            if residual.shape[-1] == h.shape[-1]:
                h = h + residual
        return h

    def forward(self, x):
        h = self._backbone(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, min=-8.0, max=4.0)
        self._last_mu = mu
        self._last_logvar = logvar
        return mu

    def get_last_stats(self):
        return self._last_mu, self._last_logvar

# -------------------------
# Decoder
# -------------------------
class ImprovedDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, dropout=0.1, use_attention=True, device=None):
        super().__init__()
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.GELU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
        )
        self.hidden_layers = nn.ModuleList()
        self.hidden_norms  = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()

        rev_dims = list(reversed(hidden_dims))
        for i in range(len(rev_dims) - 1):
            self.hidden_layers.append(nn.Linear(rev_dims[i], rev_dims[i+1]))
            self.hidden_norms.append(nn.LayerNorm(rev_dims[i+1]))
            if use_attention and i >= max(0, len(rev_dims)//2 - 1):
                self.attention_blocks.append(SelfAttentionBlock(rev_dims[i+1], hidden_gate=max(32, rev_dims[i+1]//2), dropout=dropout))
            else:
                self.attention_blocks.append(None)

        self.output_layer = nn.Linear(hidden_dims[0], output_dim)
        self.dropout = nn.Dropout(dropout)

        if device is None: device = "cpu"
        self.to(device)

    def forward(self, z):
        h = torch.relu(self.from_latent(z))
        h = self.dropout(h)
        for layer, norm, attn_block in zip(self.hidden_layers, self.hidden_norms, self.attention_blocks):
            residual = h
            h = torch.relu(norm(layer(h)))
            h = self.dropout(h)
            if attn_block is not None:
                h = attn_block(h)
            if residual.shape[-1] == h.shape[-1]:
                h = h + residual
        return self.output_layer(h)

# -------------------------
# KL Loss & Beta Schedule
# -------------------------
def ldm_kl_loss(encoder: ImprovedEncoder, reduction="mean"):
    mu, logvar = encoder.get_last_stats()
    if mu is None or logvar is None:
        raise RuntimeError("Call encoder(x) before ldm_kl_loss(encoder).")
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    else:
        return kl

def warmup_factor(epoch, warmup_epochs):
    return float(min(1.0, epoch / float(max(1, warmup_epochs))))

def beta_schedule(epoch, warmup_epochs, final_beta=1e-4):
    return warmup_factor(epoch, warmup_epochs) * final_beta

# -------------------------
# Training function
# -------------------------
def train_autoencoders(
    data_generator,
    input_dim1,
    input_dim2,
    latent_dim1=2,
    latent_dim2=1,
    hidden_dims=[32, 64, 128, 64, 32],
    dropout=0.08,
    use_attention=True,
    nepochs=5000,
    batch_size=512,
    warmup_epochs=1500,
    final_beta=1e-3,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder1 = ImprovedEncoder(input_dim1, hidden_dims, latent_dim1, dropout, use_attention, device)
    decoder1 = ImprovedDecoder(latent_dim1, hidden_dims, input_dim1, dropout, use_attention, device)
    encoder2 = ImprovedEncoder(input_dim2, hidden_dims, latent_dim2, dropout, use_attention, device)
    decoder2 = ImprovedDecoder(latent_dim2, hidden_dims, input_dim2, dropout, use_attention, device)

    optimizer1 = torch.optim.AdamW(list(encoder1.parameters()) + list(decoder1.parameters()), lr=2e-3, weight_decay=1e-5)
    optimizer2 = torch.optim.AdamW(list(encoder2.parameters()) + list(decoder2.parameters()), lr=2e-3, weight_decay=1e-5)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer1, T_0=2000, T_mult=1)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer2, T_0=2000, T_mult=1)

    reconstruction_loss_fn = nn.SmoothL1Loss(reduction="mean")

    sample_data = data_generator(10000, device=device).float().cpu().numpy()
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    scaler1.fit(sample_data[:, :input_dim1])
    scaler2.fit(sample_data[:, input_dim1:])

    val_data = data_generator(2000, device=device).float()
    val_x1 = torch.tensor(scaler1.transform(val_data[:, :input_dim1].cpu().numpy()), dtype=torch.float32, device=device)
    val_x2 = torch.tensor(scaler2.transform(val_data[:, input_dim1:].cpu().numpy()), dtype=torch.float32, device=device)

    print("Starting training on device:", device)
    progress = tqdm(range(nepochs), desc="Training Encoders")

    for epoch in progress:
        Xbatch = data_generator(batch_size, device=device).float()
        X_subset1 = torch.tensor(scaler1.transform(Xbatch[:, :input_dim1].cpu().numpy()), device=device, dtype=torch.float32)
        X_subset2 = torch.tensor(scaler2.transform(Xbatch[:, input_dim1:].cpu().numpy()), device=device, dtype=torch.float32)

        z1 = encoder1(X_subset1)
        recon1 = decoder1(z1)
        z2 = encoder2(X_subset2)
        recon2 = decoder2(z2)

        rec1 = reconstruction_loss_fn(recon1, X_subset1)
        rec2 = reconstruction_loss_fn(recon2, X_subset2)
        kl1 = ldm_kl_loss(encoder1)
        kl2 = ldm_kl_loss(encoder2)
        beta_kl = beta_schedule(epoch, warmup_epochs, final_beta=final_beta)

        loss1 = rec1  # + beta_kl * kl1
        loss2 = rec2  # + beta_kl * kl2
        total_loss = loss1 + loss2

        optimizer1.zero_grad(set_to_none=True)
        optimizer2.zero_grad(set_to_none=True)
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(list(encoder1.parameters()) + list(decoder1.parameters()), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(list(encoder2.parameters()) + list(decoder2.parameters()), max_norm=5.0)

        optimizer1.step()
        optimizer2.step()
        scheduler1.step()
        scheduler2.step()

        if (epoch % 250 == 0) or (epoch < 100 and epoch % 10 == 0):
            with torch.no_grad():
                recon1_val = decoder1(encoder1(val_x1))
                recon2_val = decoder2(encoder2(val_x2))
                val_rec1 = reconstruction_loss_fn(recon1_val, val_x1).item()
                val_rec2 = reconstruction_loss_fn(recon2_val, val_x2).item()
                val_kl1 = ldm_kl_loss(encoder1).item()
                val_kl2 = ldm_kl_loss(encoder2).item()

            progress.set_postfix({
                "total_loss": total_loss.item(),
                "rec1": rec1.item(),
                "rec2": rec2.item(),
                "val_rec1": val_rec1,
                "val_rec2": val_rec2,
                "kl1": val_kl1,
                "kl2": val_kl2,
                "beta_kl": beta_kl
            })

        if epoch % 1000 == 0:
            tqdm.write(
                f"Epoch {epoch}: total={total_loss.item():.6f} | "
                f"rec1={rec1.item():.6f} kl1={kl1.item():.6f} | "
                f"rec2={rec2.item():.6f} kl2={kl2.item():.6f} | "
                f"beta_kl={beta_kl:.6g}"
            )

    print("Training finished.")
    encoder1.eval(); decoder1.eval()
    encoder2.eval(); decoder2.eval()

    return encoder1, decoder1, encoder2, decoder2, scaler1, scaler2


def diffusion_training_data_generator(batch_size, data_generator, device, condition_on, scaler1, scaler2, encoder1,
                                      encoder2, uncond_and_cond=True):
    """
    Generate encoded training data for diffusion models.

    Args:
        batch_size (int): Number of samples to generate
        data_generator (callable): Function that generates raw data
        device (torch.device): Device for tensor allocation
        condition_on (int): Index to split data into subsets
        scaler1: Preprocessing scaler for first subset
        scaler2: Preprocessing scaler for second subset
        encoder1: Neural network encoder for first subset
        encoder2: Neural network encoder for second subset
        uncond_and_cond (bool): If True, returns concatenated encodings

    Returns:
        torch.Tensor: Encoded latent representations
    """
    # Generate raw data
    Xbatch = data_generator(batch_size, device=device).float()

    # Split and encode
    X_subset1 = scaler1.transform(Xbatch[:, :condition_on].cpu().numpy())
    if uncond_and_cond:
        X_subset2 = scaler2.transform(Xbatch[:, condition_on:].cpu().numpy())

    # Encode to latent space
    with torch.no_grad():
        X_subset1_tensor = torch.tensor(X_subset1, dtype=torch.float32).to(device)
        encoded1 = encoder1(X_subset1_tensor)
        if uncond_and_cond:
            X_subset2_tensor = torch.tensor(X_subset2, dtype=torch.float32).to(device)
            encoded2 = encoder2(X_subset2_tensor)

    if uncond_and_cond:
        return torch.cat([encoded1, encoded2], dim=1)  # Concatenate latents
    return encoded1  # Return just encoded1, no need to cat with single tensor

