import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from NN_utils import GenericNN, TimeEmbedding


# ── Simple fallback implementations (used if NN_utils not on path) ────────────

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=1)


class GenericNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t_emb=None):
        if t_emb is not None and t_emb.dim() == 2 and x.dim() == 2:
            x = x + t_emb
        return self.net(x)


# ── Karras boundaries & helpers ───────────────────────────────────────────────

def kerras_boundaries(sigma_min, sigma_max, N, rho):
    steps = torch.arange(N + 1, dtype=torch.float32,
                         device='cuda' if torch.cuda.is_available() else 'cpu')
    return (sigma_max ** (-1 / rho) +
            steps / N * (sigma_min ** (-1 / rho) - sigma_max ** (-1 / rho))) ** (-rho)


def smooth_huber_loss(x, y, c=0.00054):
    l2_sq = torch.sum((x - y) ** 2, dim=-1, keepdim=True)
    return torch.sqrt(l2_sq + c ** 2) - c


def ict_discretization_schedule(epoch, total_epochs, s0=10, s1=1280):
    K_prime = math.floor(math.log2(s1 / s0))
    if K_prime <= 0:
        return s0 + 1
    exponent = math.floor((epoch / total_epochs) * K_prime)
    return int(min(s0 * (2 ** exponent), s1) + 1)


# ── ConsistencyModel (basic CT) ───────────────────────────────────────────────

class ConsistencyModel(nn.Module):
    def __init__(self, nfeatures: int, condition_on: int = 0, eps: float = 0.002,
                 nunits: int = 128, depth: int = 6, device=None):
        super().__init__()
        self.eps        = eps
        self.nfeatures  = nfeatures
        self.condition_on = condition_on
        self.nunits     = nunits
        self.device     = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sigma_data = 0.5
        self.sigma_min  = 0.002
        self.sigma_max  = 80.0
        self.rho        = 7.0
        self.betas      = None

        if condition_on > 0:
            self.cond_embed = nn.Linear(condition_on, nunits)

        self.input_layer = nn.Linear(nfeatures, nunits)
        self.time_embed  = TimeEmbedding(nunits)
        self.out         = nn.Linear(nunits, nfeatures)
        self.net = GenericNN(input_dim=nunits, hidden_dims=[nunits] * depth, output_dim=nunits)
        self.to(self.device)

    def forward(self, x, t, cond=None):
        x_ori = x
        if isinstance(t, (float, int)):
            t = torch.tensor([t] * x.shape[0], dtype=torch.float32, device=x.device).unsqueeze(1)

        c_skip = self.sigma_data ** 2 / (t ** 2 + self.sigma_data ** 2)
        c_out  = self.sigma_data * t / torch.sqrt(t ** 2 + self.sigma_data ** 2)
        c_in   = 1 / torch.sqrt(t ** 2 + self.sigma_data ** 2)

        x = self.input_layer(c_in * x_ori)
        if self.condition_on > 0 and cond is not None:
            x = x + self.cond_embed(cond)
        x = self.net(x, self.time_embed(t))
        x = self.out(x)
        return c_skip * x_ori + c_out * x

    def loss_ct(self, x, z, t0, t1, cond=None):
        return F.mse_loss(self(x + z * t1, t1, cond=cond),
                          self(x + z * t0, t0, cond=cond))

    def multistep_loss(self, x, z, boundaries, cond=None):
        losses = []
        N = len(boundaries) - 1
        for _ in range(min(3, N - 1)):
            i  = torch.randint(0, N - 2, (1,)).item()
            t0 = boundaries[i]
            t1 = boundaries[i + 1]
            t2 = boundaries[i + 2] if i + 2 < len(boundaries) else t1
            losses.append(self.loss_ct(x, z, t0, t1, cond))
            if t2 != t1:
                losses.append(self.loss_ct(x, z, t0, t2, cond))
        return (torch.mean(torch.stack(losses)) if losses
                else self.loss_ct(x, z, boundaries[0], boundaries[1], cond))

    def adaptive_boundaries(self, epoch, max_epochs):
        max_N, min_N = 150, 4
        N = int(min_N + (max_N - min_N) * (epoch / max_epochs) ** 0.5)
        return kerras_boundaries(self.sigma_min, self.sigma_max, N, self.rho).to(self.device)

    def sample(self, nsamples=250, condition_x=None,
               ts: List[float] = [150.0, 50.0, 20.0, 10.0, 5.0, 1.], device=None):
        device = device or self.device
        if condition_x is not None:
            condition_x = condition_x.to(device)
        x = torch.randn(nsamples, self.nfeatures, device=device) * ts[0]
        for t in ts[1:]:
            z = torch.randn_like(x)
            x = x + math.sqrt(t ** 2 - self.eps ** 2) * z
            x = self(x, t, cond=condition_x)
        return x, None, None

    def train_model(self, X=None, data_generator=None, nepochs=100, batch_size=2048,
                    device="cpu", condition=None, model_diff=None, use_multistep=True,
                    wandb_run=None, wandb_prefix="cm", log_every=100):
        """
        Args:
            wandb_run:    active wandb run (or None)
            wandb_prefix: key prefix, e.g. "cm"
            log_every:    log every N epochs
        """
        assert X is not None or data_generator is not None
        self.to(device)
        optim = torch.optim.AdamW(self.parameters(), lr=1e-4)

        use_dataloader = X is not None
        if use_dataloader:
            dataloader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

        pbar = tqdm(range(1, nepochs + 1))
        for epoch in pbar:
            boundaries = self.adaptive_boundaries(epoch, nepochs)
            self.betas  = boundaries
            loss_ema    = None

            data_iter = dataloader if use_dataloader else [None]
            for batch_idx in data_iter:
                x = batch_idx[0] if use_dataloader else data_generator(batch_size, device=device)
                x = x.to(device)

                cond_x = None
                if condition is not None and condition > 0:
                    assert condition < x.shape[1]
                    cond_x = x[:, :condition]
                    x      = x[:, condition:]

                z   = torch.randn_like(x)
                N   = len(boundaries) - 1
                optim.zero_grad()
                if use_multistep and N > 3:
                    loss = self.multistep_loss(x, z, boundaries, cond=cond_x)
                else:
                    t_idx = torch.randint(0, N - 1, (x.shape[0], 1), device=device)
                    loss  = self.loss_ct(x, z, boundaries[t_idx], boundaries[t_idx + 1], cond=cond_x)

                loss.backward()
                optim.step()
                loss_ema = loss.item() if loss_ema is None else 0.9 * loss_ema + 0.1 * loss.item()

            pbar.set_description(f"loss: {loss_ema:.6f}, boundaries: {len(boundaries)}")

            if wandb_run is not None and (epoch % log_every == 0 or epoch == nepochs):
                wandb_run.log({f"{wandb_prefix}/loss": loss_ema,
                               f"{wandb_prefix}/epoch": epoch,
                               f"{wandb_prefix}/n_boundaries": len(boundaries)})


# ── ConsistencyModeliCT (improved CT) ────────────────────────────────────────

class ConsistencyModeliCT(nn.Module):
    def __init__(self, nfeatures: int, condition_on: int = 0, eps: float = 0.002,
                 nunits: int = 128, depth: int = 6,
                 cond_embed_type: str = 'linear', cond_embed_model=None,
                 add_input_norm: bool = False, add_output_norm: bool = False,
                 device=None):
        super().__init__()
        self.eps             = eps
        self.nfeatures       = nfeatures
        self.condition_on    = condition_on
        self.nunits          = nunits
        self.depth           = depth
        self.cond_embed_type = cond_embed_type
        self.add_input_norm  = add_input_norm
        self.add_output_norm = add_output_norm
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.betas  = None

        if condition_on > 0:
            if cond_embed_model is not None:
                self.cond_embed = cond_embed_model
            elif cond_embed_type == 'linear':
                self.cond_embed = nn.Linear(condition_on, nunits)
            elif cond_embed_type == 'mlp':
                self.cond_embed = nn.Sequential(
                    nn.Linear(condition_on, nunits), nn.ReLU(), nn.Linear(nunits, nunits))
            else:
                raise ValueError(f"Unknown cond_embed_type: {cond_embed_type}")

        self.input_layer = nn.Linear(nfeatures, nunits)
        if add_input_norm:
            self.input_norm = nn.LayerNorm(nunits)
        if add_output_norm:
            self.output_norm = nn.LayerNorm(nunits)

        self.time_embed = TimeEmbedding(nunits)
        self.out  = nn.Linear(nunits, nfeatures)
        self.net  = GenericNN(input_dim=nunits, hidden_dims=[nunits] * depth, output_dim=nunits)
        self.c_huber = 0.00054 * math.sqrt(nfeatures)
        self.to(self.device)

    def get_config(self):
        return {
            'nfeatures':       self.nfeatures,
            'condition_on':    self.condition_on,
            'eps':             self.eps,
            'nunits':          self.nunits,
            'depth':           self.depth,
            'cond_embed_type': self.cond_embed_type,
            'add_input_norm':  self.add_input_norm,
            'add_output_norm': self.add_output_norm,
        }

    def ict_noise_sampling(self, boundaries, batch_size, P_mean=-1.1, P_std=2.0, device='cuda'):
        N = len(boundaries) - 1
        log_s      = torch.log(boundaries[:-1])
        log_s_next = torch.log(boundaries[1:])
        sqrt_2_std = math.sqrt(2) * P_std
        probs = torch.erf((log_s_next - P_mean) / sqrt_2_std) - \
                torch.erf((log_s      - P_mean) / sqrt_2_std)
        probs = torch.clamp(probs, min=1e-8) / probs.clamp(min=1e-8).sum()
        return torch.multinomial(probs, batch_size, replacement=True).unsqueeze(1).to(device)

    def forward(self, x, t, cond=None):
        x_ori = x
        x = self.input_layer(x)
        if hasattr(self, 'input_norm'):
            x = self.input_norm(x)

        if self.condition_on > 0 and cond is not None:
            if   cond.dim() == 4: cond = cond.view(cond.size(0), -1)
            elif cond.dim() == 3: cond = cond.view(cond.size(0), -1)
            x = x + self.cond_embed(cond)

        if isinstance(t, (float, int)):
            t = torch.tensor([t] * x.shape[0], dtype=torch.float32, device=x.device).unsqueeze(1)
        elif t.dim() == 1: t = t.unsqueeze(1)
        elif t.dim() == 3: t = t.squeeze(-1)

        x = self.net(x, self.time_embed(t.squeeze(-1) if t.dim() > 1 else t))
        if hasattr(self, 'output_norm'):
            x = self.output_norm(x)
        x = self.out(x)

        t_shift   = t - self.eps
        c_skip_t  = 0.25 / (t_shift.pow(2) + 0.25)
        c_out_t   = 0.25 * t_shift / ((t_shift + self.eps).pow(2) + 0.25).sqrt()
        result    = c_skip_t * x_ori + c_out_t * x

        if self.add_output_norm and self.nfeatures == 2:
            result = result / (torch.norm(result, dim=1, keepdim=True) + 1e-8)
        return result

    def loss(self, x, z, t0, t1, ema_model, diffusion=None, cond=None, device="cpu"):
        with torch.no_grad():
            if diffusion is not None:
                x2, _ = diffusion.noise(x, t1)
                x1, _ = diffusion.sample_ddim_step(x2, t1, condition_x=cond, device=device, eta=0.0)
            else:
                x2 = x + z * t1
                x1 = x + z * t0
            x1 = ema_model(x1, t0, cond=cond)
        x2 = self(x2, t1, cond=cond)
        lambda_t    = 1.0 / (t1 - t0 + 1e-8)
        huber_loss  = smooth_huber_loss(x1, x2, c=self.c_huber)
        return (lambda_t * huber_loss).mean()

    def sample(self, nsamples=250, condition_x=None,
               ts: List[float] = [80, 40, 20, 10, 5, 2, 1, 0.5, 0.25, 0.125,
                                   0.062, 0.031, 0.015, 0.007, 0.002],
               device=None):
        device = device or self.device
        if condition_x is not None:
            condition_x = condition_x.to(device)
        x = torch.randn(nsamples, self.nfeatures, device=device) * ts[0]
        for t in ts[1:]:
            z = torch.randn_like(x)
            x = x + math.sqrt(t ** 2 - self.eps ** 2) * z
            x = self(x, t, cond=condition_x)
        return x, None, None

    def train_model(self, X=None, data_generator=None, nepochs=100, batch_size=2048,
                    device="cpu", condition=None, diffusion=None, model_diff=None,
                    use_improved_training=True,
                    wandb_run=None, wandb_prefix="ict", log_every=100):
        """
        Args:
            wandb_run:    active wandb run (or None)
            wandb_prefix: key prefix, e.g. "ict"
            log_every:    log every N epochs
        """
        assert X is not None or data_generator is not None
        self.to(device)
        optim = torch.optim.AdamW(self.parameters(), lr=1e-4)

        ema_model = self if use_improved_training else ConsistencyModeliCT(**self.get_config()).to(device)
        if not use_improved_training:
            ema_model.load_state_dict(self.state_dict())

        use_dataloader = X is not None
        if use_dataloader:
            dataloader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

        pbar = tqdm(range(1, nepochs + 1))
        mu   = 0.0

        for epoch in pbar:
            N          = ict_discretization_schedule(epoch, nepochs, s0=10, s1=1280)
            boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device)
            self.betas  = boundaries
            loss_ema    = None

            data_iter = dataloader if use_dataloader else [None]
            for batch_idx in data_iter:
                x = batch_idx[0] if use_dataloader else data_generator(batch_size, device=device)
                x = x.to(device)

                cond_x = None
                if condition is not None and condition > 0:
                    assert condition < x.shape[1]
                    cond_x = x[:, :condition]
                    x      = x[:, condition:]

                z     = torch.randn_like(x)
                t_idx = self.ict_noise_sampling(boundaries, x.shape[0],
                                                P_mean=-1.1, P_std=2.0, device=device)
                t0, t1 = boundaries[t_idx], boundaries[t_idx + 1]
                optim.zero_grad()

                if use_improved_training:
                    with torch.no_grad():
                        x2_t = x + z * t1
                        x1_t = x + z * t0
                        x1_t = ema_model(x1_t, t0, cond=cond_x)
                    x2_pred    = self(x2_t, t1, cond=cond_x)
                    lambda_t   = 1.0 / (t1 - t0 + 1e-8)
                    loss = (lambda_t * smooth_huber_loss(x1_t, x2_pred, c=self.c_huber)).mean()
                else:
                    loss = self.loss(x, z, t0, t1, ema_model=ema_model,
                                     diffusion=model_diff, cond=cond_x, device=device)

                loss.backward()
                optim.step()
                loss_ema = loss.item() if loss_ema is None else 0.9 * loss_ema + 0.1 * loss.item()

                if not use_improved_training:
                    mu = math.exp(2 * math.log(0.95) / N)
                    with torch.no_grad():
                        for p, ep in zip(self.parameters(), ema_model.parameters()):
                            ep.mul_(mu).add_(p, alpha=1 - mu)

            pbar.set_description(f"loss: {loss_ema:.6f}, mu: {mu:.4f}, N: {N}")

            # ── wandb logging ─────────────────────────────────────────────────
            if wandb_run is not None and (epoch % log_every == 0 or epoch == nepochs):
                wandb_run.log({f"{wandb_prefix}/loss":         loss_ema,
                               f"{wandb_prefix}/epoch":        epoch,
                               f"{wandb_prefix}/n_boundaries": N,
                               f"{wandb_prefix}/mu":           mu})

    def save_checkpoint(self, path, epoch, optimizer=None, scheduler=None, loss=None, **kwargs):
        ckpt = {'epoch': epoch, 'model_state_dict': self.state_dict(),
                'config': self.get_config(), 'loss': loss}
        if optimizer  is not None: ckpt['optimizer_state_dict']  = optimizer.state_dict()
        if scheduler  is not None: ckpt['scheduler_state_dict']  = scheduler.state_dict()
        ckpt.update(kwargs)
        torch.save(ckpt, path)
        print(f"Checkpoint saved: {path}")

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, cond_embed_model=None, device=None):
        ckpt       = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['model_state_dict']
        has_in     = 'input_norm.weight'  in state_dict
        has_out    = 'output_norm.weight' in state_dict
        config = ckpt.get('config', {
            'nfeatures':    ckpt.get('nfeatures', 2),
            'condition_on': ckpt.get('condition_on', ckpt.get('img_features', 784)),
            'eps':          ckpt.get('eps', 0.002),
            'nunits':       ckpt.get('nunits', 128),
            'depth':        ckpt.get('depth', 6),
            'cond_embed_type': ckpt.get('cond_embed_type', 'linear'),
            'add_input_norm':  has_in,
            'add_output_norm': has_out,
        })
        model = cls(**config, cond_embed_model=cond_embed_model, device=device)
        model.load_state_dict(state_dict)
        print(f"Loaded from: {checkpoint_path}  (epoch={ckpt.get('epoch','?')})")
        return model, ckpt