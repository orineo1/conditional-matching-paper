import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from NN_utils import GenericNN, TimeEmbedding

############### Consistency Models 2023 Song et al. ###############

# def kerras_boundaries(sigma_min, sigma_max, N, rho):
#     steps = torch.arange(N + 1, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
#     return (sigma_max ** (-1 / rho) + steps / N * (sigma_min ** (-1 / rho) - sigma_max ** (-1 / rho))) ** (-rho)


class ConsistencyModel(nn.Module):
    def __init__(self, nfeatures: int, condition_on: int = 0, eps: float = 0.002, nunits: int = 128, depth: int = 6,
                 device=None):
        super().__init__()
        self.eps = eps
        self.nfeatures = nfeatures
        self.condition_on = condition_on
        self.nunits = nunits
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # #### Changed - from Improved Consistency Training paper ###
        # Added sigma_data parameter for better parameterization
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80.0
        self.rho = 7.0
        ### End of changed ####

        self.betas = None

        if condition_on > 0:
            self.cond_embed = nn.Linear(condition_on, nunits)

        self.input_layer = nn.Linear(nfeatures, nunits)
        self.time_embed = TimeEmbedding(nunits)
        self.out = nn.Linear(nunits, nfeatures)
        self.net = GenericNN(
            input_dim=nunits,
            hidden_dims=[nunits] * depth,
            output_dim=nunits,
        )

        self.to(self.device)

    def forward(self, x, t, cond=None):
        x_ori = x

        # #### Changed - from Improved Consistency Training paper ###
        # Better parameterization with improved skip connections
        # Old version:
        # x = self.input_layer(x)
        # ...
        # c_skip_t = 0.25 / (t.pow(2) + 0.25)
        # c_out_t = 0.25 * t / ((t + self.eps).pow(2) + 0.25).sqrt()

        # New improved parameterization:
        if isinstance(t, (float, int)):
            t = torch.tensor([t] * x.shape[0], dtype=torch.float32, device=x.device).unsqueeze(1)

        c_skip = self.sigma_data ** 2 / (t ** 2 + self.sigma_data ** 2)
        c_out = self.sigma_data * t / torch.sqrt(t ** 2 + self.sigma_data ** 2)
        c_in = 1 / torch.sqrt(t ** 2 + self.sigma_data ** 2)

        x_scaled = c_in * x_ori
        x = self.input_layer(x_scaled)
        ### End of changed ####

        # Add conditioning if present
        if self.condition_on > 0 and cond is not None:
            cond_emb = self.cond_embed(cond)
            x = x + cond_emb

        time_emb = self.time_embed(t)
        x = self.net(x, time_emb)
        x = self.out(x)

        # #### Changed - from Improved Consistency Training paper ###
        # New skip connection using improved parameterization
        return c_skip * x_ori + c_out * x
        ### End of changed ####

    # #### Changed - from Consistency Models paper (Consistency Training) ###
    # Old distillation loss that required teacher model:
    # def loss(self, x, z, t0, t1, ema_model, diffusion=None, cond=None,device="cpu"):
    #     with torch.no_grad():
    #         if diffusion is not None:
    #             x2,_= diffusion.noise(x, t1)
    #             x1,_ = diffusion.sample_ddim_step( x2, t1, condition_x=cond, device=device, eta=0.0)
    #         else:
    #             x2 = x + z * t1
    #             x1 = x + z * t0
    #         x1 = ema_model(x1, t0, cond=cond)
    #     x2 = self(x2, t1, cond=cond)
    #     return F.mse_loss(x1, x2)

    # New Consistency Training loss (no teacher model needed):
    def loss_ct(self, x, z, t0, t1, cond=None):
        """Direct Consistency Training loss"""
        x_t1 = x + z * t1
        x_t0 = x + z * t0

        f_theta_t1 = self(x_t1, t1, cond=cond)
        f_theta_t0 = self(x_t0, t0, cond=cond)

        return F.mse_loss(f_theta_t1, f_theta_t0)

    def multistep_loss(self, x, z, boundaries, cond=None):
        """Multistep consistency training from later papers"""
        losses = []
        N = len(boundaries) - 1

        # Sample multiple timestep pairs
        for _ in range(min(3, N - 1)):  # Use up to 3 pairs per batch
            i = torch.randint(0, N - 2, (1,)).item()
            t0 = boundaries[i]
            t1 = boundaries[i + 1]
            t2 = boundaries[i + 2] if i + 2 < len(boundaries) else boundaries[i + 1]

            loss1 = self.loss_ct(x, z, t0, t1, cond)
            if t2 != t1:
                loss2 = self.loss_ct(x, z, t0, t2, cond)
                losses.extend([loss1, loss2])
            else:
                losses.append(loss1)

        return torch.mean(torch.stack(losses)) if losses else self.loss_ct(x, z, boundaries[0], boundaries[1], cond)

    ### End of changed ####

    def adaptive_boundaries(self, epoch, max_epochs):
        """Adaptive curriculum learning for time boundaries"""
        max_N = 150
        min_N = 4
        progress = epoch / max_epochs
        N = int(min_N + (max_N - min_N) * progress ** 0.5)
        return kerras_boundaries(self.sigma_min, self.sigma_max, N, self.rho).to(self.device)

    def sample(self, nsamples=250, condition_x=None, ts: List[float] = [150.0, 50.0, 20.0, 10.0, 5.0, 1.], device=None):
        device = self.device if device is None else device
        if condition_x is not None:
            condition_x = condition_x.to(device)

        x = torch.randn(nsamples, self.nfeatures, device=device) * ts[0]
        x = x.to(device)

        for t in ts[1:]:
            z = torch.randn_like(x)
            x = x + math.sqrt(t ** 2 - self.eps ** 2) * z
            x = self(x, t, cond=condition_x)

        return x, None, None

    def train_model(self, X: torch.Tensor = None, data_generator=None, nepochs: int = 100, batch_size: int = 2048,
                    device: str = "cpu", condition=None, model_diff=None, use_multistep=True):
        assert X is not None or data_generator is not None, "Need either X or data_generator"

        self.to(device)
        optim = torch.optim.AdamW(self.parameters(), lr=1e-4)

        # #### Changed - from Consistency Training paper ###
        # Remove EMA model requirement for CT (only needed for CD)
        # Old version used ema_model for distillation
        # New version uses direct consistency training
        ### End of changed ####

        if X is not None:
            dataset = TensorDataset(X)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            use_dataloader = True
        else:
            use_dataloader = False

        pbar = tqdm(range(1, nepochs + 1))
        for epoch in pbar:
            # #### Changed - Adaptive curriculum learning ###
            # Old: Fixed schedule
            # N = math.ceil(math.sqrt((epoch * (150 ** 2 - 4) / nepochs) + 4) - 1) + 1
            # boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device)

            # New: Adaptive boundaries with curriculum learning
            boundaries = self.adaptive_boundaries(epoch, nepochs)
            ### End of changed ####

            self.betas = boundaries
            loss_ema = None

            if use_dataloader:
                data_iter = dataloader
            else:
                data_iter = [None]  # Single iteration per epoch

            for batch_idx in data_iter:
                if use_dataloader:
                    (x,) = batch_idx
                else:
                    x = data_generator(batch_size, device=device)

                x = x.to(device)

                if condition is not None and condition > 0:
                    assert condition < x.shape[1], "Condition dimension must be less than input dimension"
                    cond_x = x[:, :condition]
                    x = x[:, condition:]
                else:
                    cond_x = None

                z = torch.randn_like(x)

                # #### Changed - New loss computation ###
                # Old: Used distillation loss with teacher model
                # optim.zero_grad()
                # loss = self.loss(x, z, t0, t1, ema_model=ema_model, diffusion=model_diff, cond=cond_x, device=device)

                # New: Direct consistency training with optional multistep
                optim.zero_grad()
                if use_multistep and len(boundaries) > 3:
                    loss = self.multistep_loss(x, z, boundaries, cond=cond_x)
                else:
                    N = len(boundaries) - 1
                    t_idx = torch.randint(0, N - 1, (x.shape[0], 1), device=device)
                    t0 = boundaries[t_idx]
                    t1 = boundaries[t_idx + 1]
                    loss = self.loss_ct(x, z, t0, t1, cond=cond_x)
                ### End of changed ####

                loss.backward()

                # #### Changed - Remove EMA updates for CT ###
                # Old: EMA model updates (only needed for CD)
                # with torch.no_grad():
                #     mu = math.exp(2 * math.log(0.95) / N)
                #     for p, ema_p in zip(self.parameters(), ema_model.parameters()):
                #         ema_p.mul_(mu).add_(p, alpha=1 - mu)

                # New: Simple gradient step (CT doesn't need EMA)
                ### End of changed ####

                optim.step()

                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.9 * loss_ema + 0.1 * loss.item()

            pbar.set_description(f"loss: {loss_ema:.6f}, boundaries: {len(boundaries)}")
        return

############### Improved Techniques for Training Consistency Models 2023 Song & Dharwil ###############



def kerras_boundaries(sigma_min, sigma_max, N, rho):
    steps = torch.arange(N + 1, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    return (sigma_max ** (-1 / rho) + steps / N * (sigma_min ** (-1 / rho) - sigma_max ** (-1 / rho))) ** (-rho)


def smooth_huber_loss(x, y, c=0.00054):
    """
    iCT's smooth Huber loss approximation: d(x, y) = √(||x - y||₂² + c²) - c

    This is equivalent to a smooth version of Huber loss:
    - For small errors: behaves like L2 loss (quadratic)
    - For large errors: behaves like L1 loss (linear)
    - Parameter c controls the transition point

    Args:
        x, y: predicted and target tensors
        c: smoothing parameter (paper uses c = 0.00054√d where d is data dimensionality)
    """
    l2_squared = torch.sum((x - y) ** 2, dim=-1, keepdim=True)
    return torch.sqrt(l2_squared + c**2) - c


def ict_discretization_schedule(epoch, total_epochs, s0=10, s1=1280):
    """
    iCT improved discretization curriculum: N(k) = min(s₀ * 2^⌊k/K'⌋, s₁) + 1
    where K' = ⌊log₂(s₁/s₀)⌋

    Args:
        epoch: current epoch (k)
        total_epochs: total training epochs (K)
        s0: minimum number of discretization steps (default: 10)
        s1: maximum number of discretization steps (default: 1280)
    """
    # Calculate K' = ⌊log₂(s₁/s₀)⌋
    K_prime = math.floor(math.log2(s1 / s0))

    # Prevent division by zero
    if K_prime <= 0:
        return s0 + 1

    # Calculate the discretization schedule
    # N(k) = min(s₀ * 2^⌊k * K' / total_epochs⌋, s₁) + 1
    progress_ratio = epoch / total_epochs
    exponent = math.floor(progress_ratio * K_prime)
    N = min(s0 * (2 ** exponent), s1) + 1

    return int(N)



# Simple implementations if NN_utils is not available
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
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return emb
class GenericNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t_emb=None):
        if t_emb is not None:
            # Ensure t_emb has the right shape for broadcasting
            if t_emb.dim() == 2 and x.dim() == 2:
                x = x + t_emb
        return self.net(x)
class ConsistencyModeliCT(nn.Module):
    # CHANGED: Made compatible with CircularAngleConsistencyModel checkpoints
    def __init__(self, nfeatures: int, condition_on: int = 0, eps: float = 0.002, nunits: int = 128, depth: int = 6,
                 cond_embed_type: str = 'linear', cond_embed_model=None,
                 add_input_norm: bool = False, add_output_norm: bool = False,  # NEW: compatibility flags
                 device=None):
        super().__init__()
        self.eps = eps
        self.nfeatures = nfeatures
        self.condition_on = condition_on
        self.nunits = nunits
        self.depth = depth
        self.cond_embed_type = cond_embed_type
        self.add_input_norm = add_input_norm  # NEW
        self.add_output_norm = add_output_norm  # NEW
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.betas = None

        # Conditioning
        if condition_on > 0:
            if cond_embed_model is not None:
                self.cond_embed = cond_embed_model
            elif cond_embed_type == 'linear':
                self.cond_embed = nn.Linear(condition_on, nunits)
            elif cond_embed_type == 'mlp':
                self.cond_embed = nn.Sequential(
                    nn.Linear(condition_on, nunits),
                    nn.ReLU(),
                    nn.Linear(nunits, nunits)
                )
            else:
                raise ValueError(f"Unknown cond_embed_type: {cond_embed_type}")

        self.input_layer = nn.Linear(nfeatures, nunits)

        # CHANGED: Add optional normalization layers for CircularAngleConsistencyModel compatibility
        if add_input_norm:
            self.input_norm = nn.LayerNorm(nunits)
        if add_output_norm:
            self.output_norm = nn.LayerNorm(nunits)
        # CHANGED: Optional norms

        self.time_embed = TimeEmbedding(nunits)
        self.out = nn.Linear(nunits, nfeatures)
        self.net = GenericNN(
            input_dim=nunits,
            hidden_dims=[nunits] * self.depth,
            output_dim=nunits,
        )

        self.c_huber = 0.00054 * math.sqrt(nfeatures)
        self.to(self.device)

    def get_config(self):
        """Return model configuration for saving"""
        return {
            'nfeatures': self.nfeatures,
            'condition_on': self.condition_on,
            'eps': self.eps,
            'nunits': self.nunits,
            'depth': self.depth,
            'cond_embed_type': self.cond_embed_type,
            'add_input_norm': self.add_input_norm,  # CHANGED: Save norm flags
            'add_output_norm': self.add_output_norm,  # CHANGED: Save norm flags
        }

    def ict_noise_sampling(self, boundaries, batch_size, P_mean=-1.1, P_std=2.0, device='cuda'):
        N = len(boundaries) - 1
        log_sigmas = torch.log(boundaries[:-1])
        log_sigmas_next = torch.log(boundaries[1:])
        sqrt_2_std = math.sqrt(2) * P_std
        erf_next = torch.erf((log_sigmas_next - P_mean) / sqrt_2_std)
        erf_curr = torch.erf((log_sigmas - P_mean) / sqrt_2_std)
        probs = erf_next - erf_curr
        probs = torch.clamp(probs, min=1e-8)
        probs = probs / probs.sum()
        t_idx = torch.multinomial(probs, batch_size, replacement=True).unsqueeze(1).to(device)
        return t_idx

    def forward(self, x, t, cond=None):
        x_ori = x
        x = self.input_layer(x)

        # CHANGED: Apply input norm if present (CircularAngleConsistencyModel has this)
        if hasattr(self, 'input_norm'):
            x = self.input_norm(x)
        # CHANGED: Optional input norm

        # Add conditioning if present
        if self.condition_on > 0 and cond is not None:
            # CHANGED: Flatten image if needed (BEFORE using cond_embed)
            if cond.dim() == 4:  # [B, 1, 28, 28]
                cond = cond.view(cond.size(0), -1)
            elif cond.dim() == 3:  # [B, 28, 28]
                cond = cond.view(cond.size(0), -1)
            # CHANGED: Flatten FIRST

            cond_emb = self.cond_embed(cond)
            x = x + cond_emb

        # Ensure t is (B, 1)
        if isinstance(t, (float, int)):
            t = torch.tensor([t] * x.shape[0], dtype=torch.float32, device=x.device).unsqueeze(1)
        elif t.dim() == 1:  # CHANGED: Handle 1D time
            t = t.unsqueeze(1)
        elif t.dim() == 3:  # CHANGED: Handle 3D time
            t = t.squeeze(-1)
        # CHANGED: More flexible time handling

        time_emb = self.time_embed(t.squeeze(-1) if t.dim() > 1 else t)  # CHANGED: Safe squeeze

        x = self.net(x, time_emb)

        # CHANGED: Apply output norm if present (CircularAngleConsistencyModel has this)
        if hasattr(self, 'output_norm'):
            x = self.output_norm(x)
        # CHANGED: Optional output norm

        x = self.out(x)

        # Output skip connection with consistency weights
        t = t - self.eps
        c_skip_t = 0.25 / (t.pow(2) + 0.25)
        c_out_t = 0.25 * t / ((t + self.eps).pow(2) + 0.25).sqrt()

        result = c_skip_t * x_ori + c_out_t * x

        # CHANGED: Optional unit circle normalization (CircularAngleConsistencyModel does this)
        if self.add_output_norm and self.nfeatures == 2:  # Only for circular angles
            result = result / (torch.norm(result, dim=1, keepdim=True) + 1e-8)
        # CHANGED: Optional circular normalization

        return result

    def loss(self, x, z, t0, t1, ema_model, diffusion=None, cond=None, device="cpu"):
        # Improved loss weighting from "Improved Consistency Models" paper (Song et al., 2023)
        # https://arxiv.org/abs/2310.14189 - Section 3.1
        # Uses λ(t) = 1/(t1-t0) weighting and smooth Huber loss metric

        with torch.no_grad():
            if diffusion is not None:
            # TODO in order to train using CT the noise also should by kerreas and the diff using score model (t should be not integers)
                x2,_= diffusion.noise(x, t1)
                x1,_ = diffusion.sample_ddim_step( x2, t1, condition_x=cond, device=device, eta=0.0)
            else:
                x2 = x + z * t1
                x1 = x + z * t0
            x1 = ema_model(x1, t0, cond=cond)

        x2 = self(x2, t1, cond=cond)

        # Apply improved loss weighting: λ(t) = 1/(t1-t0) from iCT paper
        lambda_t = 1.0 / (t1 - t0 + 1e-8)  # Add small epsilon for numerical stability

        # Use smooth Huber loss instead of MSE
        huber_loss = smooth_huber_loss(x1, x2, c=self.c_huber)
        weighted_loss = lambda_t * huber_loss

        return weighted_loss.mean()


    def sample(self, nsamples=250, condition_x=None,ts: List[float]=[80, 40, 20, 10, 5, 2, 1, 0.5, 0.25, 0.125, 0.062, 0.031, 0.015, 0.007, 0.002],device=None):
        device = self.device if device is None else device
        if condition_x is not None:
            condition_x = condition_x.to(device)

        x = torch.randn(nsamples, self.nfeatures, device=device) * ts[0]
        x = x.to(device)

        for t in ts[1:]:
            z = torch.randn_like(x)
            x = x + math.sqrt(t ** 2 - self.eps ** 2) * z
            x = self(x, t, cond=condition_x)

        return x,None,None

    def train_model(self, X: torch.Tensor = None, data_generator=None, nepochs: int = 100, batch_size: int = 2048,
                    device: str = "cpu", condition=None,diffusion=None,
                    model_diff=None, use_improved_training=True):
        assert X is not None or data_generator is not None, "Need either X or data_generator"

        self.to(device)

        optim = torch.optim.AdamW(self.parameters(), lr=1e-4)

        # Improved Consistency Models: Use μ=0 (no EMA updates)
        # From "Improved Consistency Models" (Song et al., 2023) - Section 3.2
        # https://arxiv.org/abs/2310.14189
        if use_improved_training:
            # Target model is just the current model (μ=0)
            ema_model = self
        else:
            # CHANGED: Use get_config for EMA model creation
            # OLD: ema_model = ConsistencyModelImproved(nfeatures=self.nfeatures, condition_on=self.condition_on, nunits=self.nunits)
            ema_model = ConsistencyModeliCT(**self.get_config())
            # CHANGED: Dynamic EMA creation
            ema_model.to(device)
            ema_model.load_state_dict(self.state_dict())

        if X is not None:
            dataset = TensorDataset(X)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            use_dataloader = True
        else:
            use_dataloader = False

        pbar = tqdm(range(1, nepochs + 1))
        for epoch in pbar:
            # iCT improved discretization curriculum
            # N(k) = min(s₀ * 2^⌊k/K'⌋, s₁) + 1, where K' = ⌊log₂(s₁/s₀)⌋
            N = ict_discretization_schedule(epoch, nepochs, s0=10, s1=1280)

            boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device)
            self.betas = boundaries
            loss_ema = None

            if use_dataloader:
                data_iter = dataloader
            else:
                data_iter = [None]  # Single iteration per epoch

            for batch_idx in data_iter:
                if use_dataloader:
                    (x,) = batch_idx
                else:
                    x = data_generator(batch_size, device=device)

                x = x.to(device)

                if condition is not None and condition > 0:
                    assert condition < x.shape[1], "Condition dimension must be less than input dimension"
                    cond_x = x[:, :condition]
                    x = x[:, condition:]
                else:
                    cond_x = None

                z = torch.randn_like(x)

                # iCT improved noise schedule sampling using erf distribution
                t_idx = self.ict_noise_sampling(boundaries, x.shape[0], P_mean=-1.1, P_std=2.0, device=device)
                t0 = boundaries[t_idx]
                t1 = boundaries[t_idx + 1]

                optim.zero_grad()

                # For improved training, we need to use stop_gradient on the target
                if use_improved_training:
                    with torch.no_grad():
                        if diffusion is not None:
                            x2_target, _ = diffusion.noise(x, t1)
                            x1_target, _ = diffusion.sample_ddim_step(x2_target, t1, condition_x=cond_x, device=device, eta=0.0)
                        else:
                            x2_target = x + z * t1
                            x1_target = x + z * t0
                        x1_target = ema_model(x1_target, t0, cond=cond_x)

                    x2_pred = self(x2_target, t1, cond=cond_x)

                    # Apply improved loss weighting and smooth Huber loss
                    lambda_t = 1.0 / (t1 - t0 + 1e-8)
                    huber_loss = smooth_huber_loss(x1_target, x2_pred, c=self.c_huber)
                    weighted_loss = lambda_t * huber_loss
                    loss = weighted_loss.mean()
                else:
                    loss = self.loss(x, z, t0, t1, ema_model=ema_model, diffusion=model_diff, cond=cond_x, device=device)

                loss.backward()
                optim.step()

                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.9 * loss_ema + 0.1 * loss.item()

                # Only update EMA if using original training
                if not use_improved_training:
                    with torch.no_grad():
                        mu = math.exp(2 * math.log(0.95) / N)
                        for p, ema_p in zip(self.parameters(), ema_model.parameters()):
                            ema_p.mul_(mu).add_(p, alpha=1 - mu)
                else:
                    mu = 0.0  # For display purposes

            pbar.set_description(f"loss: {loss_ema:.6f}, mu: {mu:.4f}, N: {N}")
        return

    # CHANGED: Added save/load methods
    def save_checkpoint(self, path, epoch, optimizer=None, scheduler=None, loss=None, **kwargs):
        """Save model checkpoint with configuration"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': self.get_config(),
            'loss': loss,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        checkpoint.update(kwargs)

        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, cond_embed_model=None, device=None):
        """Load model from checkpoint with automatic architecture detection"""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # CHANGED: Detect if checkpoint is from CircularAngleConsistencyModel
        state_dict = checkpoint['model_state_dict']
        has_input_norm = 'input_norm.weight' in state_dict
        has_output_norm = 'output_norm.weight' in state_dict
        # CHANGED: Auto-detect architecture

        # Load config with backwards compatibility
        config = checkpoint.get('config', {
            'nfeatures': checkpoint.get('nfeatures', 2),
            'condition_on': checkpoint.get('condition_on', checkpoint.get('img_features', 784)),  # CHANGED: Map img_features
            'eps': checkpoint.get('eps', 0.002),
            'nunits': checkpoint.get('nunits', 128),
            'depth': checkpoint.get('depth', 6),
            'cond_embed_type': checkpoint.get('cond_embed_type', 'linear'),
            'add_input_norm': has_input_norm,  # CHANGED: Auto-detect
            'add_output_norm': has_output_norm,  # CHANGED: Auto-detect
        })

        model = cls(**config, cond_embed_model=cond_embed_model, device=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Model loaded from checkpoint: {checkpoint_path}")
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            print(f"Loss: {checkpoint['loss']:.6f}")
        print(f"Config: {config}")
        print(f"Architecture: {'CircularAngle' if has_input_norm else 'Standard'} variant")

        return model, checkpoint
