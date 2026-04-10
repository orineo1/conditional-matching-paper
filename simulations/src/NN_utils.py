import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.nn as nn


# Define Swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MLPBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_out)
        self.linear2 = nn.Linear(dim_in, dim_out)
        self.norm1 = nn.LayerNorm(dim_in)  # LayerNorm added
        self.norm2 = nn.LayerNorm(dim_in)  # LayerNorm added

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor):
        residual = x
        out = self.linear1(x)
        out = self.norm1(out)  # Apply normalization
        out = self.swish(out)
        out = self.linear2(out)
        out = self.norm2(out)  # Apply normalization
        out = self.swish(out)
        return out + residual


class GenericNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(GenericNN, self).__init__()
        D = hidden_dims[0]
        assert all(h == D for h in hidden_dims), "All hidden dims must be the same for residuals"

        self.input_dim = input_dim
        self.output_dim = output_dim


        layers = []
        for _ in hidden_dims:
            layers.append(MLPBlock(D, D))
        self.res_blocks = nn.Sequential(*layers)

        self.output_layer = nn.Linear(D, output_dim)
        # self.denorm_output = nn.Identity()

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x, time_emb,always_time=False):
        x_original = x
        # x = self.norm_input(x)
        x = x + time_emb
        x = self.swish(x)
        for block in self.res_blocks:
            if always_time:
                x = block(x + time_emb)
            else:
                x = block(x)
        x = x + x_original
        x = self.output_layer(x)
        # x = self.denorm_output(x)
        return x

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "model": self.res_blocks  # Fixed to reflect actual submodule
        }



class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # For scalar timesteps, we need to start with a smaller dimension and project up
        self.dim = dim
        self.time_dim = 32  # Start with a smaller dimension for the time embedding

        # Projection layers
        self.proj1 = nn.Linear(self.time_dim, dim * 4)
        self.proj2 = nn.Linear(dim * 4, dim)

    def forward(self, t):
        # Ensure t is the right shape (batch_size, 1)
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)

        # batch_size = t.shape[0]

        # Create sinusoidal position embedding
        half_dim = self.time_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)

        # Shape: (batch_size, 1, half_dim)
        emb = t * emb.unsqueeze(0)

        # Apply sin and cos and reshape
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Now emb shape should be (batch_size, time_dim)

        # Project to higher dimension and back
        emb = self.proj1(emb)
        emb = F.silu(emb)
        emb = self.proj2(emb)

        return emb