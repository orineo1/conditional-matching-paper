"""
uncond_model.py
===============
Unconditional DDPM UNet for 28x28 MNIST images.

Exports:
    - UnconditionalUnet
"""

import torch
import torch.nn as nn
from diffusers import UNet2DModel


class UnconditionalUnet(nn.Module):
    """
    Thin wrapper around HuggingFace UNet2DModel.
    No class conditioning — predicts noise from noisy image + timestep only.
    """

    def __init__(self):
        super().__init__()
        self.model = UNet2DModel(
            sample_size=28,
            in_channels=1, out_channels=1,
            layers_per_block=2,
            block_out_channels=(32, 64, 128),
            down_block_types=('DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'),
            up_block_types=('AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D'),
            dropout=0.1,
        )

    def forward(self, x, t):
        return self.model(x, t).sample
