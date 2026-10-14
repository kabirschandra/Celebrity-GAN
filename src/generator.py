
"""Convolutional generator for RGB face images.

Uses the resolution and latent dimension from constants.py.
"""

import torch
import torch.nn as nn
from constants import res, latent_dim


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        if res < 8 or (res & (res - 1)) != 0:
            raise ValueError("res must be a power of 2 >= 8")

        # Start with 512 feature maps at 4x4.
        start_channels = 512

        layers = [
            nn.Linear(latent_dim, start_channels * 4 * 4),
            nn.Unflatten(1, (start_channels, 4, 4)),
            nn.BatchNorm2d(start_channels),
            nn.ReLU()
        ]

        current_res = 4
        channels = start_channels

        # Double spatial resolution at each stage.
        while current_res < res // 2:
            next_channels = max(channels // 2, 64)

            layers.extend([
                nn.ConvTranspose2d(
                    channels, next_channels, 4, 2, 1
                ),
                nn.BatchNorm2d(next_channels),
                nn.ReLU()
            ])

            channels = next_channels
            current_res *= 2

        # Convert feature maps into RGB pixels.
        layers.extend([
            nn.ConvTranspose2d(channels, 3, 4, 2, 1),
            nn.Tanh()
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)
