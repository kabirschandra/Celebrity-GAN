
"""Convolutional generator for producing 32x32 RGB face images.

Takes 100 random numbers and progressively transforms them into
a 32x32 RGB image using transposed convolutional layers.
"""


import torch
import torch.nn as nn
from constants import res, latent_dim


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # Resolution must be a power of 2, at least 8.
        if res < 8 or (res & (res - 1)) != 0:
            raise ValueError("res must be 8, 16, 32, 64, 128, etc.")

        layers = [
            # Convert random noise into 256 feature maps of size 4x4.
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.Unflatten(1, (256, 4, 4)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ]

        # Repeatedly double the image resolution.
        current_res = 4
        channels = 256

        while current_res < res // 2:
            next_channels = max(channels // 2, 32)

            layers.extend([
                # Double width and height.
                nn.ConvTranspose2d(
                    channels, next_channels, 4, 2, 1
                ),
                nn.BatchNorm2d(next_channels),
                nn.ReLU()
            ])

            channels = next_channels
            current_res *= 2

        # Final upsampling layer produces a 3-channel RGB image.
        layers.extend([
            nn.ConvTranspose2d(channels, 3, 4, 2, 1),
            nn.Tanh()
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

