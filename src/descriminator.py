
"""Convolutional discriminator for classifying real and fake faces.

Receives a 32x32 RGB image and progressively extracts features
using convolutional layers before producing one probability.
"""


import torch
import torch.nn as nn
from constants import res


class Descriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Resolution must be a power of 2, at least 8.
        if res < 8 or (res & (res - 1)) != 0:
            raise ValueError("res must be 8, 16, 32, 64, 128, etc.")

        layers = []

        current_res = res
        in_channels = 3
        out_channels = 32

        # Repeatedly halve the image resolution until 4x4.
        while current_res > 4:
            layers.extend([
                nn.Conv2d(
                    in_channels, out_channels, 4, 2, 1
                ),
                nn.LeakyReLU(0.2)
            ])

            in_channels = out_channels
            out_channels = min(out_channels * 2, 256)
            current_res //= 2

        # Convert the final 4x4 feature maps into a probability.
        layers.extend([
            nn.Flatten(),
            nn.Linear(in_channels * 4 * 4, 1),
            nn.Sigmoid()
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
