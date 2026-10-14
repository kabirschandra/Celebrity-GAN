
"""Convolutional discriminator for RGB face images."""

import torch
import torch.nn as nn
from constants import res


class Descriminator(nn.Module):
    def __init__(self):
        super().__init__()

        if res < 8 or (res & (res - 1)) != 0:
            raise ValueError("res must be a power of 2 >= 8")

        layers = []

        current_res = res
        in_channels = 3
        out_channels = 64

        # Reduce image resolution while increasing feature maps.
        while current_res > 4:
            layers.extend([
                nn.Conv2d(
                    in_channels, out_channels, 4, 2, 1
                ),
                nn.LeakyReLU(0.2)
            ])

            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)
            current_res //= 2

        # Classify the final feature maps.
        layers.extend([
            nn.Flatten(),
            nn.Linear(in_channels * 4 * 4, 1),
            nn.Sigmoid()
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
