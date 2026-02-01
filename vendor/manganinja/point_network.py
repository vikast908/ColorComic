"""PointNet for MangaNinja (from src/point_network.py).

Simple convolutional network that produces multi-scale spatial embeddings
from point correspondence maps, matching the 4 UNet scale levels.
"""

import torch
from torch import nn

from diffusers.models.modeling_utils import ModelMixin


class PointNet(ModelMixin):
    """Convolutional point encoder producing multi-scale embeddings.

    Input: ``(B, 1, 512, 512)`` — a point correspondence map.
    Output: list of 4 embeddings at different scales:
        - ``(B, 64*64, 320)``
        - ``(B, 32*32, 640)``
        - ``(B, 16*16, 1280)``
        - ``(B, 8*8, 1280)``
    """

    def __init__(self):
        super().__init__()

        # Block 1: 512 -> 64 (stride 8 via 3 layers of stride ~2-3)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=6, stride=6, padding=0),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(64),
            nn.Conv2d(128, 320, kernel_size=1),
        )

        # Block 2: 64 -> 32
        self.block2 = nn.Sequential(
            nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(320, 640, kernel_size=1),
        )

        # Block 3: 32 -> 16
        self.block3 = nn.Sequential(
            nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(640, 1280, kernel_size=1),
        )

        # Block 4: 16 -> 8
        self.block4 = nn.Sequential(
            nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(1280, 1280, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> list:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Point map of shape ``(B, 1, H, W)`` where H=W=512.

        Returns
        -------
        list of torch.Tensor
            Multi-scale embeddings transposed to ``(B, HW, C)``.
        """
        out1 = self.block1(x)  # (B, 320, 64, 64)
        out2 = self.block2(out1)  # (B, 640, 32, 32)
        out3 = self.block3(out2)  # (B, 1280, 16, 16)
        out4 = self.block4(out3)  # (B, 1280, 8, 8)

        results = []
        for feat in [out1, out2, out3, out4]:
            b, c, h, w = feat.shape
            results.append(feat.reshape(b, c, h * w).permute(0, 2, 1))  # (B, HW, C)

        return results
