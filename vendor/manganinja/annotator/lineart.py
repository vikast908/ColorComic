"""Lineart annotator for MangaNinja (from src/annotator/lineart/__init__.py).

Extracts clean line art from manga images using a learned ResNet generator.
Auto-downloads sk_model.pth from HuggingFace Annotators repo.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


class ResidualBlock(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    """ResNet-based image-to-image generator for line art extraction."""

    def __init__(self, input_nc: int, output_nc: int, n_residual_blocks: int = 3):
        super().__init__()

        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class BatchLineartDetector:
    """Batch-capable line art detector.

    Loads ``sk_model.pth`` from the HuggingFace Annotators repo and uses it
    to extract clean line art from input images.
    """

    def __init__(self, ckpt_dir: str):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.model = Generator(3, 1, 3)

        model_path = os.path.join(ckpt_dir, "sk_model.pth")
        if not os.path.exists(model_path):
            os.makedirs(ckpt_dir, exist_ok=True)
            print("Downloading lineart annotator weights...")
            model_path = hf_hub_download(
                repo_id="lllyasviel/Annotators",
                filename="sk_model.pth",
                local_dir=ckpt_dir,
            )
            print("Lineart annotator weights downloaded.")

        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.model = self.model.to(device=device, dtype=dtype or self.dtype)
        return self

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Extract line art from a batch of images.

        Parameters
        ----------
        images : torch.Tensor
            Batch of RGB images, shape ``(B, 3, H, W)``, values in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Line art images, shape ``(B, 1, H, W)``, values in ``[0, 1]``.
        """
        # Normalize: mean=-1, std=2 -> maps [0,1] to [-1, 0]
        # Actually the annotator expects input normalized differently:
        # x = (x - 0.5) / 0.5 = x*2 - 1  (maps [0,1] to [-1,1])
        x = images.to(device=self.device, dtype=self.dtype)
        x = (x - 0.5) / 0.5  # normalize to [-1, 1]
        output = self.model(x)
        # Output is in [-1, 1] (tanh), convert to [0, 1] as line art
        lineart = 1.0 - (output + 1.0) / 2.0  # invert: dark lines on white
        return lineart.clamp(0, 1)
