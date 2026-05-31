from __future__ import annotations

import torch
from torch import nn


class SmallCNNEncoder(nn.Module):
    """Compact image encoder for WILDS/CIFAR smoke benchmarks.

    The encoder maps image tensors [B, C, H, W] into latent vectors consumed by
    the existing EAN abstraction field. It is intentionally lightweight so it can
    run in Colab CPU/GPU sessions without requiring pretrained weights.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        base_channels: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected image tensor [batch, channels, height, width], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")
        return self.projection(self.features(x))
