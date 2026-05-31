from __future__ import annotations

import torch
from torch import nn


class MLPEncoder(nn.Module):
    """Small modality-agnostic encoder for vectors.

    This prototype encoder can be replaced by a transformer, ViT, audio encoder,
    or multimodal encoder without changing the evolutionary abstraction core.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape [batch, input_dim], got {tuple(x.shape)}")
        return self.net(x)
