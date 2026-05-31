from __future__ import annotations

import torch
from torch import nn


class AbstractionField(nn.Module):
    """Produces multiple abstraction levels from a latent observation.

    Each level has its own projection. Low levels can encode concrete patterns;
    high levels can specialize toward broader relational structure.
    """

    def __init__(self, latent_dim: int, abstraction_dim: int, levels: int = 3):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be >= 1")
        self.levels = levels
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_dim, abstraction_dim),
                    nn.GELU(),
                    nn.LayerNorm(abstraction_dim),
                )
                for _ in range(levels)
            ]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"Expected z with shape [batch, latent_dim], got {tuple(z.shape)}")
        return torch.stack([proj(z) for proj in self.projections], dim=1)
