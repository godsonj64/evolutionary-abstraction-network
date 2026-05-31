from __future__ import annotations

import torch
from torch import nn


class ConceptAggregator(nn.Module):
    def __init__(self, concept_output_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(concept_output_dim)
        self.proj = nn.Sequential(
            nn.Linear(concept_output_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, concept_outputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if concept_outputs.ndim != 3:
            raise ValueError("concept_outputs must have shape [batch, k, output_dim]")
        weighted = (concept_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return self.proj(self.norm(weighted))
