from __future__ import annotations

import torch
from torch import nn

from .concept_population import ConceptPopulation


class ConceptRouter(nn.Module):
    """Routes inputs to concepts using learned scores plus prototype affinity."""

    def __init__(self, latent_dim: int, abstraction_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(latent_dim + abstraction_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.query = nn.Linear(hidden_dim, abstraction_dim)
        self.bias = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor, abstraction_summary: torch.Tensor, population: ConceptPopulation) -> torch.Tensor:
        h = self.score(torch.cat([z, abstraction_summary], dim=-1))
        q = torch.nn.functional.normalize(self.query(h), dim=-1)
        prototypes = population.prototypes(device=z.device)
        p = torch.nn.functional.normalize(prototypes, dim=-1)
        affinity = q @ p.T
        return affinity + self.bias(h)
