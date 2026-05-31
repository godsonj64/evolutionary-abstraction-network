from __future__ import annotations

import torch
from torch import nn


class ConceptModule(nn.Module):
    """A small neural concept organism.

    The prototype stores the semantic centroid of experiences this concept has
    explained. The stability flag allows consolidation without freezing the full
    model externally.
    """

    def __init__(self, latent_dim: int, abstraction_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.abstraction_dim = abstraction_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + abstraction_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        self.register_buffer("prototype", torch.zeros(abstraction_dim))
        self.register_buffer("age", torch.zeros((), dtype=torch.long))
        self.register_buffer("usage", torch.zeros(()))
        self.register_buffer("fitness", torch.zeros(()))
        self.consolidated = False

    def forward(self, z: torch.Tensor, abstraction_summary: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, abstraction_summary], dim=-1)
        return self.net(x)

    @torch.no_grad()
    def update_stats(self, abstraction_summary: torch.Tensor, responsibility: torch.Tensor, momentum: float = 0.95) -> None:
        if abstraction_summary.ndim != 2:
            raise ValueError("abstraction_summary must have shape [batch, abstraction_dim]")
        responsibility = responsibility.detach().float().reshape(-1)
        total = responsibility.sum().clamp_min(1e-8)
        centroid = (abstraction_summary.detach() * responsibility[:, None]).sum(dim=0) / total
        if self.usage.item() <= 0:
            self.prototype.copy_(centroid)
        else:
            self.prototype.mul_(momentum).add_(centroid, alpha=1.0 - momentum)
        self.usage.add_(total / max(1, abstraction_summary.shape[0]))
        self.age.add_(1)

    def mutate(self, sigma: float = 1e-3) -> None:
        with torch.no_grad():
            for p in self.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * sigma)
