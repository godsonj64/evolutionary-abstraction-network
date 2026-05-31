from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .concept_module import ConceptModule


@dataclass(frozen=True)
class ConceptPopulationConfig:
    latent_dim: int
    abstraction_dim: int
    output_dim: int
    hidden_dim: int = 128
    initial_concepts: int = 4
    max_concepts: int = 32


class ConceptPopulation(nn.Module):
    """Dynamic population of concept modules.

    Uses ModuleList so PyTorch correctly registers new concepts introduced after
    initialization. Newly born concepts inherit the current population device,
    which is essential when the model is already on CUDA/MPS.
    """

    def __init__(self, config: ConceptPopulationConfig):
        super().__init__()
        self.config = config
        self.concepts = nn.ModuleList([self._new_concept() for _ in range(config.initial_concepts)])

    def _new_concept(self) -> ConceptModule:
        return ConceptModule(
            latent_dim=self.config.latent_dim,
            abstraction_dim=self.config.abstraction_dim,
            output_dim=self.config.output_dim,
            hidden_dim=self.config.hidden_dim,
        )

    def _population_device(self) -> torch.device:
        if len(self.concepts) == 0:
            return torch.device("cpu")
        return next(self.concepts[0].parameters()).device

    def __len__(self) -> int:
        return len(self.concepts)

    def __iter__(self) -> Iterable[ConceptModule]:
        return iter(self.concepts)

    def __getitem__(self, idx: int) -> ConceptModule:
        return self.concepts[idx]

    def append_concept(self, prototype: torch.Tensor | None = None) -> int:
        if len(self.concepts) >= self.config.max_concepts:
            return -1

        device = self._population_device()
        concept = self._new_concept().to(device)

        if prototype is not None:
            if prototype.shape[-1] != self.config.abstraction_dim:
                raise ValueError("prototype has wrong dimension")
            with torch.no_grad():
                concept.prototype.copy_(prototype.detach().to(device=device, dtype=concept.prototype.dtype))

        self.concepts.append(concept)
        return len(self.concepts) - 1

    def remove_concepts(self, keep_indices: list[int]) -> None:
        if not keep_indices:
            raise ValueError("Cannot remove all concepts")
        self.concepts = nn.ModuleList([self.concepts[i] for i in keep_indices])

    @torch.no_grad()
    def prototypes(self, device: torch.device | None = None) -> torch.Tensor:
        if len(self.concepts) == 0:
            raise RuntimeError("Population is empty")
        target_device = device if device is not None else self._population_device()
        return torch.stack([c.prototype.detach().to(target_device) for c in self.concepts], dim=0)
