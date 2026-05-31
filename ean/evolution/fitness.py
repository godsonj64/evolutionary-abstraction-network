from __future__ import annotations

import torch

from ean.concepts.concept_population import ConceptPopulation


class FitnessEvaluator:
    """Computes concept fitness from measurable evolutionary pressures."""

    def __init__(self, usefulness_weight=1.0, novelty_weight=0.25, stability_weight=0.25, redundancy_weight=0.5):
        self.usefulness_weight = usefulness_weight
        self.novelty_weight = novelty_weight
        self.stability_weight = stability_weight
        self.redundancy_weight = redundancy_weight

    @torch.no_grad()
    def __call__(
        self,
        population: ConceptPopulation,
        routing_weights: torch.Tensor,
        reconstruction_error: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = len(population)
        if routing_weights.ndim != 2 or routing_weights.shape[1] != n:
            raise ValueError("routing_weights must have shape [batch, num_concepts]")

        usefulness = routing_weights.mean(dim=0)
        prototypes = population.prototypes(device=routing_weights.device)
        normed = torch.nn.functional.normalize(prototypes, dim=-1)
        sim = normed @ normed.T
        redundancy = (sim.sum(dim=1) - 1.0) / max(1, n - 1)
        novelty = 1.0 - redundancy.clamp(0, 1)
        ages = torch.tensor([float(c.age.item()) for c in population], device=routing_weights.device)
        stability = torch.tanh(ages / 25.0)

        fitness = (
            self.usefulness_weight * usefulness
            + self.novelty_weight * novelty
            + self.stability_weight * stability
            - self.redundancy_weight * redundancy.clamp_min(0)
        )
        if reconstruction_error is not None:
            fitness = fitness - 0.1 * reconstruction_error.mean().detach()

        for i, concept in enumerate(population):
            concept.fitness.copy_(fitness[i].detach().cpu())
        return fitness
