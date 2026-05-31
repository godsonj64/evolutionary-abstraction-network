from __future__ import annotations

import torch

from ean.concepts.concept_population import ConceptPopulation
from .fitness import FitnessEvaluator


class EvolutionController:
    """Birth, mutation, merge, pruning, and consolidation of concept modules."""

    def __init__(
        self,
        birth_error_threshold: float = 0.8,
        novelty_threshold: float = 0.35,
        prune_threshold: float = -0.05,
        merge_similarity_threshold: float = 0.985,
        mutation_sigma: float = 1e-3,
        min_concepts: int = 2,
    ):
        self.birth_error_threshold = birth_error_threshold
        self.novelty_threshold = novelty_threshold
        self.prune_threshold = prune_threshold
        self.merge_similarity_threshold = merge_similarity_threshold
        self.mutation_sigma = mutation_sigma
        self.min_concepts = min_concepts
        self.fitness_evaluator = FitnessEvaluator()

    @torch.no_grad()
    def novelty(self, abstraction_summary: torch.Tensor, population: ConceptPopulation) -> torch.Tensor:
        prototypes = population.prototypes(device=abstraction_summary.device)
        q = torch.nn.functional.normalize(abstraction_summary, dim=-1)
        p = torch.nn.functional.normalize(prototypes, dim=-1)
        max_sim = (q @ p.T).max(dim=1).values
        return 1.0 - max_sim

    def step(
        self,
        population: ConceptPopulation,
        abstraction_summary: torch.Tensor,
        routing_weights_full: torch.Tensor,
        prediction_error: torch.Tensor | None = None,
    ) -> dict[str, int]:
        self.fitness_evaluator(population, routing_weights_full, prediction_error)
        events = {"born": 0, "mutated": 0, "merged": 0, "pruned": 0, "consolidated": 0}

        error_value = 0.0 if prediction_error is None else float(prediction_error.detach().mean().cpu())
        novelty_value = float(self.novelty(abstraction_summary.detach(), population).mean().cpu())
        if error_value > self.birth_error_threshold and novelty_value > self.novelty_threshold:
            idx = population.append_concept(abstraction_summary.detach().mean(dim=0))
            if idx >= 0:
                events["born"] += 1

        for c in population:
            if float(c.fitness.item()) > 0.1 and not c.consolidated:
                c.mutate(self.mutation_sigma)
                events["mutated"] += 1
            if float(c.fitness.item()) > 0.4 and int(c.age.item()) > 10:
                c.consolidated = True
                events["consolidated"] += 1

        events["merged"] += self._merge_redundant(population, device=abstraction_summary.device)
        events["pruned"] += self._prune_weak(population)
        return events

    @torch.no_grad()
    def _merge_redundant(self, population: ConceptPopulation, device: torch.device | None = None) -> int:
        n = len(population)
        if n <= self.min_concepts:
            return 0
        p = torch.nn.functional.normalize(population.prototypes(device=device), dim=-1)
        sim = p @ p.T
        sim.fill_diagonal_(-1.0)
        young = torch.tensor([int(c.age.item()) < 5 for c in population], dtype=torch.bool, device=sim.device)
        if young.any():
            sim[young, :] = -1.0
            sim[:, young] = -1.0
        max_val, flat_idx = sim.flatten().max(dim=0)
        if float(max_val) < self.merge_similarity_threshold:
            return 0
        i = int(flat_idx // n)
        j = int(flat_idx % n)
        if i == j:
            return 0
        keep, drop = min(i, j), max(i, j)
        merged = 0.5 * (population[keep].prototype.to(sim.device) + population[drop].prototype.to(sim.device))
        population[keep].prototype.copy_(merged.to(population[keep].prototype.device))
        keep_indices = [idx for idx in range(n) if idx != drop]
        population.remove_concepts(keep_indices)
        return 1

    @torch.no_grad()
    def _prune_weak(self, population: ConceptPopulation) -> int:
        n = len(population)
        if n <= self.min_concepts:
            return 0

        min_keep = min(self.min_concepts, n)
        fitness = torch.tensor([float(c.fitness.item()) for c in population])
        mature = torch.tensor([int(c.age.item()) >= 5 for c in population], dtype=torch.bool)

        keep_set: set[int] = set()

        # Always preserve young concepts long enough to collect evidence.
        for i in range(n):
            if not bool(mature[i]):
                keep_set.add(i)

        # Preserve mature concepts whose fitness is above the pruning threshold.
        for i in range(n):
            if bool(mature[i]) and float(fitness[i]) >= self.prune_threshold:
                keep_set.add(i)

        # Safety invariant: never allow ecological collapse of the concept pool.
        if len(keep_set) < min_keep:
            strongest = torch.topk(fitness, k=min_keep).indices.tolist()
            keep_set.update(int(i) for i in strongest)

        keep = sorted(keep_set)
        removed = n - len(keep)
        if removed > 0:
            population.remove_concepts(keep)
        return removed
