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
            idx = population.append_concept(abstraction_summary.detach().mean(dim=0).cpu())
            if idx >= 0:
                events["born"] += 1

        for c in population:
            if float(c.fitness.item()) > 0.1 and not c.consolidated:
                c.mutate(self.mutation_sigma)
                events["mutated"] += 1
            if float(c.fitness.item()) > 0.4 and int(c.age.item()) > 10:
                c.consolidated = True
                events["consolidated"] += 1

        events["merged"] += self._merge_redundant(population)
        events["pruned"] += self._prune_weak(population)
        return events

    @torch.no_grad()
    def _merge_redundant(self, population: ConceptPopulation) -> int:
        n = len(population)
        if n <= self.min_concepts:
            return 0
        p = torch.nn.functional.normalize(population.prototypes(), dim=-1)
        sim = p @ p.T
        sim.fill_diagonal_(-1.0)
        young = torch.tensor([int(c.age.item()) < 5 for c in population], dtype=torch.bool)
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
        merged = 0.5 * (population[keep].prototype + population[drop].prototype)
        population[keep].prototype.copy_(merged)
        keep_indices = [idx for idx in range(n) if idx != drop]
        population.remove_concepts(keep_indices)
        return 1

    @torch.no_grad()
    def _prune_weak(self, population: ConceptPopulation) -> int:
        if len(population) <= self.min_concepts:
            return 0
        keep = []
        for i, c in enumerate(population):
            too_young_to_prune = int(c.age.item()) < 5
            must_keep_for_minimum = len(population) - len(keep) <= self.min_concepts
            if too_young_to_prune or must_keep_for_minimum or float(c.fitness.item()) >= self.prune_threshold:
                keep.append(i)
        removed = len(population) - len(keep)
        if removed > 0:
            population.remove_concepts(keep)
        return removed
