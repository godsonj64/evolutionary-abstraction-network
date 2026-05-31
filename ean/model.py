from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ean.abstraction.abstraction_field import AbstractionField
from ean.concepts.aggregator import ConceptAggregator
from ean.concepts.concept_population import ConceptPopulation, ConceptPopulationConfig
from ean.concepts.router import ConceptRouter
from ean.encoders.simple_encoder import MLPEncoder
from ean.evolution.controller import EvolutionController
from ean.memory.episodic_memory import EpisodicMemory
from ean.world_model.latent_predictor import LatentWorldModel


@dataclass(frozen=True)
class EANConfig:
    input_dim: int
    output_dim: int
    latent_dim: int = 64
    abstraction_dim: int = 64
    hidden_dim: int = 128
    abstraction_levels: int = 3
    initial_concepts: int = 4
    max_concepts: int = 32
    top_k: int = 2
    memory_capacity: int = 4096


class EvolutionaryAbstractionNetwork(nn.Module):
    """Evolutionary Abstraction Network prototype.

    The architecture keeps the original concept intact:
    - perception encoder creates latent evidence;
    - abstraction field creates multi-level abstractions;
    - concept population represents evolving internal abstractions;
    - router selects concepts sparsely;
    - world model pressures concepts to be predictive;
    - evolution controller changes the concept population over time.
    """

    def __init__(self, config: EANConfig):
        super().__init__()
        if config.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if config.initial_concepts < config.top_k:
            raise ValueError("initial_concepts must be >= top_k")
        self.config = config
        self.encoder = MLPEncoder(config.input_dim, config.latent_dim, config.hidden_dim)
        self.abstraction_field = AbstractionField(config.latent_dim, config.abstraction_dim, config.abstraction_levels)
        self.population = ConceptPopulation(
            ConceptPopulationConfig(
                latent_dim=config.latent_dim,
                abstraction_dim=config.abstraction_dim,
                output_dim=config.hidden_dim,
                hidden_dim=config.hidden_dim,
                initial_concepts=config.initial_concepts,
                max_concepts=config.max_concepts,
            )
        )
        self.router = ConceptRouter(config.latent_dim, config.abstraction_dim, config.hidden_dim)
        self.aggregator = ConceptAggregator(config.hidden_dim, config.hidden_dim)
        self.output_head = nn.Linear(config.hidden_dim, config.output_dim)
        self.world_model = LatentWorldModel(config.latent_dim, config.hidden_dim)
        self.memory = EpisodicMemory(config.memory_capacity)
        self.evolution = EvolutionController()

    def _abstraction_summary(self, abstractions: torch.Tensor) -> torch.Tensor:
        return abstractions.mean(dim=1)

    def forward(self, x: torch.Tensor, store_memory: bool = False) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        abstractions = self.abstraction_field(z)
        abstraction_summary = self._abstraction_summary(abstractions)

        scores = self.router(z, abstraction_summary, self.population)
        routing_weights_full = torch.softmax(scores, dim=-1)
        k = min(self.config.top_k, len(self.population))
        active_weights, active_ids = torch.topk(routing_weights_full, k=k, dim=-1)
        active_weights = active_weights / active_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        concept_outputs = []
        batch_size = x.shape[0]
        for slot in range(k):
            slot_output = torch.zeros(batch_size, self.config.hidden_dim, device=x.device, dtype=z.dtype)
            for concept_id in torch.unique(active_ids[:, slot]).tolist():
                mask = active_ids[:, slot] == int(concept_id)
                if mask.any():
                    slot_output[mask] = self.population[int(concept_id)](z[mask], abstraction_summary[mask])
            concept_outputs.append(slot_output)

        concept_outputs_tensor = torch.stack(concept_outputs, dim=1)
        h = self.aggregator(concept_outputs_tensor, active_weights)
        output = self.output_head(h)
        next_latent_pred = self.world_model(z, h)

        if store_memory:
            self.memory.add(x, z, output)
            self._update_concept_stats(abstraction_summary, routing_weights_full)

        return {
            "output": output,
            "latent": z,
            "abstractions": abstractions,
            "abstraction_summary": abstraction_summary,
            "routing_scores": scores,
            "routing_weights_full": routing_weights_full,
            "active_concepts": active_ids,
            "active_weights": active_weights,
            "hidden": h,
            "next_latent_prediction": next_latent_pred,
        }

    @torch.no_grad()
    def _update_concept_stats(self, abstraction_summary: torch.Tensor, routing_weights_full: torch.Tensor) -> None:
        for i, concept in enumerate(self.population):
            concept.update_stats(abstraction_summary, routing_weights_full[:, i])

    def evolve_from_outputs(self, outputs: dict[str, torch.Tensor], next_latent_target: torch.Tensor | None = None) -> dict[str, int]:
        if next_latent_target is None:
            prediction_error = None
        else:
            prediction_error = (outputs["next_latent_prediction"].detach() - next_latent_target.detach()).pow(2).mean(dim=-1)
        return self.evolution.step(
            self.population,
            outputs["abstraction_summary"].detach(),
            outputs["routing_weights_full"].detach(),
            prediction_error,
        )
