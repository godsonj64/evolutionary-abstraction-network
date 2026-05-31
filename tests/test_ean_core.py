from __future__ import annotations

import torch

from ean import EANConfig, EvolutionaryAbstractionNetwork
from ean.losses.ean_loss import ean_loss


def make_model() -> EvolutionaryAbstractionNetwork:
    torch.manual_seed(7)
    return EvolutionaryAbstractionNetwork(
        EANConfig(
            input_dim=16,
            output_dim=3,
            latent_dim=32,
            abstraction_dim=32,
            hidden_dim=48,
            abstraction_levels=3,
            initial_concepts=4,
            max_concepts=8,
            top_k=2,
        )
    )


def test_forward_shapes_and_routing_normalization():
    model = make_model()
    x = torch.randn(5, 16)
    out = model(x)

    assert out["output"].shape == (5, 3)
    assert out["latent"].shape == (5, 32)
    assert out["abstractions"].shape == (5, 3, 32)
    assert out["abstraction_summary"].shape == (5, 32)
    assert out["active_concepts"].shape == (5, 2)
    assert out["active_weights"].shape == (5, 2)
    assert out["next_latent_prediction"].shape == (5, 32)
    assert torch.allclose(out["routing_weights_full"].sum(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.allclose(out["active_weights"].sum(dim=-1), torch.ones(5), atol=1e-6)


def test_loss_backward_updates_gradients():
    model = make_model()
    x = torch.randn(8, 16)
    target = torch.randint(0, 3, (8,))
    out = model(x)
    losses = ean_loss(
        output=out["output"],
        target=target,
        next_latent_pred=out["next_latent_prediction"],
        next_latent_target=out["latent"],
        routing_weights_full=out["routing_weights_full"],
    )
    losses["total"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_memory_and_concept_stats_update():
    model = make_model()
    x = torch.randn(6, 16)
    _ = model(x, store_memory=True)
    assert len(model.memory) == 1
    assert any(float(c.usage.item()) > 0 for c in model.population)
    assert all(int(c.age.item()) == 1 for c in model.population)


def test_evolution_birth_when_error_and_novelty_are_high():
    model = make_model()
    model.evolution.birth_error_threshold = 0.01
    model.evolution.novelty_threshold = -1.0
    before = len(model.population)
    x = torch.randn(10, 16) + 5.0
    out = model(x, store_memory=True)
    target = out["latent"] + 10.0
    events = model.evolve_from_outputs(out, next_latent_target=target)
    assert events["born"] == 1
    assert len(model.population) == before + 1


def test_population_respects_max_concepts():
    model = EvolutionaryAbstractionNetwork(
        EANConfig(
            input_dim=4,
            output_dim=2,
            latent_dim=8,
            abstraction_dim=8,
            hidden_dim=16,
            initial_concepts=2,
            max_concepts=2,
            top_k=1,
        )
    )
    idx = model.population.append_concept(torch.zeros(8))
    assert idx == -1
    assert len(model.population) == 2


def test_merge_redundant_concepts_after_maturation():
    model = EvolutionaryAbstractionNetwork(
        EANConfig(
            input_dim=4,
            output_dim=2,
            latent_dim=8,
            abstraction_dim=8,
            hidden_dim=16,
            initial_concepts=3,
            max_concepts=5,
            top_k=1,
        )
    )
    with torch.no_grad():
        model.population[0].prototype.fill_(1.0)
        model.population[1].prototype.fill_(1.0)
        model.population[2].prototype.copy_(torch.arange(8).float())
        for concept in model.population:
            concept.age.fill_(20)
    model.evolution.min_concepts = 2
    model.evolution.merge_similarity_threshold = 0.99
    merged = model.evolution._merge_redundant(model.population)
    assert merged == 1
    assert len(model.population) == 2


def test_merge_protects_immature_redundant_concepts():
    model = EvolutionaryAbstractionNetwork(
        EANConfig(
            input_dim=4,
            output_dim=2,
            latent_dim=8,
            abstraction_dim=8,
            hidden_dim=16,
            initial_concepts=3,
            max_concepts=5,
            top_k=1,
        )
    )
    with torch.no_grad():
        model.population[0].prototype.fill_(1.0)
        model.population[1].prototype.fill_(1.0)
        model.population[2].prototype.copy_(torch.arange(8).float())
        for concept in model.population:
            concept.age.fill_(5)
    model.evolution.min_concepts = 2
    model.evolution.merge_similarity_threshold = 0.99
    merged = model.evolution._merge_redundant(model.population)
    assert merged == 0
    assert len(model.population) == 3


def test_invalid_input_shape_raises():
    model = make_model()
    bad = torch.randn(2, 3, 16)
    try:
        model(bad)
    except ValueError as exc:
        assert "Expected x" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid input shape")
