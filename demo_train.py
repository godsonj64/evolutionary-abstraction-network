from __future__ import annotations

import torch
from torch import optim

from ean import EANConfig, EvolutionaryAbstractionNetwork
from ean.losses.ean_loss import ean_loss


def make_toy_batch(batch_size: int = 32, input_dim: int = 16, classes: int = 3):
    x = torch.randn(batch_size, input_dim)
    score = torch.stack([x[:, :5].sum(dim=1), x[:, 5:10].sum(dim=1), x[:, 10:].sum(dim=1)], dim=1)
    y = score.argmax(dim=1) % classes
    return x, y


def main():
    torch.manual_seed(42)
    model = EvolutionaryAbstractionNetwork(EANConfig(input_dim=16, output_dim=3, initial_concepts=4, max_concepts=8))
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(1):
        x, y = make_toy_batch()
        out = model(x, store_memory=True)
        losses = ean_loss(
            output=out["output"],
            target=y,
            next_latent_pred=out["next_latent_prediction"],
            next_latent_target=out["latent"],
            routing_weights_full=out["routing_weights_full"],
        )
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 5 == 0:
            events = model.evolve_from_outputs(out, next_latent_target=out["latent"].detach())
        else:
            events = {"born": 0, "mutated": 0, "merged": 0, "pruned": 0, "consolidated": 0}

        print(
            f"step={step:02d} loss={float(losses['total'].detach()):.4f} "
            f"concepts={len(model.population)} events={events}"
        )


if __name__ == "__main__":
    main()
