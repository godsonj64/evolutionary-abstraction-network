from __future__ import annotations

import argparse
import time

import torch
from torch import optim

from ean import EANConfig, EvolutionaryAbstractionNetwork
from ean.losses.ean_loss import ean_loss


def make_toy_batch(batch_size: int, input_dim: int, classes: int, device: torch.device):
    """Synthetic nonlinear classification batch used as a GPU smoke test."""
    x = torch.randn(batch_size, input_dim, device=device)
    chunks = torch.chunk(x, classes, dim=1)
    scores = torch.stack([c.sum(dim=1) for c in chunks], dim=1)
    y = scores.argmax(dim=1) % classes
    return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-ready smoke training for Evolutionary Abstraction Network.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=48)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--abstraction-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=3)
    parser.add_argument("--initial-concepts", type=int, default=6)
    parser.add_argument("--max-concepts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--evolve-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    if choice == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("--device mps was requested, but Apple MPS is not available.")
        return torch.device("mps")
    if choice == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    if args.input_dim % args.output_dim != 0:
        raise ValueError("input_dim must be divisible by output_dim for this toy demo.")
    if args.initial_concepts < args.top_k:
        raise ValueError("initial_concepts must be >= top_k.")

    torch.manual_seed(42)
    device = resolve_device(args.device)

    cfg = EANConfig(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        abstraction_dim=args.abstraction_dim,
        initial_concepts=args.initial_concepts,
        max_concepts=args.max_concepts,
        top_k=args.top_k,
    )
    model = EvolutionaryAbstractionNetwork(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    start = time.time()
    last_events = {"born": 0, "mutated": 0, "merged": 0, "pruned": 0, "consolidated": 0}

    for step in range(args.steps):
        x, y = make_toy_batch(args.batch_size, args.input_dim, args.output_dim, device)
        out = model(x, store_memory=True)
        losses = ean_loss(
            output=out["output"],
            target=y,
            next_latent_pred=out["next_latent_prediction"],
            next_latent_target=out["latent"].detach(),
            routing_weights_full=out["routing_weights_full"],
        )

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.evolve_every == 0:
            last_events = model.evolve_from_outputs(out, next_latent_target=out["latent"].detach())
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        if step == 0 or (step + 1) % max(1, args.steps // 5) == 0 or step == args.steps - 1:
            pred = out["output"].argmax(dim=1)
            acc = (pred == y).float().mean().item()
            print(
                f"step={step + 1:04d} loss={float(losses['total'].detach()):.4f} "
                f"acc={acc:.3f} concepts={len(model.population)} events={last_events}"
            )

    elapsed = time.time() - start
    print(f"done steps={args.steps} concepts={len(model.population)} elapsed_sec={elapsed:.2f}")


if __name__ == "__main__":
    main()
