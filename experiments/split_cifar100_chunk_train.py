from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "torchvision is required for this experiment. Install with: pip install torchvision"
    ) from exc

from ean import EANConfig, EvolutionaryAbstractionNetwork
from ean.losses.ean_loss import ean_loss


class FlattenImageDataset(Dataset):
    """Wrap an image dataset and return flattened image vectors.

    The current EAN prototype is vector-based. This experiment intentionally uses
    flattened CIFAR-100 images so we can test the evolutionary abstraction
    machinery before adding a CNN or ViT encoder.
    """

    def __init__(self, base: Dataset, allowed_classes: Sequence[int]):
        self.base = base
        self.allowed_classes = list(allowed_classes)
        self.class_to_local = {cls: i for i, cls in enumerate(self.allowed_classes)}
        targets = getattr(base, "targets")
        self.indices = [i for i, y in enumerate(targets) if int(y) in self.class_to_local]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y = self.base[self.indices[idx]]
        return x.flatten(), torch.tensor(self.class_to_local[int(y)], dtype=torch.long)


def limited_subset(dataset: Dataset, max_samples: int, seed: int) -> Subset:
    generator = torch.Generator().manual_seed(seed)
    n = min(max_samples, len(dataset))
    perm = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    return Subset(dataset, perm)


def build_class_chunks(num_chunks: int, classes_per_chunk: int) -> list[list[int]]:
    chunks = []
    start = 0
    for _ in range(num_chunks):
        chunks.append(list(range(start, start + classes_per_chunk)))
        start += classes_per_chunk
    return chunks


@torch.no_grad()
def evaluate(model: EvolutionaryAbstractionNetwork, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x, store_memory=False)
        pred = out["output"].argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    model.train()
    return correct / max(1, total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunked Split-CIFAR100 experiment for EAN.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default="outputs/cifar100_chunk_metrics.csv")
    parser.add_argument("--num-chunks", type=int, default=3)
    parser.add_argument("--classes-per-chunk", type=int, default=5)
    parser.add_argument("--train-samples-per-chunk", type=int, default=500)
    parser.add_argument("--test-samples-per-chunk", type=int, default=200)
    parser.add_argument("--epochs-per-chunk", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--abstraction-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--initial-concepts", type=int, default=6)
    parser.add_argument("--max-concepts", type=int, default=18)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--evolve-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if choice == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
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
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_base = datasets.CIFAR100(root=args.data_dir, train=True, transform=transform, download=True)
    test_base = datasets.CIFAR100(root=args.data_dir, train=False, transform=transform, download=True)

    chunks = build_class_chunks(args.num_chunks, args.classes_per_chunk)
    output_dim = args.classes_per_chunk
    input_dim = 3 * 32 * 32

    model = EvolutionaryAbstractionNetwork(
        EANConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=args.latent_dim,
            abstraction_dim=args.abstraction_dim,
            hidden_dim=args.hidden_dim,
            initial_concepts=args.initial_concepts,
            max_concepts=args.max_concepts,
            top_k=args.top_k,
        )
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"device={device}")
    print(f"chunks={chunks}")

    rows: list[dict[str, object]] = []
    global_step = 0
    seen_test_loaders: list[tuple[int, DataLoader]] = []

    for chunk_id, class_ids in enumerate(chunks):
        train_ds = FlattenImageDataset(train_base, class_ids)
        test_ds = FlattenImageDataset(test_base, class_ids)
        train_subset = limited_subset(train_ds, args.train_samples_per_chunk, args.seed + chunk_id)
        test_subset = limited_subset(test_ds, args.test_samples_per_chunk, args.seed + 1000 + chunk_id)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
        seen_test_loaders.append((chunk_id, test_loader))

        for epoch in range(args.epochs_per_chunk):
            for x, y in train_loader:
                global_step += 1
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

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

                events = {"born": 0, "mutated": 0, "merged": 0, "pruned": 0, "consolidated": 0}
                if global_step % args.evolve_every == 0:
                    events = model.evolve_from_outputs(out, next_latent_target=out["latent"].detach())
                    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

            evals = {f"acc_chunk_{cid}": evaluate(model, loader, device) for cid, loader in seen_test_loaders}
            row = {
                "chunk_id": chunk_id,
                "epoch": epoch,
                "global_step": global_step,
                "loss": float(losses["total"].detach().cpu()),
                "concepts": len(model.population),
                **events,
                **evals,
            }
            rows.append(row)
            print(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved={output_path}")
    print(f"final_concepts={len(model.population)}")


if __name__ == "__main__":
    main()
