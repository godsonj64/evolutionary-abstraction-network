from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

# Allow direct execution by file path, for example:
# python experiments/wilds_image_benchmark_train.py
# In that mode, Python places experiments/ on sys.path rather than the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from torchvision import transforms
except ImportError as exc:  # pragma: no cover
    raise SystemExit("torchvision is required. Install with: pip install torchvision") from exc

try:
    from wilds import get_dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit("wilds is required. Install with: pip install wilds") from exc

from ean import EANConfig, EvolutionaryAbstractionNetwork
from ean.image_model import ImageEANConfig, ImageEvolutionaryAbstractionNetwork
from ean.losses.ean_loss import ean_loss


class WildsImageDataset(Dataset):
    """Wrap a WILDS image subset and keep image tensors in spatial form."""

    def __init__(self, subset: Dataset, flatten: bool = False):
        self.subset = subset
        self.flatten = flatten

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        item = self.subset[idx]
        if len(item) == 3:
            x, y, metadata = item
        elif len(item) == 2:
            x, y = item
            metadata = torch.empty(0)
        else:
            raise ValueError(f"Unexpected WILDS item length: {len(item)}")
        if not torch.is_tensor(x):
            raise TypeError("Expected transformed image tensor from WILDS subset")
        if self.flatten:
            x = x.flatten()
        y = torch.as_tensor(y).long().view(-1)[0]
        return x, y, torch.as_tensor(metadata)


def limited_subset(dataset: Dataset, max_samples: int, seed: int) -> Subset:
    generator = torch.Generator().manual_seed(seed)
    n = min(max_samples, len(dataset))
    perm = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    return Subset(dataset, perm)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x, store_memory=False)
        pred = out["output"].argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    model.train()
    return correct / max(1, total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WILDS image benchmark smoke test for EAN.")
    parser.add_argument("--dataset", type=str, default="camelyon17", help="WILDS image dataset name. Default: camelyon17")
    parser.add_argument("--data-dir", type=str, default="./data/wilds")
    parser.add_argument("--output", type=str, default="outputs/wilds_benchmark_metrics.csv")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-splits", type=str, default="id_val,val,test", help="Comma-separated split names. Missing splits are skipped.")
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--encoder", type=str, default="cnn", choices=["cnn", "flatten"], help="Use CNN image encoder or flattened vector encoder. Default: cnn")
    parser.add_argument("--cnn-base-channels", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--abstraction-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--initial-concepts", type=int, default=6)
    parser.add_argument("--max-concepts", type=int, default=18)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--evolve-every", type=int, default=20)
    parser.add_argument("--download", action="store_true", help="Download dataset if missing.")
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


def get_split_safely(dataset: Any, split: str, transform: Any):
    try:
        return dataset.get_subset(split, transform=transform)
    except Exception as exc:
        print(f"Skipping split={split!r}: {exc}")
        return None


def build_model(args: argparse.Namespace, n_classes: int) -> nn.Module:
    if args.encoder == "cnn":
        return ImageEvolutionaryAbstractionNetwork(
            ImageEANConfig(
                output_dim=n_classes,
                image_channels=3,
                latent_dim=args.latent_dim,
                abstraction_dim=args.abstraction_dim,
                hidden_dim=args.hidden_dim,
                cnn_base_channels=args.cnn_base_channels,
                initial_concepts=args.initial_concepts,
                max_concepts=args.max_concepts,
                top_k=args.top_k,
            )
        )

    input_dim = 3 * args.image_size * args.image_size
    return EvolutionaryAbstractionNetwork(
        EANConfig(
            input_dim=input_dim,
            output_dim=n_classes,
            latent_dim=args.latent_dim,
            abstraction_dim=args.abstraction_dim,
            hidden_dim=args.hidden_dim,
            initial_concepts=args.initial_concepts,
            max_concepts=args.max_concepts,
            top_k=args.top_k,
        )
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = get_dataset(dataset=args.dataset, root_dir=args.data_dir, download=args.download)
    n_classes = int(getattr(dataset, "n_classes", 2))
    flatten = args.encoder == "flatten"

    train_subset_raw = get_split_safely(dataset, args.train_split, transform)
    if train_subset_raw is None:
        raise RuntimeError(f"Could not load train split {args.train_split!r}")
    train_data = WildsImageDataset(train_subset_raw, flatten=flatten)
    train_data = limited_subset(train_data, args.train_samples, args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    eval_loaders = {}
    for split in [s.strip() for s in args.eval_splits.split(",") if s.strip()]:
        raw = get_split_safely(dataset, split, transform)
        if raw is None:
            continue
        wrapped = WildsImageDataset(raw, flatten=flatten)
        wrapped = limited_subset(wrapped, args.eval_samples, args.seed + hash(split) % 10000)
        eval_loaders[split] = DataLoader(
            wrapped,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=device.type == "cuda",
        )

    model = build_model(args, n_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"dataset={args.dataset}")
    print(f"device={device}")
    print(f"encoder={args.encoder}")
    print(f"n_classes={n_classes}")
    print(f"image_size={args.image_size}")
    print(f"train_samples={len(train_data)}")
    print(f"eval_splits={list(eval_loaders.keys())}")

    rows: list[dict[str, object]] = []
    global_step = 0
    last_events = {"born": 0, "mutated": 0, "merged": 0, "pruned": 0, "consolidated": 0}

    for epoch in range(args.epochs):
        for batch in train_loader:
            x, y = batch[0], batch[1]
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

            if global_step % args.evolve_every == 0:
                last_events = model.evolve_from_outputs(out, next_latent_target=out["latent"].detach())
                optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        evals = {f"acc_{split}": evaluate(model, loader, device) for split, loader in eval_loaders.items()}
        row = {
            "dataset": args.dataset,
            "encoder": args.encoder,
            "epoch": epoch,
            "global_step": global_step,
            "loss": float(losses["total"].detach().cpu()),
            "concepts": len(model.population),
            **last_events,
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
