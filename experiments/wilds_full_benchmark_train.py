from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from wilds import get_dataset

from ean.losses.ean_loss import ean_loss
from experiments.wilds_image_benchmark_train import (
    WildsImageDataset,
    build_model,
    evaluate,
    get_split_safely,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full WILDS/CNN-EAN benchmark with richer logs.")
    p.add_argument("--dataset", default="camelyon17")
    p.add_argument("--data-dir", default="./data/wilds")
    p.add_argument("--output", default="outputs/wilds_full_metrics.csv")
    p.add_argument("--summary-output", default="outputs/wilds_full_summary.json")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-splits", default="id_val,val,test")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--encoder", default="cnn", choices=["cnn", "flatten"])
    p.add_argument("--cnn-base-channels", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--abstraction-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--initial-concepts", type=int, default=8)
    p.add_argument("--max-concepts", type=int, default=24)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--evolve-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--download", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output = Path(args.output)
    summary_output = Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = get_dataset(dataset=args.dataset, root_dir=args.data_dir, download=args.download)
    n_classes = int(getattr(dataset, "n_classes", 2))
    flatten = args.encoder == "flatten"

    train_raw = get_split_safely(dataset, args.train_split, transform)
    if train_raw is None:
        raise RuntimeError(f"Could not load train split {args.train_split!r}")
    train_data = WildsImageDataset(train_raw, flatten=flatten)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")

    eval_loaders = {}
    eval_sizes = {}
    for split in [s.strip() for s in args.eval_splits.split(",") if s.strip()]:
        raw = get_split_safely(dataset, split, transform)
        if raw is None:
            continue
        wrapped = WildsImageDataset(raw, flatten=flatten)
        eval_sizes[split] = len(wrapped)
        eval_loaders[split] = DataLoader(wrapped, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")

    model = build_model(args, n_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print({
        "dataset": args.dataset,
        "device": str(device),
        "encoder": args.encoder,
        "train_samples": len(train_data),
        "eval_sizes": eval_sizes,
        "epochs": args.epochs,
    })

    rows = []
    best = {}
    global_step = 0
    last_events = {"born": 0, "mutated": 0, "merged": 0, "pruned": 0, "consolidated": 0}
    start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        loss_sum = 0.0
        correct = 0
        total = 0
        model.train()

        for batch_idx, batch in enumerate(train_loader, start=1):
            global_step += 1
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            out = model(x, store_memory=True)
            losses = ean_loss(out["output"], y, out["next_latent_prediction"], out["latent"].detach(), out["routing_weights_full"])
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = int(y.numel())
            loss_sum += float(losses["total"].detach().cpu()) * bs
            correct += int((out["output"].argmax(dim=1) == y).sum().item())
            total += bs

            if global_step % args.evolve_every == 0:
                last_events = model.evolve_from_outputs(out, next_latent_target=out["latent"].detach())
                optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

            if args.log_every > 0 and global_step % args.log_every == 0:
                print({
                    "type": "batch",
                    "epoch": epoch,
                    "batch": batch_idx,
                    "global_step": global_step,
                    "train_loss": loss_sum / max(1, total),
                    "train_acc": correct / max(1, total),
                    "concepts": len(model.population),
                    **last_events,
                })

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": loss_sum / max(1, total),
            "train_acc": correct / max(1, total),
            "concepts": len(model.population),
            "epoch_sec": time.time() - epoch_start,
            **last_events,
        }
        for split, loader in eval_loaders.items():
            acc = evaluate(model, loader, device)
            row[f"acc_{split}"] = acc
            if split not in best or acc > best[split]["accuracy"]:
                best[split] = {"epoch": epoch, "accuracy": acc, "global_step": global_step, "concepts": len(model.population)}
        rows.append(row)
        print(row)

    fields = sorted({k for r in rows for k in r})
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "device": str(device),
        "train_samples": len(train_data),
        "eval_sizes": eval_sizes,
        "epochs": args.epochs,
        "global_steps": global_step,
        "final_concepts": len(model.population),
        "runtime_sec": time.time() - start,
        "best_by_split": best,
        "last_epoch": rows[-1] if rows else {},
    }
    with summary_output.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"saved={output}")
    print(f"summary_saved={summary_output}")
    print(f"best_by_split={best}")


if __name__ == "__main__":
    main()
