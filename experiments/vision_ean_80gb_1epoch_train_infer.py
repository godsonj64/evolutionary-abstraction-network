from __future__ import annotations

"""High-memory GPU Vision-EAN one-epoch train + checkpoint + inference workflow.

Designed for large GPU instances, for example 80 GB VRAM and high system RAM.
It trains one full Camelyon17-WILDS epoch with a large batch, saves checkpoints,
evaluates all requested splits, and immediately produces inference figures.

Example:
    python experiments/vision_ean_80gb_1epoch_train_infer.py \
      --device cuda \
      --batch-size 512 \
      --eval-batch-size 512 \
      --num-workers 8 \
      --download

Outputs are written to:
    outputs/vision_ean_80gb_1epoch/
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from wilds import get_dataset

from ean import VisionEANConfig, VisionEvolutionaryAbstractionNetwork
from ean.losses.ean_loss import ean_loss
from experiments.wilds_image_benchmark_train import WildsImageDataset, get_split_safely, resolve_device

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
EVENT_KEYS = ("born", "mutated", "merged", "pruned", "consolidated")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="High-memory one-epoch Vision-EAN train + inference workflow.")
    p.add_argument("--dataset", default="camelyon17")
    p.add_argument("--data-dir", default="./data/wilds")
    p.add_argument("--output-dir", default="outputs/vision_ean_80gb_1epoch")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-splits", default="id_val,val,test")
    p.add_argument("--epochs", type=int, default=1, help="Default is 1 full epoch as requested.")
    p.add_argument("--batch-size", type=int, default=512, help="Large training batch for 80 GB GPU.")
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--cnn-base-channels", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--abstraction-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--initial-concepts", type=int, default=8)
    p.add_argument("--max-concepts", type=int, default=24)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--evolve-every", type=int, default=25)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--inference-split", default="test", choices=["train", "id_val", "val", "test"])
    p.add_argument("--inference-samples", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--download", action="store_true")
    p.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA. Recommended for 80 GB GPUs.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )


def build_model(args: argparse.Namespace, n_classes: int) -> VisionEvolutionaryAbstractionNetwork:
    return VisionEvolutionaryAbstractionNetwork(
        VisionEANConfig(
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


def empty_events() -> dict[str, int]:
    return {k: 0 for k in EVENT_KEYS}


def add_events(total: dict[str, int], events: dict[str, int]) -> None:
    for k in EVENT_KEYS:
        total[k] += int(events.get(k, 0))


def concept_usage_entropy(model: VisionEvolutionaryAbstractionNetwork) -> float:
    usages = torch.tensor([float(c.usage.item()) for c in model.population], dtype=torch.float32)
    if usages.numel() == 0 or float(usages.sum()) <= 0.0:
        return 0.0
    probs = usages / usages.sum().clamp_min(1e-8)
    return float((-(probs * probs.clamp_min(1e-8).log()).sum()).item())


@torch.no_grad()
def encode_latent(model: VisionEvolutionaryAbstractionNetwork, batch: Any, device: torch.device) -> torch.Tensor:
    x = batch[0].to(device, non_blocking=True)
    return model.encoder(x).detach()


def align_latent_batches(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.shape[0] == source.shape[0]:
        return target
    if target.shape[0] > source.shape[0]:
        return target[: source.shape[0]]
    repeats = (source.shape[0] + target.shape[0] - 1) // target.shape[0]
    return target.repeat((repeats, 1))[: source.shape[0]]


@torch.no_grad()
def evaluate(model: VisionEvolutionaryAbstractionNetwork, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y, _metadata in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x, store_memory=False)
        pred = out["output"].argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    model.train()
    return correct / max(1, total)


def save_checkpoint(path: Path, model: VisionEvolutionaryAbstractionNetwork, optimizer: optim.Optimizer, args: argparse.Namespace, epoch: int, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_family": "Vision-EAN",
            "vision_encoder": "cnn",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": {
                "output_dim": int(model.config.output_dim),
                "image_channels": int(model.config.image_channels),
                "latent_dim": int(model.config.latent_dim),
                "abstraction_dim": int(model.config.abstraction_dim),
                "hidden_dim": int(model.config.hidden_dim),
                "cnn_base_channels": int(model.config.cnn_base_channels),
                "initial_concepts": int(model.config.initial_concepts),
                "max_concepts": int(model.config.max_concepts),
                "top_k": int(model.config.top_k),
            },
            "args": vars(args),
        },
        path,
    )


def denormalize(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(DEFAULT_MEAN).view(3, 1, 1)
    std = torch.tensor(DEFAULT_STD).view(3, 1, 1)
    y = (x.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return y


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :2] * s[:2]


def deterministic_subset(dataset: Dataset, n: int, seed: int) -> Dataset:
    if n <= 0 or n >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    return Subset(dataset, idx)


@torch.no_grad()
def collect_inference(model: VisionEvolutionaryAbstractionNetwork, loader: DataLoader, device: torch.device) -> dict[str, torch.Tensor]:
    model.eval()
    images, labels, probs, preds, latents, routing, active = [], [], [], [], [], [], []
    for x, y, _metadata in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x, store_memory=False)
        prob = torch.softmax(out["output"], dim=1)
        pred = prob.argmax(dim=1)
        images.append(x.detach().cpu())
        labels.append(y.detach().cpu())
        probs.append(prob.detach().cpu())
        preds.append(pred.detach().cpu())
        latents.append(out["latent"].detach().cpu())
        routing.append(out["routing_weights_full"].detach().cpu())
        active.append(out["active_concepts"].detach().cpu())
    return {
        "images": torch.cat(images),
        "labels": torch.cat(labels),
        "probs": torch.cat(probs),
        "preds": torch.cat(preds),
        "latents": torch.cat(latents),
        "routing": torch.cat(routing),
        "active": torch.cat(active),
    }


def write_predictions_csv(data: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = data["labels"].numpy()
    preds = data["preds"].numpy()
    probs = data["probs"].numpy()
    active = data["active"].numpy()
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["sample", "true_label", "pred_label"] + [f"prob_class_{i}" for i in range(probs.shape[1])] + ["top_concepts"]
        writer.writerow(header)
        for i in range(len(labels)):
            writer.writerow([i, int(labels[i]), int(preds[i]), *[float(v) for v in probs[i]], " ".join(map(str, active[i].tolist()))])


def plot_prediction_grid(data: dict[str, torch.Tensor], path: Path, max_images: int = 24) -> None:
    n = min(max_images, data["images"].shape[0])
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.75))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i in range(n):
        pred = int(data["preds"][i])
        label = int(data["labels"][i])
        conf = float(data["probs"][i, pred])
        concepts = ",".join(map(str, data["active"][i].tolist()))
        axes[i].imshow(denormalize(data["images"][i]))
        axes[i].set_title(f"y={label} pred={pred} p={conf:.2f}\nC:{concepts}", fontsize=8)
    fig.suptitle("Vision-EAN inference sample grid", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_routing_heatmap(data: dict[str, torch.Tensor], path: Path) -> None:
    routing = data["routing"].numpy()
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(routing, aspect="auto", interpolation="nearest")
    ax.set_title("Vision-EAN concept routing map", fontsize=15, fontweight="bold")
    ax.set_xlabel("Concept index")
    ax.set_ylabel("Inference sample")
    fig.colorbar(im, ax=ax, label="routing weight")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_concept_usage(data: dict[str, torch.Tensor], path: Path) -> None:
    usage = data["routing"].numpy().mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(np.arange(len(usage)), usage)
    ax.set_title("Mean concept usage on inference sample", fontsize=15, fontweight="bold")
    ax.set_xlabel("Concept index")
    ax.set_ylabel("mean routing weight")
    ax.set_xticks(np.arange(len(usage)))
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_latent_pca(data: dict[str, torch.Tensor], path: Path) -> None:
    coords = pca_2d(data["latents"].numpy())
    labels = data["labels"].numpy()
    preds = data["preds"].numpy()
    correct = (labels == preds).astype(int)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=correct, s=42, alpha=0.85)
    ax.set_title("Vision-EAN latent PCA map", fontsize=15, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    handles, _ = scatter.legend_elements()
    if len(handles) >= 2:
        ax.legend(handles, ["incorrect", "correct"], title="prediction")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    fig_dir = output_dir / "inference_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
    ])
    dataset = get_dataset(dataset=args.dataset, root_dir=args.data_dir, download=args.download)
    n_classes = int(getattr(dataset, "n_classes", 2))

    train_raw = get_split_safely(dataset, args.train_split, transform)
    if train_raw is None:
        raise RuntimeError(f"Could not load train split {args.train_split!r}")
    train_data = WildsImageDataset(train_raw, flatten=False)
    train_loader = make_loader(train_data, args.batch_size, True, args.num_workers, device)

    eval_loaders = {}
    eval_sizes = {}
    for split in [s.strip() for s in args.eval_splits.split(",") if s.strip()]:
        raw = get_split_safely(dataset, split, transform)
        if raw is None:
            continue
        wrapped = WildsImageDataset(raw, flatten=False)
        eval_sizes[split] = len(wrapped)
        eval_loaders[split] = make_loader(wrapped, args.eval_batch_size, False, args.num_workers, device)

    model = build_model(args, n_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    print({
        "model_family": "Vision-EAN",
        "workflow": "80gb_1epoch_train_infer",
        "dataset": args.dataset,
        "device": str(device),
        "train_samples": len(train_data),
        "eval_sizes": eval_sizes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "amp": bool(scaler.is_enabled()),
    })

    rows = []
    best_metric = -1.0
    best_split = "val" if "val" in eval_loaders else next(iter(eval_loaders), "")
    global_step = 0
    cumulative_events = empty_events()
    last_events = empty_events()
    run_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        iterator = iter(train_loader)
        try:
            batch = next(iterator)
        except StopIteration:
            break

        loss_sum = 0.0
        correct = 0
        total = 0
        batch_idx = 0

        while True:
            try:
                lookahead = next(iterator)
                has_next = True
            except StopIteration:
                lookahead = batch
                has_next = False

            batch_idx += 1
            global_step += 1
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)

            with torch.no_grad():
                next_target = encode_latent(model, lookahead, device)

            with autocast(enabled=scaler.is_enabled()):
                out = model(x, store_memory=True)
                next_target = align_latent_batches(out["latent"], next_target)
                losses = ean_loss(out["output"], y, out["next_latent_prediction"], next_target, out["routing_weights_full"])

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = int(y.numel())
            loss_sum += float(losses["total"].detach().cpu()) * bs
            correct += int((out["output"].argmax(dim=1) == y).sum().item())
            total += bs

            if global_step % args.evolve_every == 0:
                last_events = model.evolve_from_outputs(out, next_latent_target=next_target.detach())
                add_events(cumulative_events, last_events)
                optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            if args.log_every > 0 and (global_step % args.log_every == 0 or not has_next):
                print({
                    "type": "batch",
                    "epoch": epoch,
                    "batch": batch_idx,
                    "global_step": global_step,
                    "train_loss": loss_sum / max(1, total),
                    "train_acc": correct / max(1, total),
                    "concepts": len(model.population),
                    "concept_entropy": concept_usage_entropy(model),
                    **last_events,
                    **{f"total_{k}": v for k, v in cumulative_events.items()},
                })

            if not has_next:
                break
            batch = lookahead

        evals = {f"acc_{split}": evaluate(model, loader, device) for split, loader in eval_loaders.items()}
        row = {
            "model_family": "Vision-EAN",
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": loss_sum / max(1, total),
            "train_acc": correct / max(1, total),
            "concepts": len(model.population),
            "concept_entropy": concept_usage_entropy(model),
            "epoch_sec": time.time() - epoch_start,
            **last_events,
            **{f"total_{k}": v for k, v in cumulative_events.items()},
            **evals,
        }
        rows.append(row)
        print(row)

        save_checkpoint(ckpt_dir / "vision_ean_last.pt", model, optimizer, args, epoch, row)
        current_metric = float(row.get(f"acc_{best_split}", -1.0))
        if current_metric > best_metric:
            best_metric = current_metric
            save_checkpoint(ckpt_dir / "vision_ean_best_val.pt", model, optimizer, args, epoch, row)

    metrics_csv = output_dir / "vision_ean_80gb_metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        fields = sorted({k for r in rows for k in r}) if rows else []
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model_family": "Vision-EAN",
        "workflow": "80gb_1epoch_train_infer",
        "dataset": args.dataset,
        "device": str(device),
        "train_samples": len(train_data),
        "eval_sizes": eval_sizes,
        "epochs": args.epochs,
        "global_steps": global_step,
        "final_concepts": len(model.population),
        "final_concept_entropy": concept_usage_entropy(model),
        "cumulative_events": cumulative_events,
        "runtime_sec": time.time() - run_start,
        "last_epoch": rows[-1] if rows else {},
        "last_checkpoint": str(ckpt_dir / "vision_ean_last.pt"),
        "best_checkpoint": str(ckpt_dir / "vision_ean_best_val.pt"),
    }
    summary_json = output_dir / "vision_ean_80gb_summary.json"
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)

    raw_infer = get_split_safely(dataset, args.inference_split, transform)
    if raw_infer is None:
        raise RuntimeError(f"Could not load inference split {args.inference_split!r}")
    infer_data = deterministic_subset(WildsImageDataset(raw_infer, flatten=False), args.inference_samples, args.seed)
    infer_loader = make_loader(infer_data, args.eval_batch_size, False, args.num_workers, device)
    inference = collect_inference(model, infer_loader, device)
    infer_acc = float((inference["labels"] == inference["preds"]).float().mean().item())

    write_predictions_csv(inference, output_dir / "inference_predictions.csv")
    plot_prediction_grid(inference, fig_dir / "vision_ean_sample_predictions.png")
    plot_routing_heatmap(inference, fig_dir / "vision_ean_concept_routing_heatmap.png")
    plot_concept_usage(inference, fig_dir / "vision_ean_concept_usage_bar.png")
    plot_latent_pca(inference, fig_dir / "vision_ean_latent_pca_map.png")

    print({
        "saved_metrics": str(metrics_csv),
        "saved_summary": str(summary_json),
        "saved_last_checkpoint": str(ckpt_dir / "vision_ean_last.pt"),
        "saved_best_checkpoint": str(ckpt_dir / "vision_ean_best_val.pt"),
        "inference_split": args.inference_split,
        "inference_samples": args.inference_samples,
        "inference_sample_accuracy": infer_acc,
        "inference_figures": str(fig_dir),
    })


if __name__ == "__main__":
    main()
