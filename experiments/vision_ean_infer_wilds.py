from __future__ import annotations

"""Vision-EAN inference and visualization on WILDS/Camelyon17.

This script loads a trained Vision-EAN checkpoint, runs inference on a WILDS
split, and writes:

- prediction image grid
- concept-routing heatmap
- concept-usage bar plot
- latent PCA map
- predictions CSV

Example Colab usage:
    !python experiments/vision_ean_infer_wilds.py \
      --checkpoint outputs/vision_ean_last.pt \
      --dataset camelyon17 \
      --split test \
      --num-samples 64 \
      --device cuda

If your older training run did not save a checkpoint, rerun training with a
checkpoint-saving version or save model.state_dict() manually before using this.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from wilds import get_dataset

from ean import VisionEANConfig, VisionEvolutionaryAbstractionNetwork
from experiments.wilds_image_benchmark_train import WildsImageDataset, get_split_safely, resolve_device


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Vision-EAN inference and produce plots/maps.")
    p.add_argument("--checkpoint", type=str, default="outputs/vision_ean_last.pt")
    p.add_argument("--dataset", type=str, default="camelyon17")
    p.add_argument("--data-dir", type=str, default="./data/wilds")
    p.add_argument("--split", type=str, default="test", choices=["train", "id_val", "val", "test"])
    p.add_argument("--output-dir", type=str, default="outputs/vision_ean_inference")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--abstraction-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--initial-concepts", type=int, default=8)
    p.add_argument("--max-concepts", type=int, default=24)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--cnn-base-channels", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--download", action="store_true")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()


def denormalize(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(DEFAULT_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(DEFAULT_STD, device=x.device).view(3, 1, 1)
    y = x * std + mean
    y = y.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    return y


def deterministic_subset(dataset: Dataset, n: int, seed: int) -> Dataset:
    if n <= 0 or n >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    return Subset(dataset, idx)


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


def load_checkpoint_if_available(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> bool:
    if not checkpoint_path.exists():
        print(f"WARNING: checkpoint not found: {checkpoint_path}")
        print("Running with randomly initialized weights. Predictions will not represent your trained model.")
        return False
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    print(f"loaded_checkpoint={checkpoint_path}")
    return True


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :2] * s[:2]


@torch.no_grad()
def collect_outputs(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
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
        "images": torch.cat(images, dim=0),
        "labels": torch.cat(labels, dim=0),
        "probs": torch.cat(probs, dim=0),
        "preds": torch.cat(preds, dim=0),
        "latents": torch.cat(latents, dim=0),
        "routing": torch.cat(routing, dim=0),
        "active": torch.cat(active, dim=0),
    }


def save_predictions_csv(data: dict[str, Any], path: Path) -> None:
    labels = data["labels"].numpy()
    preds = data["preds"].numpy()
    probs = data["probs"].numpy()
    active = data["active"].numpy()
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "true_label", "pred_label", "prob_class_0", "prob_class_1", "top_concepts"])
        for i in range(len(labels)):
            row = [i, int(labels[i]), int(preds[i])]
            row += [float(v) for v in probs[i, :2]]
            row += [" ".join(map(str, active[i].tolist()))]
            writer.writerow(row)


def plot_image_grid(data: dict[str, Any], output: Path, max_images: int = 16) -> None:
    n = min(max_images, data["images"].shape[0])
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 3.2))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i in range(n):
        img = denormalize(data["images"][i])
        label = int(data["labels"][i])
        pred = int(data["preds"][i])
        conf = float(data["probs"][i, pred])
        concepts = ",".join(map(str, data["active"][i].tolist()))
        axes[i].imshow(img)
        axes[i].set_title(f"y={label} | pred={pred} | p={conf:.2f}\nconcepts: {concepts}", fontsize=9)
    fig.suptitle("Vision-EAN inference samples", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_routing_heatmap(data: dict[str, Any], output: Path) -> None:
    routing = data["routing"].numpy()
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(routing, aspect="auto", interpolation="nearest")
    ax.set_title("Vision-EAN concept routing weights", fontsize=15, fontweight="bold")
    ax.set_xlabel("Concept index")
    ax.set_ylabel("Sample index")
    fig.colorbar(im, ax=ax, label="routing weight")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_concept_usage(data: dict[str, Any], output: Path) -> None:
    routing = data["routing"].numpy()
    usage = routing.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(np.arange(len(usage)), usage)
    ax.set_title("Mean concept usage over inference samples", fontsize=15, fontweight="bold")
    ax.set_xlabel("Concept index")
    ax.set_ylabel("mean routing weight")
    ax.set_xticks(np.arange(len(usage)))
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_latent_pca(data: dict[str, Any], output: Path) -> None:
    z = data["latents"].numpy()
    coords = pca_2d(z)
    labels = data["labels"].numpy()
    preds = data["preds"].numpy()
    correct = labels == preds
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=correct.astype(int), s=42, alpha=0.85)
    ax.set_title("Latent PCA map: correct vs incorrect predictions", fontsize=14, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    handles, _ = scatter.legend_elements()
    if len(handles) >= 2:
        ax.legend(handles, ["incorrect", "correct"], title="prediction")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
    ])
    dataset = get_dataset(dataset=args.dataset, root_dir=args.data_dir, download=args.download)
    n_classes = int(getattr(dataset, "n_classes", 2))
    raw_split = get_split_safely(dataset, args.split, transform)
    if raw_split is None:
        raise RuntimeError(f"Could not load split {args.split!r}")
    wrapped = WildsImageDataset(raw_split, flatten=False)
    subset = deterministic_subset(wrapped, args.num_samples, args.seed)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")

    model = build_model(args, n_classes).to(device)
    checkpoint_loaded = load_checkpoint_if_available(model, Path(args.checkpoint), device)
    data = collect_outputs(model, loader, device)

    csv_path = output_dir / "predictions.csv"
    grid_path = output_dir / "prediction_image_grid.png"
    heatmap_path = output_dir / "concept_routing_heatmap.png"
    usage_path = output_dir / "concept_usage_bar.png"
    pca_path = output_dir / "latent_pca_map.png"

    save_predictions_csv(data, csv_path)
    plot_image_grid(data, grid_path)
    plot_routing_heatmap(data, heatmap_path)
    plot_concept_usage(data, usage_path)
    plot_latent_pca(data, pca_path)

    acc = float((data["labels"] == data["preds"]).float().mean().item())
    print({
        "model_family": "Vision-EAN",
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": int(data["labels"].numel()),
        "accuracy_on_sample": acc,
        "output_dir": str(output_dir),
    })
    print(f"saved={csv_path}")
    print(f"saved={grid_path}")
    print(f"saved={heatmap_path}")
    print(f"saved={usage_path}")
    print(f"saved={pca_path}")


if __name__ == "__main__":
    main()
