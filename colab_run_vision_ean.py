from __future__ import annotations

"""One-command Colab launcher for Vision-EAN on WILDS.

Usage in Google Colab:
    !python colab_run_vision_ean.py --device cuda --quick

Full Camelyon17-WILDS benchmark:
    !python colab_run_vision_ean.py --device cuda --full --epochs 3

Scientific naming note:
    Vision-EAN is the visual-input form of EAN. The current implementation uses
    a compact CNN encoder, so the tested model is a CNN-instantiated Vision-EAN.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab one-command launcher for Vision-EAN WILDS experiments.")
    parser.add_argument("--wilds-dataset", default="camelyon17", help="WILDS dataset name. Default: camelyon17")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--quick", action="store_true", help="Small debug run using sampled WILDS splits.")
    parser.add_argument("--full", action="store_true", help="Run full WILDS train/eval splits with richer logs.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip dependency installation.")
    parser.add_argument("--no-download", action="store_true", help="Do not download benchmark data if missing.")
    return parser.parse_args()


def resolve_device(choice: str) -> str:
    import torch

    if choice != "auto":
        if choice == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sampled_args(args: argparse.Namespace) -> list[str]:
    if args.quick:
        return [
            "--dataset", args.wilds_dataset,
            "--train-samples", "128",
            "--eval-samples", "64",
            "--epochs", str(args.epochs if args.epochs is not None else 10),
            "--batch-size", "32",
            "--image-size", "48",
            "--latent-dim", "64",
            "--abstraction-dim", "64",
            "--hidden-dim", "128",
            "--initial-concepts", "5",
            "--max-concepts", "12",
            "--top-k", "2",
            "--evolve-every", "10",
        ]
    return [
        "--dataset", args.wilds_dataset,
        "--train-samples", "512",
        "--eval-samples", "256",
        "--epochs", str(args.epochs if args.epochs is not None else 10),
        "--batch-size", "64",
        "--image-size", "64",
        "--latent-dim", "128",
        "--abstraction-dim", "128",
        "--hidden-dim", "256",
        "--initial-concepts", "6",
        "--max-concepts", "18",
        "--top-k", "3",
        "--evolve-every", "20",
    ]


def full_args(args: argparse.Namespace) -> list[str]:
    return [
        "--dataset", args.wilds_dataset,
        "--epochs", str(args.epochs if args.epochs is not None else 3),
        "--batch-size", "64",
        "--image-size", "64",
        "--latent-dim", "128",
        "--abstraction-dim", "128",
        "--hidden-dim", "256",
        "--initial-concepts", "8",
        "--max-concepts", "24",
        "--top-k", "3",
        "--evolve-every", "50",
        "--log-every", "100",
    ]


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    os.environ["PYTHONPATH"] = str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(root / "requirements.txt")])

    import torch

    device = resolve_device(args.device)
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"selected_device={device}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")

    if args.full:
        script = root / "experiments" / "wilds_full_benchmark_train.py"
        exp_args = full_args(args)
        done_msg = "Finished. Metrics saved to outputs/wilds_full_metrics.csv and outputs/wilds_full_summary.json"
    else:
        script = root / "experiments" / "wilds_image_benchmark_train.py"
        exp_args = sampled_args(args)
        done_msg = "Finished. Metrics saved to outputs/wilds_benchmark_metrics.csv"

    if not args.no_download:
        exp_args.append("--download")

    run([
        sys.executable,
        str(script),
        *exp_args,
        "--device",
        device,
    ])

    print(f"\n{done_msg}")


if __name__ == "__main__":
    main()
