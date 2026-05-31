from __future__ import annotations

"""One-command Colab launcher for the chunked CIFAR-100 EAN experiment.

Usage in Colab:
    !python colab_run_cifar_chunk.py

Optional:
    !python colab_run_cifar_chunk.py --device cuda --quick
    !python colab_run_cifar_chunk.py --device cpu
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
    parser = argparse.ArgumentParser(description="Colab one-command launcher for EAN CIFAR-100 chunk experiment.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--quick", action="store_true", help="Use a smaller debug run.")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip dependency installation.")
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


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    # Make the repo importable without relying on pip editable installs.
    # This avoids Colab build-backend failures and keeps the launcher robust.
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

    if args.quick:
        exp_args = [
            "--num-chunks", "2",
            "--classes-per-chunk", "3",
            "--train-samples-per-chunk", "120",
            "--test-samples-per-chunk", "60",
            "--epochs-per-chunk", "1",
            "--batch-size", "32",
            "--latent-dim", "64",
            "--abstraction-dim", "64",
            "--hidden-dim", "128",
            "--initial-concepts", "5",
            "--max-concepts", "12",
            "--top-k", "2",
            "--evolve-every", "10",
        ]
    else:
        exp_args = [
            "--num-chunks", "3",
            "--classes-per-chunk", "5",
            "--train-samples-per-chunk", "500",
            "--test-samples-per-chunk", "200",
            "--epochs-per-chunk", "1",
            "--batch-size", "64",
            "--latent-dim", "128",
            "--abstraction-dim", "128",
            "--hidden-dim", "256",
            "--initial-concepts", "6",
            "--max-concepts", "18",
            "--top-k", "3",
            "--evolve-every", "20",
        ]

    run([
        sys.executable,
        str(root / "experiments" / "split_cifar100_chunk_train.py"),
        *exp_args,
        "--device",
        device,
    ])

    print("\nFinished. Metrics saved to outputs/cifar100_chunk_metrics.csv")


if __name__ == "__main__":
    main()
