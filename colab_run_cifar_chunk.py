from __future__ import annotations

"""One-command Colab launcher for EAN benchmark experiments.

Default usage in Colab:
    !python colab_run_cifar_chunk.py

The default now runs a real WILDS benchmark smoke test. CIFAR-100 remains
available as a fallback/debug mode.
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
    parser = argparse.ArgumentParser(description="Colab one-command launcher for EAN benchmark experiments.")
    parser.add_argument("--benchmark", default="wilds", choices=["wilds", "cifar"], help="Benchmark to run. Default: wilds")
    parser.add_argument("--wilds-dataset", default="camelyon17", help="WILDS dataset name. Default: camelyon17")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--quick", action="store_true", help="Use a smaller debug run.")
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


def wilds_args(args: argparse.Namespace) -> list[str]:
    if args.quick:
        return [
            "--dataset", args.wilds_dataset,
            "--train-samples", "128",
            "--eval-samples", "64",
            "--epochs", "1",
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
        "--epochs", "1",
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


def cifar_args(args: argparse.Namespace) -> list[str]:
    if args.quick:
        return [
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
    return [
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

    if args.benchmark == "wilds":
        exp_args = wilds_args(args)
        if not args.no_download:
            exp_args.append("--download")
        script = root / "experiments" / "wilds_image_benchmark_train.py"
        done_msg = "Finished. Metrics saved to outputs/wilds_benchmark_metrics.csv"
    else:
        exp_args = cifar_args(args)
        script = root / "experiments" / "split_cifar100_chunk_train.py"
        done_msg = "Finished. Metrics saved to outputs/cifar100_chunk_metrics.csv"

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
