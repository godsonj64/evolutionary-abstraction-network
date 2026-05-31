from __future__ import annotations

"""Safe high-memory Vision-EAN runner.

Runs the 80 GB one-epoch workflow and enables WILDS download by default.
Use --no-download only when the dataset already exists under --data-dir.
"""

import runpy
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    target = root / "experiments" / "vision_ean_80gb_1epoch_train_infer.py"
    args = sys.argv[1:]
    no_download = "--no-download" in args
    args = [a for a in args if a != "--no-download"]
    if not no_download and "--download" not in args:
        args.append("--download")
    sys.argv = [str(target), *args]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
