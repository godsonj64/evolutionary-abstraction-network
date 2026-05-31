# Vision-EAN Google Colab Quickstart

Open a new Colab notebook and set:

```text
Runtime > Change runtime type > GPU
```

Then run:

```python
!git clone https://github.com/godsonj64/evolutionary-abstraction-network.git
%cd evolutionary-abstraction-network
!python colab_run_vision_ean.py --device cuda --quick
```

## Full Camelyon17-WILDS run

Start with 1 to 3 epochs because Camelyon17-WILDS is large.

```python
!python colab_run_vision_ean.py --device cuda --full --epochs 3
```

For a longer run:

```python
!python colab_run_vision_ean.py --device cuda --full --epochs 10
```

## Verify GPU manually

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## Outputs

Sampled/quick run:

```text
outputs/wilds_benchmark_metrics.csv
```

Full run:

```text
outputs/wilds_full_metrics.csv
outputs/wilds_full_summary.json
```

## Naming note

Vision-EAN is the visual-input form of EAN. The current implementation uses a compact CNN encoder, so the tested model is a CNN-instantiated Vision-EAN.
