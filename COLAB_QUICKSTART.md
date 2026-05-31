# Google Colab Quickstart

Open a new Colab notebook, set **Runtime > Change runtime type > GPU**, then run:

```python
!git clone https://github.com/godsonj64/evolutionary-abstraction-network.git
%cd evolutionary-abstraction-network
!python colab_run_cifar_chunk.py
```

For a faster debug run:

```python
!python colab_run_cifar_chunk.py --quick
```

To force CUDA:

```python
!python colab_run_cifar_chunk.py --device cuda
```

Verify GPU manually:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

The launcher automatically installs dependencies, installs the package in editable mode, selects CUDA if available, runs the chunked CIFAR-100 experiment, and saves metrics to:

```text
outputs/cifar100_chunk_metrics.csv
```
