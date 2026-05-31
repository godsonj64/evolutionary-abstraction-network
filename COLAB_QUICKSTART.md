# Google Colab Quickstart

Open a new Colab notebook, set **Runtime > Change runtime type > GPU**, then run:

```python
!git clone https://github.com/godsonj64/evolutionary-abstraction-network.git
%cd evolutionary-abstraction-network
!pip install -e .
!pip install -r requirements.txt
!pytest -q
!python gpu_demo_train.py --steps 50
```

Verify GPU:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```
