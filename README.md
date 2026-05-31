# Vision-EAN: Evolutionary Abstraction Networks for Visual Concept Learning

**Vision-EAN** is the visual-input form of the **Evolutionary Abstraction Network** (EAN): a research prototype where internal concepts are treated as a dynamic population of abstractions rather than as a fixed hidden layer.

The current Vision-EAN implementation uses a compact CNN encoder, so the tested model is a **CNN-instantiated Vision-EAN**. The name Vision-EAN is intentionally broader than CNN-EAN because future encoders may include ResNet, ConvNeXt, Swin, ViT, or medical foundation encoders.

## Core principle

> Learning is the continuous evolution of abstractions under experience pressure.

EAN keeps gradient learning, but adds explicit concept-level dynamics:

- concept birth
- concept merge
- concept pruning
- concept mutation
- concept consolidation
- concept usage and entropy logging
- next-batch latent prediction pressure

## Architecture

```text
Image
  ↓
Visual encoder currently implemented as compact CNN
  ↓
Latent representation z
  ↓
Abstraction Field
  ↓
Concept Router
  ↓
Dynamic Concept Population
  ↓
Top-k Concept Modules
  ↓
Concept Aggregator
  ↓
Classifier + Latent World Model
  ↓
Evolution Controller
```

The WILDS full benchmark runner uses next-batch latent prediction:

```text
current batch latent z_t
predicted next latent z_hat_{t+1}
true next latent z_{t+1}
error = ||z_{t+1} - z_hat_{t+1}||
```

That prediction error is used by the evolution controller together with routing statistics, concept usage, concept age, fitness, novelty, redundancy, and concept entropy.

## Google Colab quickstart

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

This runs a small Camelyon17-WILDS Vision-EAN smoke test.

## Full Camelyon17-WILDS run

The full run downloads Camelyon17-WILDS and uses the full train/evaluation splits. It can take a long time on Colab, so start with 1 to 3 epochs.

```python
!python colab_run_vision_ean.py --device cuda --full --epochs 3
```

For a longer experiment:

```python
!python colab_run_vision_ean.py --device cuda --full --epochs 10
```

Outputs:

```text
outputs/wilds_full_metrics.csv
outputs/wilds_full_summary.json
```

## Direct WILDS scripts

Sampled WILDS experiment:

```bash
python experiments/wilds_image_benchmark_train.py \
  --dataset camelyon17 \
  --train-samples 512 \
  --eval-samples 256 \
  --epochs 10 \
  --batch-size 64 \
  --image-size 64 \
  --latent-dim 128 \
  --abstraction-dim 128 \
  --hidden-dim 256 \
  --initial-concepts 6 \
  --max-concepts 18 \
  --top-k 3 \
  --evolve-every 20 \
  --download \
  --device cuda
```

Full WILDS experiment with richer logs:

```bash
python experiments/wilds_full_benchmark_train.py \
  --dataset camelyon17 \
  --epochs 3 \
  --batch-size 64 \
  --image-size 64 \
  --latent-dim 128 \
  --abstraction-dim 128 \
  --hidden-dim 256 \
  --initial-concepts 8 \
  --max-concepts 24 \
  --top-k 3 \
  --evolve-every 50 \
  --log-every 100 \
  --download \
  --device cuda
```

## Public API

```python
from ean import VisionEANConfig, VisionEvolutionaryAbstractionNetwork

model = VisionEvolutionaryAbstractionNetwork(
    VisionEANConfig(
        output_dim=2,
        image_channels=3,
        latent_dim=128,
        abstraction_dim=128,
        hidden_dim=256,
        initial_concepts=8,
        max_concepts=24,
        top_k=3,
    )
)
```

Backward-compatible aliases are retained:

```python
from ean import ImageEANConfig, ImageEvolutionaryAbstractionNetwork
```

## Manual install

```bash
pip install -r requirements.txt
```

The Colab launcher sets `PYTHONPATH` automatically, so editable installation is not required.

## Run tests

```bash
python -m pytest -q
```

## Other demos

Synthetic GPU demo:

```bash
python gpu_demo_train.py --steps 50 --device cuda
```

CIFAR-100 chunk debug experiment remains available as a fallback:

```bash
python experiments/split_cifar100_chunk_train.py --device cuda
```

## Research status

This is a working research prototype, not a state-of-the-art claim. Current evidence shows that Vision-EAN can run on WILDS/Camelyon17, train with a CNN visual encoder, and log concept birth, merge, pruning, and entropy dynamics. Proper scientific validation still requires:

- CNN-only baseline
- ResNet or ViT baseline
- repeated random seeds
- ablation of birth/merge/prune/mutation/consolidation
- domain-shift analysis
- stronger diversity regularization

## Attribution

Original design: **Godson Johnson**.
