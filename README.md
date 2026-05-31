# Evolutionary Abstraction Network (EAN)

A research prototype for **Evolutionary Abstraction Learning**: a neural architecture where internal concepts are treated as an evolving population of abstractions rather than a fixed stack of layers.

## Core idea

Learning is modeled as continuous evolution of abstractions under experience pressure. The system includes:

- perception encoder
- multi-level abstraction field
- dynamic concept population
- sparse concept router
- latent world model
- episodic memory
- fitness evaluation
- concept birth, mutation, merge, pruning, and consolidation

## Google Colab: direct run

Open a new Colab notebook, set **Runtime > Change runtime type > GPU**, then run:

```python
!git clone https://github.com/godsonj64/evolutionary-abstraction-network.git
%cd evolutionary-abstraction-network
!python colab_run_cifar_chunk.py
```

Fast debug run:

```python
!python colab_run_cifar_chunk.py --quick
```

Force CUDA:

```python
!python colab_run_cifar_chunk.py --device cuda
```

The launcher installs dependencies, installs the package in editable mode, selects CUDA when available, runs the chunked CIFAR-100 experiment, and saves metrics to:

```text
outputs/cifar100_chunk_metrics.csv
```

## Manual install

```bash
pip install -r requirements.txt
pip install -e .
```

## Run tests

```bash
python -m pytest -q
```

## CPU demo

```bash
python demo_train.py
```

## GPU-ready toy demo

```bash
python gpu_demo_train.py --steps 50 --device cuda
```

## Chunked CIFAR-100 experiment

```bash
python experiments/split_cifar100_chunk_train.py \
  --num-chunks 3 \
  --classes-per-chunk 5 \
  --train-samples-per-chunk 500 \
  --test-samples-per-chunk 200 \
  --epochs-per-chunk 1 \
  --batch-size 64 \
  --latent-dim 128 \
  --abstraction-dim 128 \
  --hidden-dim 256 \
  --initial-concepts 6 \
  --max-concepts 18 \
  --top-k 3 \
  --evolve-every 20 \
  --device cuda
```

## Research status

This is a tested prototype, not a frontier-scale model. The present version validates the architectural mechanism: dynamic concept modules, abstraction routing, episodic memory, and evolution operators. Next research stages should add semantic/procedural memory, continual-task benchmarks, real-world datasets, and larger pretrained encoders.

## Architecture

```text
Input
  ↓
MLP / Encoder
  ↓
Abstraction Field
  ↓
Concept Router
  ↓
Top-K Concept Modules
  ↓
Concept Aggregator
  ↓
Output Head + Latent World Model
  ↓
Evolution Controller
```

## Principle

> Learning is the continuous evolution of abstractions under experience pressure.
