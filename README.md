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

## Install

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

## GPU-ready demo

```bash
python gpu_demo_train.py --steps 50
```

Force CPU:

```bash
python gpu_demo_train.py --steps 10 --device cpu
```

Force CUDA:

```bash
python gpu_demo_train.py --steps 100 --device cuda
```

## Google Colab

Open `colab_train_ean.ipynb`, set **Runtime > Change runtime type > GPU**, then run the cells.

Minimal Colab commands:

```python
!git clone https://github.com/godsonj64/evolutionary-abstraction-network.git
%cd evolutionary-abstraction-network
!pip install -e .
!pip install -r requirements.txt
!pytest -q
!python gpu_demo_train.py --steps 50
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
