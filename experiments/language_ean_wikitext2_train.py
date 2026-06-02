from __future__ import annotations

"""Train a small Evolutionary Abstraction Network language model on WikiText-2.

This script is intentionally Colab-friendly. It uses a compact character-level
language modeling benchmark built from the public WikiText-2 raw split through
Hugging Face Datasets. The goal is not state-of-the-art perplexity; the goal is
a runnable, scientifically inspectable Language-EAN baseline with explicit
training, validation, next-latent prediction pressure, and concept-evolution logs.

Colab quickstart:
    !git clone https://github.com/godsonj64/evolutionary-abstraction-network.git
    %cd evolutionary-abstraction-network
    !pip install -q -r requirements_language.txt
    !python experiments/language_ean_wikitext2_train.py --device cuda --epochs 3 --quick
"""

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from datasets import load_dataset

from ean import EANConfig, EvolutionaryAbstractionNetwork


@dataclass(frozen=True)
class LanguageEANConfig:
    vocab_size: int
    block_size: int = 128
    embedding_dim: int = 128
    latent_dim: int = 128
    abstraction_dim: int = 128
    hidden_dim: int = 256
    initial_concepts: int = 8
    max_concepts: int = 24
    top_k: int = 3
    dropout: float = 0.10


class CharTokenizer:
    """Deterministic character tokenizer fitted only on training text."""

    pad_token = "<pad>"
    unk_token = "<unk>"

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.itos = [self.pad_token, self.unk_token] + chars
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.pad_id = self.stoi[self.pad_token]
        self.unk_id = self.stoi[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids if int(i) >= 2)

    def to_dict(self) -> dict[str, object]:
        return {"itos": self.itos, "pad_id": self.pad_id, "unk_id": self.unk_id}


class CharBlockDataset(Dataset):
    """Fixed-length next-character prediction windows."""

    def __init__(self, token_ids: list[int], block_size: int, stride: int | None = None, max_blocks: int | None = None):
        if len(token_ids) <= block_size + 1:
            raise ValueError("token sequence is too short for the requested block size")
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = int(block_size)
        self.stride = int(stride if stride is not None else block_size)
        starts = list(range(0, len(token_ids) - block_size - 1, self.stride))
        if max_blocks is not None:
            starts = starts[: int(max_blocks)]
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        x = self.token_ids[start : start + self.block_size]
        y = self.token_ids[start + 1 : start + self.block_size + 1]
        return x, y


class LanguageEAN(nn.Module):
    """Autoregressive character language model using EAN as the abstraction core.

    Sequence tokens are embedded, enriched with learned positions, summarized by a
    GRU encoder into a sequence-level evidence vector, and passed to the general
    EvolutionaryAbstractionNetwork. The EAN hidden state conditions a lightweight
    causal token decoder. This preserves the existing EAN core while adapting its
    output to next-token language modeling.
    """

    def __init__(self, config: LanguageEANConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.block_size, config.embedding_dim)
        self.sequence_encoder = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.embedding_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.ean_core = EvolutionaryAbstractionNetwork(
            EANConfig(
                input_dim=config.embedding_dim,
                output_dim=config.embedding_dim,
                latent_dim=config.latent_dim,
                abstraction_dim=config.abstraction_dim,
                hidden_dim=config.hidden_dim,
                initial_concepts=config.initial_concepts,
                max_concepts=config.max_concepts,
                top_k=config.top_k,
            )
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=4,
            dim_feedforward=4 * config.embedding_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.ean_to_decoder = nn.Linear(config.hidden_dim, config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, store_memory: bool = False) -> dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.config.block_size}")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        token_features = self.token_embedding(input_ids) + self.position_embedding(positions)
        token_features = self.dropout(token_features)

        _, h_last = self.sequence_encoder(token_features)
        evidence = h_last[-1]
        ean_outputs = self.ean_core(evidence, store_memory=store_memory)

        concept_context = self.ean_to_decoder(ean_outputs["hidden"]).unsqueeze(1)
        decoder_input = token_features + concept_context
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1,
        )
        decoded = self.decoder(decoder_input, mask=causal_mask)
        logits = self.lm_head(decoded)

        return {
            **ean_outputs,
            "logits": logits,
            "sequence_evidence": evidence,
        }

    @torch.no_grad()
    def concept_entropy(self) -> float:
        usage = []
        for concept in self.ean_core.population:
            value = float(getattr(concept, "usage", torch.tensor(0.0)).detach().cpu())
            usage.append(max(value, 0.0))
        if not usage or sum(usage) <= 0:
            return 0.0
        p = torch.tensor(usage, dtype=torch.float32)
        p = p / p.sum().clamp_min(1e-8)
        return float(-(p * p.clamp_min(1e-8).log()).sum().item())

    def evolve_from_outputs(self, outputs: dict[str, torch.Tensor], next_latent_target: torch.Tensor | None) -> dict[str, int]:
        return self.ean_core.evolve_from_outputs(outputs, next_latent_target=next_latent_target)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_wikitext2_text() -> tuple[str, str, str]:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def join_split(split: str) -> str:
        lines = [row["text"] for row in dataset[split] if row["text"] and row["text"].strip()]
        return "\n".join(lines)

    return join_split("train"), join_split("validation"), join_split("test")


def build_dataloaders(args: argparse.Namespace) -> tuple[CharTokenizer, DataLoader, DataLoader, DataLoader]:
    train_text, val_text, test_text = load_wikitext2_text()
    if args.quick:
        train_text = train_text[: args.quick_train_chars]
        val_text = val_text[: args.quick_eval_chars]
        test_text = test_text[: args.quick_eval_chars]

    tokenizer = CharTokenizer(train_text)
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    test_ids = tokenizer.encode(test_text)

    train_ds = CharBlockDataset(
        train_ids,
        block_size=args.block_size,
        stride=args.train_stride,
        max_blocks=args.max_train_blocks,
    )
    val_ds = CharBlockDataset(
        val_ids,
        block_size=args.block_size,
        stride=args.block_size,
        max_blocks=args.max_eval_blocks,
    )
    test_ds = CharBlockDataset(
        test_ids,
        block_size=args.block_size,
        stride=args.block_size,
        max_blocks=args.max_eval_blocks,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return tokenizer, train_loader, val_loader, test_loader


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))


def loss_from_outputs(outputs: dict[str, torch.Tensor], targets: torch.Tensor, latent_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["logits"]
    lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    latent_loss = F.mse_loss(outputs["next_latent_prediction"], outputs["latent"].detach())
    total = lm_loss + latent_weight * latent_loss
    return total, {
        "lm_loss": float(lm_loss.detach().cpu()),
        "latent_loss": float(latent_loss.detach().cpu()),
        "loss": float(total.detach().cpu()),
    }


def evaluate(model: LanguageEAN, loader: DataLoader, device: torch.device, latent_weight: float) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_lm = 0.0
    total_latent = 0.0
    total_tokens = 0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            outputs = model(x, store_memory=False)
            loss, pieces = loss_from_outputs(outputs, y, latent_weight)
            tokens = y.numel()
            total_tokens += tokens
            total_loss += pieces["loss"] * tokens
            total_lm += pieces["lm_loss"] * tokens
            total_latent += pieces["latent_loss"] * tokens
            pred = outputs["logits"].argmax(dim=-1)
            correct += int((pred == y).sum().detach().cpu())

    mean_loss = total_loss / max(total_tokens, 1)
    mean_lm = total_lm / max(total_tokens, 1)
    mean_latent = total_latent / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "lm_loss": mean_lm,
        "latent_loss": mean_latent,
        "perplexity": float(math.exp(min(mean_lm, 20.0))),
        "token_accuracy": correct / max(total_tokens, 1),
    }


def train_one_epoch(
    model: LanguageEAN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> tuple[dict[str, float], torch.optim.Optimizer, dict[str, int]]:
    model.train()
    total_loss = 0.0
    total_lm = 0.0
    total_latent = 0.0
    total_tokens = 0
    total_events = {"born": 0, "mutated": 0, "merged": 0, "pruned": 0, "consolidated": 0}

    pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False)
    for step, (x, y) in enumerate(pbar, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x, store_memory=True)
        loss, pieces = loss_from_outputs(outputs, y, args.latent_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.evolve_every == 0:
            events = model.evolve_from_outputs(outputs, next_latent_target=outputs["latent"].detach())
            for key, value in events.items():
                total_events[key] = total_events.get(key, 0) + int(value)
            if events.get("born", 0) or events.get("pruned", 0) or events.get("merged", 0):
                optimizer = make_optimizer(model, args)

        tokens = y.numel()
        total_tokens += tokens
        total_loss += pieces["loss"] * tokens
        total_lm += pieces["lm_loss"] * tokens
        total_latent += pieces["latent_loss"] * tokens
        pbar.set_postfix(
            loss=f"{total_loss / max(total_tokens, 1):.3f}",
            ppl=f"{math.exp(min(total_lm / max(total_tokens, 1), 20.0)):.2f}",
            concepts=len(model.ean_core.population),
        )

    metrics = {
        "loss": total_loss / max(total_tokens, 1),
        "lm_loss": total_lm / max(total_tokens, 1),
        "latent_loss": total_latent / max(total_tokens, 1),
        "perplexity": float(math.exp(min(total_lm / max(total_tokens, 1), 20.0))),
    }
    return metrics, optimizer, total_events


@torch.no_grad()
def generate_text(model: LanguageEAN, tokenizer: CharTokenizer, prompt: str, device: torch.device, max_new_chars: int = 256, temperature: float = 0.9) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.unk_id]
    context = torch.tensor(ids[-model.config.block_size :], dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_chars):
        if context.size(1) < model.config.block_size:
            pad = torch.full((1, model.config.block_size - context.size(1)), tokenizer.pad_id, dtype=torch.long, device=device)
            model_input = torch.cat([pad, context], dim=1)
            last_index = model.config.block_size - 1
        else:
            model_input = context[:, -model.config.block_size :]
            last_index = model.config.block_size - 1
        logits = model(model_input, store_memory=False)["logits"][:, last_index, :] / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_id], dim=1)
    return tokenizer.decode(context.squeeze(0).tolist())


def save_checkpoint(
    output_dir: Path,
    model: LanguageEAN,
    tokenizer: CharTokenizer,
    args: argparse.Namespace,
    epoch: int,
    val_metrics: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": asdict(model.config),
            "tokenizer": tokenizer.to_dict(),
            "args": vars(args),
            "val_metrics": val_metrics,
        },
        output_dir / "language_ean_best.pt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Language-EAN on WikiText-2 raw character LM.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--train-stride", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--abstraction-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--initial-concepts", type=int, default=8)
    parser.add_argument("--max-concepts", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--latent-weight", type=float, default=0.05)
    parser.add_argument("--evolve-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="Use a small subset for a fast Colab smoke test.")
    parser.add_argument("--quick-train-chars", type=int, default=450_000)
    parser.add_argument("--quick-eval-chars", type=int, default=80_000)
    parser.add_argument("--max-train-blocks", type=int, default=None)
    parser.add_argument("--max-eval-blocks", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/language_ean_wikitext2")
    parser.add_argument("--prompt", default="The meaning of language is")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    tokenizer, train_loader, val_loader, test_loader = build_dataloaders(args)
    model = LanguageEAN(
        LanguageEANConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            embedding_dim=args.embedding_dim,
            latent_dim=args.latent_dim,
            abstraction_dim=args.abstraction_dim,
            hidden_dim=args.hidden_dim,
            initial_concepts=args.initial_concepts,
            max_concepts=args.max_concepts,
            top_k=args.top_k,
            dropout=args.dropout,
        )
    ).to(device)
    optimizer = make_optimizer(model, args)

    print(json.dumps({
        "device": str(device),
        "vocab_size": tokenizer.vocab_size,
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "test_batches": len(test_loader),
        "config": asdict(model.config),
    }, indent=2))

    best_val = float("inf")
    history_path = output_dir / "metrics.csv"
    fieldnames = [
        "epoch", "split", "loss", "lm_loss", "latent_loss", "perplexity", "token_accuracy",
        "concepts", "concept_entropy", "born", "mutated", "merged", "pruned", "consolidated", "seconds"
    ]
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics, optimizer, events = train_one_epoch(model, train_loader, optimizer, device, args, epoch)
        val_metrics = evaluate(model, val_loader, device, args.latent_weight)
        elapsed = time.time() - start

        common = {
            "epoch": epoch,
            "concepts": len(model.ean_core.population),
            "concept_entropy": model.concept_entropy(),
            **events,
            "seconds": elapsed,
        }
        rows = [
            {"split": "train", "token_accuracy": "", **train_metrics, **common},
            {"split": "validation", **val_metrics, **common},
        ]
        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for row in rows:
                writer.writerow(row)

        print(
            f"epoch={epoch} train_ppl={train_metrics['perplexity']:.2f} "
            f"val_ppl={val_metrics['perplexity']:.2f} val_acc={val_metrics['token_accuracy']:.4f} "
            f"concepts={len(model.ean_core.population)} entropy={model.concept_entropy():.3f} events={events}"
        )
        if val_metrics["perplexity"] < best_val:
            best_val = val_metrics["perplexity"]
            save_checkpoint(output_dir, model, tokenizer, args, epoch, val_metrics)

    test_metrics = evaluate(model, test_loader, device, args.latent_weight)
    sample = generate_text(model, tokenizer, args.prompt, device=device)
    summary = {
        "best_validation_perplexity": best_val,
        "test_metrics": test_metrics,
        "final_concepts": len(model.ean_core.population),
        "final_concept_entropy": model.concept_entropy(),
        "sample_prompt": args.prompt,
        "sample_generation": sample,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "sample_generation.txt").write_text(sample, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
