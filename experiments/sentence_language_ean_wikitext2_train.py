from __future__ import annotations

"""Sentence-level Language-EAN training on WikiText-2.

This experiment upgrades the character-level smoke test into a sentence-pair
language modeling task. The model reads one sentence and learns to generate the
next sentence. This is a better fit for EAN because the abstraction field and
concept population operate over sentence-level semantic evidence rather than raw
character transitions.

Colab quickstart:
    !git clone https://github.com/godsonj64/evolutionary-abstraction-network.git
    %cd evolutionary-abstraction-network
    !pip install -q -r requirements_language.txt
    !python experiments/sentence_language_ean_wikitext2_train.py --device cuda --epochs 3 --quick
"""

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer

from ean import EANConfig, EvolutionaryAbstractionNetwork


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class SentenceLanguageEANConfig:
    vocab_size: int
    pad_token_id: int
    context_length: int = 64
    target_length: int = 64
    embedding_dim: int = 256
    latent_dim: int = 256
    abstraction_dim: int = 256
    hidden_dim: int = 512
    initial_concepts: int = 12
    max_concepts: int = 32
    top_k: int = 4
    dropout: float = 0.10


class SentencePairDataset(Dataset):
    """Dataset of adjacent sentence pairs: input sentence -> next sentence."""

    def __init__(
        self,
        sentences: list[str],
        tokenizer: AutoTokenizer,
        context_length: int,
        target_length: int,
        max_pairs: int | None = None,
    ):
        if len(sentences) < 2:
            raise ValueError("At least two sentences are required.")
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.context_length = int(context_length)
        self.target_length = int(target_length)
        self.max_pairs = max_pairs
        self.n_pairs = len(sentences) - 1 if max_pairs is None else min(len(sentences) - 1, int(max_pairs))

    def __len__(self) -> int:
        return self.n_pairs

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        src = self.sentences[idx]
        tgt = self.sentences[idx + 1]
        src_enc = self.tokenizer(
            src,
            max_length=self.context_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tgt_enc = self.tokenizer(
            tgt,
            max_length=self.target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "context_ids": src_enc["input_ids"].squeeze(0).long(),
            "context_mask": src_enc["attention_mask"].squeeze(0).long(),
            "target_ids": tgt_enc["input_ids"].squeeze(0).long(),
            "target_mask": tgt_enc["attention_mask"].squeeze(0).long(),
        }


class SentenceLanguageEAN(nn.Module):
    """Sentence-conditioned next-sentence generator with an EAN abstraction core."""

    def __init__(self, config: SentenceLanguageEANConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_token_id)
        self.context_position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.target_position_embedding = nn.Embedding(config.target_length, config.embedding_dim)
        self.context_encoder = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.embedding_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.context_projection = nn.Linear(2 * config.embedding_dim, config.embedding_dim)
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
            nhead=8,
            dim_feedforward=4 * config.embedding_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=3)
        self.ean_to_decoder = nn.Linear(config.hidden_dim, config.embedding_dim)
        self.latent_to_decoder = nn.Linear(config.latent_dim, config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

    def encode_context(self, context_ids: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = context_ids.shape
        pos = torch.arange(seq_len, device=context_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(context_ids) + self.context_position_embedding(pos)
        x = self.dropout(x)
        encoded, _ = self.context_encoder(x)
        pooled = self.masked_mean(encoded, context_mask)
        return self.context_projection(pooled)

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        store_memory: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch_size, target_len = decoder_input_ids.shape
        if target_len > self.config.target_length:
            raise ValueError(f"target length {target_len} exceeds target_length {self.config.target_length}")
        evidence = self.encode_context(context_ids, context_mask)
        ean_outputs = self.ean_core(evidence, store_memory=store_memory)

        pos = torch.arange(target_len, device=decoder_input_ids.device).unsqueeze(0).expand(batch_size, target_len)
        y = self.token_embedding(decoder_input_ids) + self.target_position_embedding(pos)
        concept_context = self.ean_to_decoder(ean_outputs["hidden"]).unsqueeze(1)
        latent_context = self.latent_to_decoder(ean_outputs["latent"]).unsqueeze(1)
        decoder_input = self.dropout(y + concept_context + latent_context)
        causal_mask = torch.triu(
            torch.full((target_len, target_len), float("-inf"), device=decoder_input_ids.device),
            diagonal=1,
        )
        decoded = self.decoder(decoder_input, mask=causal_mask)
        logits = self.lm_head(decoded)
        return {
            **ean_outputs,
            "logits": logits,
            "context_evidence": evidence,
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
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def split_sentences(text: str) -> list[str]:
    raw = _SENTENCE_SPLIT_RE.split(text.replace("\n", " "))
    cleaned = []
    for sentence in raw:
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if 20 <= len(sentence) <= 500 and not sentence.startswith("="):
            cleaned.append(sentence)
    return cleaned


def load_wikitext2_sentences() -> tuple[list[str], list[str], list[str]]:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def collect(split: str) -> list[str]:
        text = "\n".join(row["text"] for row in dataset[split] if row["text"] and row["text"].strip())
        return split_sentences(text)

    return collect("train"), collect("validation"), collect("test")


def build_dataloaders(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_sentences, val_sentences, test_sentences = load_wikitext2_sentences()
    if args.quick:
        train_sentences = train_sentences[: args.quick_train_sentences]
        val_sentences = val_sentences[: args.quick_eval_sentences]
        test_sentences = test_sentences[: args.quick_eval_sentences]

    train_ds = SentencePairDataset(train_sentences, tokenizer, args.context_length, args.target_length, args.max_train_pairs)
    val_ds = SentencePairDataset(val_sentences, tokenizer, args.context_length, args.target_length, args.max_eval_pairs)
    test_ds = SentencePairDataset(test_sentences, tokenizer, args.context_length, args.target_length, args.max_eval_pairs)
    pin_memory = torch.cuda.is_available() and args.device.startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    return tokenizer, train_loader, val_loader, test_loader


def shift_targets(target_ids: torch.Tensor, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    decoder_input_ids = target_ids[:, :-1].contiguous()
    labels = target_ids[:, 1:].contiguous()
    labels = labels.masked_fill(labels == pad_token_id, -100)
    return decoder_input_ids, labels


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))


def loss_from_outputs(outputs: dict[str, torch.Tensor], labels: torch.Tensor, latent_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["logits"]
    lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
    latent_loss = F.mse_loss(outputs["next_latent_prediction"], outputs["latent"].detach())
    total = lm_loss + latent_weight * latent_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "lm_loss": float(lm_loss.detach().cpu()),
        "latent_loss": float(latent_loss.detach().cpu()),
    }


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    pred = logits.argmax(dim=-1)
    valid = labels != -100
    correct = int(((pred == labels) & valid).sum().detach().cpu())
    total = int(valid.sum().detach().cpu())
    return correct, total


def evaluate(model: SentenceLanguageEAN, loader: DataLoader, device: torch.device, pad_token_id: int, latent_weight: float) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_lm = 0.0
    total_latent = 0.0
    correct = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            context_ids = batch["context_ids"].to(device, non_blocking=True)
            context_mask = batch["context_mask"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)
            decoder_input_ids, labels = shift_targets(target_ids, pad_token_id)
            outputs = model(context_ids, context_mask, decoder_input_ids, store_memory=False)
            _, pieces = loss_from_outputs(outputs, labels, latent_weight)
            batch_correct, batch_total = token_accuracy(outputs["logits"], labels)
            correct += batch_correct
            total_tokens += batch_total
            total_loss += pieces["loss"] * batch_total
            total_lm += pieces["lm_loss"] * batch_total
            total_latent += pieces["latent_loss"] * batch_total
    mean_lm = total_lm / max(total_tokens, 1)
    return {
        "loss": total_loss / max(total_tokens, 1),
        "lm_loss": mean_lm,
        "latent_loss": total_latent / max(total_tokens, 1),
        "perplexity": float(math.exp(min(mean_lm, 20.0))),
        "token_accuracy": correct / max(total_tokens, 1),
    }


def train_one_epoch(
    model: SentenceLanguageEAN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_token_id: int,
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
    for step, batch in enumerate(pbar, start=1):
        context_ids = batch["context_ids"].to(device, non_blocking=True)
        context_mask = batch["context_mask"].to(device, non_blocking=True)
        target_ids = batch["target_ids"].to(device, non_blocking=True)
        decoder_input_ids, labels = shift_targets(target_ids, pad_token_id)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(context_ids, context_mask, decoder_input_ids, store_memory=True)
        loss, pieces = loss_from_outputs(outputs, labels, args.latent_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.evolve_every == 0:
            events = model.evolve_from_outputs(outputs, next_latent_target=outputs["latent"].detach())
            for key, value in events.items():
                total_events[key] = total_events.get(key, 0) + int(value)
            if events.get("born", 0) or events.get("pruned", 0) or events.get("merged", 0):
                optimizer = make_optimizer(model, args)

        _, batch_tokens = token_accuracy(outputs["logits"], labels)
        total_tokens += batch_tokens
        total_loss += pieces["loss"] * batch_tokens
        total_lm += pieces["lm_loss"] * batch_tokens
        total_latent += pieces["latent_loss"] * batch_tokens
        mean_lm = total_lm / max(total_tokens, 1)
        pbar.set_postfix(
            loss=f"{total_loss / max(total_tokens, 1):.3f}",
            ppl=f"{math.exp(min(mean_lm, 20.0)):.2f}",
            concepts=len(model.ean_core.population),
        )

    mean_lm = total_lm / max(total_tokens, 1)
    return {
        "loss": total_loss / max(total_tokens, 1),
        "lm_loss": mean_lm,
        "latent_loss": total_latent / max(total_tokens, 1),
        "perplexity": float(math.exp(min(mean_lm, 20.0))),
    }, optimizer, total_events


@torch.no_grad()
def generate_next_sentence(
    model: SentenceLanguageEAN,
    tokenizer: AutoTokenizer,
    sentence: str,
    device: torch.device,
    context_length: int,
    target_length: int,
    temperature: float = 0.9,
) -> str:
    model.eval()
    enc = tokenizer(sentence, max_length=context_length, padding="max_length", truncation=True, return_tensors="pt")
    context_ids = enc["input_ids"].to(device)
    context_mask = enc["attention_mask"].to(device)
    generated = torch.tensor([[tokenizer.bos_token_id or tokenizer.eos_token_id]], dtype=torch.long, device=device)
    for _ in range(target_length - 1):
        if generated.size(1) >= target_length:
            break
        outputs = model(context_ids, context_mask, generated, store_memory=False)
        logits = outputs["logits"][:, -1, :] / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_id], dim=1)
        if int(next_id.item()) == int(tokenizer.eos_token_id):
            break
    return tokenizer.decode(generated.squeeze(0), skip_special_tokens=True).strip()


def save_checkpoint(output_dir: Path, model: SentenceLanguageEAN, tokenizer: AutoTokenizer, args: argparse.Namespace, epoch: int, val_metrics: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": asdict(model.config),
            "tokenizer_name": args.tokenizer_name,
            "pad_token_id": tokenizer.pad_token_id,
            "args": vars(args),
            "val_metrics": val_metrics,
        },
        output_dir / "sentence_language_ean_best.pt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sentence-level Language-EAN on adjacent WikiText-2 sentence pairs.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer-name", default="gpt2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--target-length", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--abstraction-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--initial-concepts", type=int, default=12)
    parser.add_argument("--max-concepts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--latent-weight", type=float, default=0.02)
    parser.add_argument("--evolve-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-train-sentences", type=int, default=6000)
    parser.add_argument("--quick-eval-sentences", type=int, default=800)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-eval-pairs", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/sentence_language_ean_wikitext2")
    parser.add_argument("--prompt", default="The meaning of language is shaped by memory and prediction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    tokenizer, train_loader, val_loader, test_loader = build_dataloaders(args)

    model = SentenceLanguageEAN(
        SentenceLanguageEANConfig(
            vocab_size=len(tokenizer),
            pad_token_id=int(tokenizer.pad_token_id),
            context_length=args.context_length,
            target_length=args.target_length - 1,
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
        "tokenizer": args.tokenizer_name,
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "test_batches": len(test_loader),
        "config": asdict(model.config),
    }, indent=2))

    history_path = output_dir / "metrics.csv"
    fieldnames = [
        "epoch", "split", "loss", "lm_loss", "latent_loss", "perplexity", "token_accuracy",
        "concepts", "concept_entropy", "born", "mutated", "merged", "pruned", "consolidated", "seconds"
    ]
    with history_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics, optimizer, events = train_one_epoch(model, train_loader, optimizer, device, tokenizer.pad_token_id, args, epoch)
        val_metrics = evaluate(model, val_loader, device, tokenizer.pad_token_id, args.latent_weight)
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

    test_metrics = evaluate(model, test_loader, device, tokenizer.pad_token_id, args.latent_weight)
    generated = generate_next_sentence(model, tokenizer, args.prompt, device, args.context_length, args.target_length)
    summary = {
        "best_validation_perplexity": best_val,
        "test_metrics": test_metrics,
        "final_concepts": len(model.ean_core.population),
        "final_concept_entropy": model.concept_entropy(),
        "prompt_sentence": args.prompt,
        "generated_next_sentence": generated,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "sample_generation.txt").write_text(f"PROMPT: {args.prompt}\nGENERATED: {generated}\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
