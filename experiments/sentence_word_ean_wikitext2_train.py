from __future__ import annotations

"""Word-level sentence Language-EAN on WikiText-2.

Recommended Colab sentence-level experiment. It reads one sentence and learns to
predict the next sentence using a compact word vocabulary built from the training
split, avoiding the 50k randomly initialized GPT-2 output space.

Run:
    python experiments/sentence_word_ean_wikitext2_train.py --device cuda --epochs 10 --quick
"""

import argparse, csv, json, math, random, re, sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from datasets import load_dataset

from ean import EANConfig, EvolutionaryAbstractionNetwork

SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
TOK_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]")


@dataclass(frozen=True)
class LMConfig:
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    context_len: int = 48
    target_len: int = 47
    emb_dim: int = 192
    latent_dim: int = 192
    abstraction_dim: int = 192
    hidden_dim: int = 384
    initial_concepts: int = 12
    max_concepts: int = 32
    top_k: int = 4
    dropout: float = 0.10


class WordTokenizer:
    PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"

    def __init__(self, sentences: list[str], vocab_size: int, min_freq: int = 1):
        counts = Counter()
        for s in sentences:
            counts.update(self.tokenize(s))
        special = [self.PAD, self.UNK, self.BOS, self.EOS]
        words = [w for w, c in counts.most_common(max(0, vocab_size - 4)) if c >= min_freq]
        self.itos = special + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id = self.stoi[self.PAD]
        self.unk_id = self.stoi[self.UNK]
        self.bos_id = self.stoi[self.BOS]
        self.eos_id = self.stoi[self.EOS]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [t.lower() for t in TOK_RE.findall(text)]

    def encode(self, text: str, length: int, bos: bool = False, eos: bool = False) -> tuple[list[int], list[int]]:
        ids = [self.stoi.get(t, self.unk_id) for t in self.tokenize(text)]
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        ids = ids[:length]
        mask = [1] * len(ids)
        ids += [self.pad_id] * (length - len(ids))
        mask += [0] * (length - len(mask))
        return ids, mask

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i in {self.pad_id, self.bos_id}:
                continue
            if i == self.eos_id:
                break
            out.append(self.itos[i] if 0 <= i < len(self.itos) else self.UNK)
        text = " ".join(out)
        return re.sub(r"\s+([.,!?;:])", r"\1", text).strip()

    def state(self) -> dict:
        return {"itos": self.itos, "pad_id": self.pad_id, "unk_id": self.unk_id, "bos_id": self.bos_id, "eos_id": self.eos_id}


def split_sentences(text: str) -> list[str]:
    sent = []
    for s in SPLIT_RE.split(text.replace("\n", " ")):
        s = re.sub(r"\s+", " ", s).strip()
        n = len(WordTokenizer.tokenize(s))
        if 6 <= n <= 70 and not s.startswith("="):
            sent.append(s)
    return sent


def load_sentences() -> tuple[list[str], list[str], list[str]]:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    def collect(split: str) -> list[str]:
        text = "\n".join(r["text"] for r in ds[split] if r["text"] and r["text"].strip())
        return split_sentences(text)
    return collect("train"), collect("validation"), collect("test")


class SentencePairs(Dataset):
    def __init__(self, sentences: list[str], tok: WordTokenizer, context_len: int, target_len: int, max_pairs: int | None = None):
        self.sentences, self.tok = sentences, tok
        self.context_len, self.target_len = context_len, target_len
        self.n = len(sentences) - 1 if max_pairs is None else min(len(sentences) - 1, max_pairs)
        if self.n <= 0:
            raise ValueError("Need at least two valid sentences.")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        c_ids, c_mask = self.tok.encode(self.sentences[idx], self.context_len, eos=True)
        y_ids, _ = self.tok.encode(self.sentences[idx + 1], self.target_len, bos=True, eos=True)
        return {"context_ids": torch.tensor(c_ids), "context_mask": torch.tensor(c_mask), "target_ids": torch.tensor(y_ids)}


class SentenceWordEAN(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_id)
        self.cpos = nn.Embedding(cfg.context_len, cfg.emb_dim)
        self.ypos = nn.Embedding(cfg.target_len, cfg.emb_dim)
        enc = nn.TransformerEncoderLayer(cfg.emb_dim, nhead=6, dim_feedforward=4 * cfg.emb_dim, dropout=cfg.dropout, activation="gelu", batch_first=True)
        self.context_encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.ean = EvolutionaryAbstractionNetwork(EANConfig(input_dim=cfg.emb_dim, output_dim=cfg.emb_dim, latent_dim=cfg.latent_dim, abstraction_dim=cfg.abstraction_dim, hidden_dim=cfg.hidden_dim, initial_concepts=cfg.initial_concepts, max_concepts=cfg.max_concepts, top_k=cfg.top_k))
        self.h_to_dec = nn.Linear(cfg.hidden_dim, cfg.emb_dim)
        self.z_to_dec = nn.Linear(cfg.latent_dim, cfg.emb_dim)
        dec = nn.TransformerEncoderLayer(cfg.emb_dim, nhead=6, dim_feedforward=4 * cfg.emb_dim, dropout=cfg.dropout, activation="gelu", batch_first=True)
        self.decoder = nn.TransformerEncoder(dec, num_layers=3)
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float().unsqueeze(-1)
        return (x * m).sum(1) / m.sum(1).clamp_min(1.0)

    def encode_context(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device).unsqueeze(0).expand(b, t)
        x = self.emb(ids) + self.cpos(pos)
        x = self.context_encoder(self.drop(x), src_key_padding_mask=(mask == 0))
        return self.masked_mean(x, mask)

    def forward(self, context_ids: torch.Tensor, context_mask: torch.Tensor, decoder_ids: torch.Tensor, store_memory: bool = False) -> dict[str, torch.Tensor]:
        b, t = decoder_ids.shape
        evidence = self.encode_context(context_ids, context_mask)
        e = self.ean(evidence, store_memory=store_memory)
        pos = torch.arange(t, device=decoder_ids.device).unsqueeze(0).expand(b, t)
        y = self.emb(decoder_ids) + self.ypos(pos) + self.h_to_dec(e["hidden"]).unsqueeze(1) + self.z_to_dec(e["latent"]).unsqueeze(1)
        mask = torch.triu(torch.full((t, t), float("-inf"), device=decoder_ids.device), diagonal=1)
        return {**e, "logits": self.head(self.decoder(self.drop(y), mask=mask))}

    @torch.no_grad()
    def concept_entropy(self) -> float:
        u = torch.tensor([max(float(getattr(c, "usage", torch.tensor(0.0)).detach().cpu()), 0.0) for c in self.ean.population])
        if u.sum() <= 0:
            return 0.0
        p = u / u.sum()
        return float(-(p * p.clamp_min(1e-8).log()).sum())

    def evolve(self, out: dict[str, torch.Tensor]) -> dict[str, int]:
        return self.ean.evolve_from_outputs(out, next_latent_target=out["latent"].detach())


def shift_targets(y: torch.Tensor, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = y[:, :-1].contiguous()
    labels = y[:, 1:].contiguous().masked_fill(y[:, 1:].contiguous() == pad_id, -100)
    return x, labels


def compute_loss(out: dict[str, torch.Tensor], labels: torch.Tensor, latent_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lm = F.cross_entropy(out["logits"].reshape(-1, out["logits"].size(-1)), labels.reshape(-1), ignore_index=-100)
    latent = F.mse_loss(out["next_latent_prediction"], out["latent"].detach())
    return lm + latent_weight * latent, lm.detach(), latent.detach()


def eval_model(model: SentenceWordEAN, loader: DataLoader, device: torch.device, pad_id: int, latent_weight: float) -> dict[str, float]:
    model.eval(); total_lm = total_lat = 0.0; correct = total = 0
    with torch.no_grad():
        for batch in loader:
            c = batch["context_ids"].to(device); m = batch["context_mask"].to(device); y = batch["target_ids"].to(device)
            x, labels = shift_targets(y, pad_id)
            out = model(c, m, x)
            _, lm, lat = compute_loss(out, labels, latent_weight)
            valid = labels != -100; pred = out["logits"].argmax(-1); n = int(valid.sum().cpu())
            correct += int(((pred == labels) & valid).sum().cpu()); total += n
            total_lm += float(lm.cpu()) * n; total_lat += float(lat.cpu()) * n
    mlm = total_lm / max(total, 1)
    return {"lm_loss": mlm, "latent_loss": total_lat / max(total, 1), "perplexity": math.exp(min(mlm, 20.0)), "token_accuracy": correct / max(total, 1)}


@torch.no_grad()
def generate(model: SentenceWordEAN, tok: WordTokenizer, prompt: str, device: torch.device, temperature: float = 0.85) -> str:
    model.eval()
    c, m = tok.encode(prompt, model.cfg.context_len, eos=True)
    c = torch.tensor([c], device=device); m = torch.tensor([m], device=device)
    y = torch.tensor([[tok.bos_id]], device=device)
    for _ in range(model.cfg.target_len - 1):
        out = model(c, m, y)
        logits = out["logits"][:, -1, :] / max(temperature, 1e-5)
        logits[:, tok.pad_id] = -float("inf"); logits[:, tok.bos_id] = -float("inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        y = torch.cat([y, nxt], 1)
        if int(nxt.item()) == tok.eos_id:
            break
    return tok.decode(y.squeeze(0).tolist())


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_loader(ds: Dataset, batch: int, shuffle: bool, device: str) -> DataLoader:
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available() and device.startswith("cuda"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=10); p.add_argument("--quick", action="store_true")
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--vocab-size", type=int, default=6000); p.add_argument("--context-length", type=int, default=48); p.add_argument("--target-length", type=int, default=48)
    p.add_argument("--lr", type=float, default=5e-4); p.add_argument("--latent-weight", type=float, default=0.01); p.add_argument("--evolve-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--output-dir", default="outputs/sentence_word_ean_wikitext2")
    p.add_argument("--prompt", default="The meaning of language is shaped by memory and prediction.")
    args = p.parse_args(); set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    tr, va, te = load_sentences()
    if args.quick:
        tr, va, te = tr[:9000], va[:1200], te[:1200]
    tok = WordTokenizer(tr, args.vocab_size)
    train = SentencePairs(tr, tok, args.context_length, args.target_length)
    val = SentencePairs(va, tok, args.context_length, args.target_length)
    test = SentencePairs(te, tok, args.context_length, args.target_length)
    train_loader = make_loader(train, args.batch_size, True, args.device)
    val_loader = make_loader(val, args.eval_batch_size, False, args.device)
    test_loader = make_loader(test, args.eval_batch_size, False, args.device)

    cfg = LMConfig(tok.vocab_size, tok.pad_id, tok.bos_id, tok.eos_id, args.context_length, args.target_length - 1)
    model = SentenceWordEAN(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    print(json.dumps({"device": str(device), "vocab_size": tok.vocab_size, "train_pairs": len(train), "val_pairs": len(val), "test_pairs": len(test), "config": asdict(cfg)}, indent=2))

    metrics_path = outdir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_ppl", "val_ppl", "val_acc", "concepts", "concept_entropy", "events"])
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); total_lm = 0.0; total_tok = 0; events_total = {"born": 0, "mutated": 0, "merged": 0, "pruned": 0, "consolidated": 0}
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}", leave=False), 1):
            c = batch["context_ids"].to(device); m = batch["context_mask"].to(device); y = batch["target_ids"].to(device)
            x, labels = shift_targets(y, tok.pad_id)
            opt.zero_grad(set_to_none=True); out = model(c, m, x, store_memory=True)
            loss, lm, _ = compute_loss(out, labels, args.latent_weight); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            valid = labels != -100; n = int(valid.sum().cpu()); total_tok += n; total_lm += float(lm.cpu()) * n
            if step % args.evolve_every == 0:
                ev = model.evolve(out)
                for k, v in ev.items(): events_total[k] = events_total.get(k, 0) + int(v)
                if ev.get("born", 0) or ev.get("merged", 0) or ev.get("pruned", 0):
                    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
        train_ppl = math.exp(min(total_lm / max(total_tok, 1), 20.0)); val_m = eval_model(model, val_loader, device, tok.pad_id, args.latent_weight)
        print(f"epoch={epoch} train_ppl={train_ppl:.2f} val_ppl={val_m['perplexity']:.2f} val_acc={val_m['token_accuracy']:.4f} concepts={len(model.ean.population)} entropy={model.concept_entropy():.3f} events={events_total}")
        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_ppl, val_m["perplexity"], val_m["token_accuracy"], len(model.ean.population), model.concept_entropy(), json.dumps(events_total)])
        if val_m["perplexity"] < best:
            best = val_m["perplexity"]
            torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg), "tokenizer": tok.state(), "val_metrics": val_m}, outdir / "sentence_word_ean_best.pt")
    test_m = eval_model(model, test_loader, device, tok.pad_id, args.latent_weight)
    sample = generate(model, tok, args.prompt, device)
    summary = {"best_validation_perplexity": best, "test_metrics": test_m, "final_concepts": len(model.ean.population), "final_concept_entropy": model.concept_entropy(), "prompt": args.prompt, "generated_next_sentence": sample}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "sample_generation.txt").write_text(f"PROMPT: {args.prompt}\nGENERATED: {sample}\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
