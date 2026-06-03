from __future__ import annotations

"""Qwen-style staged training pipeline for a compact EAN chat model.

This is not a Qwen clone. It is a Colab-scale EAN language model trained with a
Qwen-inspired curriculum and a modern hybrid tokenizer path:

1. base causal language pretraining
2. supervised instruction tuning
3. chat-format fine-tuning

Tokenizer strategy:
- try a Qwen tokenizer first;
- fall back to GPT-style byte-level BPE if Qwen is unavailable;
- inject EAN/ChatML special tokens;
- optionally extend the tokenizer with frequent corpus tokens.

Run:
    python experiments/qwen_style_ean_chat_pipeline.py --device cuda --quick
"""

import argparse, json, math, random, re, sys
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
from transformers import AutoTokenizer

from ean import EANConfig, EvolutionaryAbstractionNetwork

EAN_SPECIALS = ["<|system|>", "<|user|>", "<|assistant|>", "<|think|>", "<|answer|>", "<|end|>"]
DOMAIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{3,}")


@dataclass(frozen=True)
class ChatEANConfig:
    vocab_size: int
    pad_id: int
    block_size: int = 128
    emb_dim: int = 256
    latent_dim: int = 256
    abstraction_dim: int = 256
    hidden_dim: int = 512
    initial_concepts: int = 12
    max_concepts: int = 32
    top_k: int = 4
    dropout: float = 0.10


class HybridEANTokenizer:
    """Modern tokenizer wrapper for EAN chat experiments.

    Qwen-family tokenizers are preferred because they already encode ChatML-style
    prompting well. GPT-2 byte-level BPE is used as a robust fallback because it
    has byte coverage and avoids catastrophic unknown-token behavior on arbitrary
    text. Extra EAN control tokens are injected for staged chat training.
    """

    def __init__(self, model_name: str, texts: list[str], extra_domain_tokens: int = 0):
        self.model_name = model_name
        self.tokenizer = self._load_tokenizer(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        special = {"additional_special_tokens": [t for t in EAN_SPECIALS if t not in self.tokenizer.get_vocab()]}
        self.tokenizer.add_special_tokens(special)
        if extra_domain_tokens > 0:
            self._extend_with_domain_tokens(texts, extra_domain_tokens)
        self.pad_id = int(self.tokenizer.pad_token_id)
        self.eos_id = int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else self.pad_id

    def _load_tokenizer(self, model_name: str):
        candidates = [model_name]
        if model_name != "Qwen/Qwen3-0.6B":
            candidates.append("Qwen/Qwen3-0.6B")
        candidates += ["Qwen/Qwen2.5-0.5B", "gpt2"]
        errors = []
        for name in candidates:
            try:
                tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=True)
                self.loaded_name = name
                return tok
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        raise RuntimeError("Could not load any tokenizer. Errors: " + " | ".join(errors))

    def _extend_with_domain_tokens(self, texts: list[str], limit: int) -> None:
        vocab = self.tokenizer.get_vocab()
        counts = Counter()
        for text in texts:
            counts.update(DOMAIN_TOKEN_RE.findall(text))
        added = []
        for token, freq in counts.most_common(limit * 5):
            if freq < 3:
                continue
            if token in vocab or token.lower() in vocab:
                continue
            # Only add reasonably semantic whole tokens. Byte/BPE handles the rest.
            if 4 <= len(token) <= 32:
                added.append(token)
            if len(added) >= limit:
                break
        if added:
            self.tokenizer.add_tokens(added)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def state(self) -> dict:
        return {"tokenizer_name": self.loaded_name, "pad_id": self.pad_id, "eos_id": self.eos_id, "vocab_size": self.vocab_size, "ean_specials": EAN_SPECIALS}


class LMDataset(Dataset):
    def __init__(self, texts: list[str], tok: HybridEANTokenizer, block_size: int):
        self.samples = []
        self.block_size = block_size
        self.pad_id = tok.pad_id
        for text in texts:
            ids = tok.encode(text, add_special_tokens=True)
            if len(ids) < 4:
                continue
            for start in range(0, max(1, len(ids) - 1), block_size):
                chunk = ids[start:start + block_size + 1]
                if len(chunk) >= 4:
                    self.samples.append(chunk)
        if not self.samples:
            raise ValueError("No valid LM samples were built.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx][:self.block_size + 1]
        ids = ids + [self.pad_id] * (self.block_size + 1 - len(ids))
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        y = y.masked_fill(y == self.pad_id, -100)
        return x, y


class QwenStyleEANLM(nn.Module):
    def __init__(self, cfg: ChatEANConfig):
        super().__init__(); self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_id)
        self.pos = nn.Embedding(cfg.block_size, cfg.emb_dim)
        self.evidence_norm = nn.LayerNorm(cfg.emb_dim)
        self.ean = EvolutionaryAbstractionNetwork(EANConfig(input_dim=cfg.emb_dim, output_dim=cfg.emb_dim, latent_dim=cfg.latent_dim, abstraction_dim=cfg.abstraction_dim, hidden_dim=cfg.hidden_dim, initial_concepts=cfg.initial_concepts, max_concepts=cfg.max_concepts, top_k=cfg.top_k))
        self.h_proj = nn.Linear(cfg.hidden_dim, cfg.emb_dim)
        self.z_proj = nn.Linear(cfg.latent_dim, cfg.emb_dim)
        layer = nn.TransformerEncoderLayer(cfg.emb_dim, nhead=8, dim_feedforward=4 * cfg.emb_dim, dropout=cfg.dropout, activation="gelu", batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=4)
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, ids: torch.Tensor, store_memory: bool = False) -> dict[str, torch.Tensor]:
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device).unsqueeze(0).expand(b, t)
        x0 = self.emb(ids) + self.pos(pos)
        mask_valid = ids != self.cfg.pad_id
        evidence = (x0 * mask_valid.float().unsqueeze(-1)).sum(1) / mask_valid.float().sum(1, keepdim=True).clamp_min(1.0)
        evidence = self.evidence_norm(evidence)
        e = self.ean(evidence, store_memory=store_memory)
        x = x0 + self.h_proj(e["hidden"]).unsqueeze(1) + self.z_proj(e["latent"]).unsqueeze(1)
        causal = torch.triu(torch.full((t, t), float("-inf"), device=ids.device), diagonal=1)
        h = self.blocks(self.drop(x), mask=causal, src_key_padding_mask=~mask_valid)
        return {**e, "logits": self.head(h)}

    @torch.no_grad()
    def concept_entropy(self) -> float:
        u = torch.tensor([max(float(getattr(c, "usage", torch.tensor(0.0)).detach().cpu()), 0.0) for c in self.ean.population])
        if u.sum() <= 0: return 0.0
        p = u / u.sum(); return float(-(p * p.clamp_min(1e-8).log()).sum())


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def load_pretrain(n: int) -> list[str]:
    try:
        ds = load_dataset("roneneldan/TinyStories", split=f"train[:{n}]")
        return [clean(r["text"]) for r in ds if clean(r["text"])]
    except Exception:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        return [clean(r["text"]) for r in ds if clean(r["text"])][:n]


def load_instruct(n: int) -> list[str]:
    try:
        ds = load_dataset("yahma/alpaca-cleaned", split=f"train[:{n}]")
    except Exception:
        ds = load_dataset("tatsu-lab/alpaca", split=f"train[:{n}]")
    rows = []
    for r in ds:
        instr = clean(r.get("instruction", "")); inp = clean(r.get("input", "")); out = clean(r.get("output", ""))
        if not instr or not out: continue
        user = instr if not inp else instr + "\n" + inp
        rows.append(f"<|user|> {user} <|assistant|> <|answer|> {out} <|end|>")
    return rows[:n]


def load_chat(n: int) -> list[str]:
    try:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{n}]")
        rows = []
        for r in ds:
            parts = []
            for m in r.get("messages", []):
                role = m.get("role", "user"); content = clean(m.get("content", ""))
                if content and role in {"user", "assistant", "system"}:
                    parts.append(f"<|{role}|> {content}")
            if parts: rows.append(" ".join(parts) + " <|end|>")
        return rows[:n]
    except Exception:
        return load_instruct(n)


def make_loader(texts, tok, block, batch, shuffle, device):
    return DataLoader(LMDataset(texts, tok, block), batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available() and str(device).startswith("cuda"))


def loss_fn(out, y, latent_weight):
    lm = F.cross_entropy(out["logits"].reshape(-1, out["logits"].size(-1)), y.reshape(-1), ignore_index=-100)
    lat = F.mse_loss(out["next_latent_prediction"], out["latent"].detach())
    return lm + latent_weight * lat, lm.detach(), lat.detach()


def train_stage(model, loader, opt, device, args, name):
    model.train(); total_lm = 0.0; total_tok = 0; events = {"born":0,"mutated":0,"merged":0,"pruned":0,"consolidated":0}
    for step, (x, y) in enumerate(tqdm(loader, desc=name, leave=False), 1):
        x = x.to(device); y = y.to(device)
        opt.zero_grad(set_to_none=True); out = model(x, store_memory=True)
        loss, lm, _ = loss_fn(out, y, args.latent_weight); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt.step()
        n = int((y != -100).sum().cpu()); total_tok += n; total_lm += float(lm.cpu()) * n
        if (not args.no_evolution) and step % args.evolve_every == 0:
            ev = model.ean.evolve_from_outputs(out, next_latent_target=out["latent"].detach())
            for k, v in ev.items(): events[k] = events.get(k, 0) + int(v)
            if ev.get("born",0) or ev.get("merged",0) or ev.get("pruned",0):
                opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ppl = math.exp(min(total_lm / max(total_tok, 1), 20.0))
    return {"ppl": ppl, "events": events, "concepts": len(model.ean.population), "entropy": model.concept_entropy()}, opt


@torch.no_grad()
def generate(model, tok, prompt, device, max_new=80, temperature=0.8):
    model.eval(); ids = tok.encode(prompt, add_special_tokens=True)[-model.cfg.block_size:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        inp = x[:, -model.cfg.block_size:]
        logits = model(inp)["logits"][:, -1, :] / max(temperature, 1e-5)
        logits[:, tok.pad_id] = -float("inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        x = torch.cat([x, nxt], 1)
        if int(nxt.item()) == tok.eos_id: break
    return tok.decode(x.squeeze(0).tolist())


def configure_evolution(model, args):
    ev = model.ean.evolution
    ev.min_concepts = min(args.min_concepts, len(model.ean.population))
    ev.merge_similarity_threshold = args.merge_threshold
    ev.birth_error_threshold = args.birth_error_threshold
    ev.novelty_threshold = args.novelty_threshold
    ev.min_age_before_merge = args.min_age_before_merge
    ev.min_age_before_prune = args.min_age_before_prune


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--quick", action="store_true")
    p.add_argument("--tokenizer-name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--extra-domain-tokens", type=int, default=256)
    p.add_argument("--block-size", type=int, default=128); p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--pretrain-epochs", type=int, default=2); p.add_argument("--instruct-epochs", type=int, default=2); p.add_argument("--chat-epochs", type=int, default=2)
    p.add_argument("--pretrain-samples", type=int, default=6000); p.add_argument("--instruct-samples", type=int, default=2500); p.add_argument("--chat-samples", type=int, default=2500)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--weight-decay", type=float, default=0.02); p.add_argument("--latent-weight", type=float, default=0.01); p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--evolve-every", type=int, default=100); p.add_argument("--min-concepts", type=int, default=10); p.add_argument("--merge-threshold", type=float, default=0.9999)
    p.add_argument("--birth-error-threshold", type=float, default=0.25); p.add_argument("--novelty-threshold", type=float, default=0.15); p.add_argument("--min-age-before-merge", type=int, default=100); p.add_argument("--min-age-before-prune", type=int, default=100); p.add_argument("--no-evolution", action="store_true")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--output-dir", default="outputs/qwen_style_ean_chat")
    args = p.parse_args(); set_seed(args.seed)
    if args.quick:
        args.pretrain_samples = min(args.pretrain_samples, 2000); args.instruct_samples = min(args.instruct_samples, 800); args.chat_samples = min(args.chat_samples, 800)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    pre = load_pretrain(args.pretrain_samples); inst = load_instruct(args.instruct_samples); chat = load_chat(args.chat_samples)
    tok = HybridEANTokenizer(args.tokenizer_name, pre[:2000] + inst + chat, extra_domain_tokens=args.extra_domain_tokens)
    cfg = ChatEANConfig(tok.vocab_size, tok.pad_id, args.block_size)
    model = QwenStyleEANLM(cfg).to(device); configure_evolution(model, args)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(json.dumps({"device":str(device), "tokenizer":tok.state(), "config":asdict(cfg), "samples":{"pretrain":len(pre),"instruct":len(inst),"chat":len(chat)}}, indent=2))

    history = []
    stages = [("pretrain", pre, args.pretrain_epochs), ("instruct", inst, args.instruct_epochs), ("chat", chat, args.chat_epochs)]
    for stage, texts, epochs in stages:
        loader = make_loader(texts, tok, args.block_size, args.batch_size, True, device)
        for ep in range(1, epochs + 1):
            metrics, opt = train_stage(model, loader, opt, device, args, f"{stage} {ep}/{epochs}")
            row = {"stage":stage, "epoch":ep, **metrics}; history.append(row); print(json.dumps(row))
        torch.save({"model_state_dict":model.state_dict(), "config":asdict(cfg), "tokenizer":tok.state(), "history":history}, outdir / f"ean_{stage}.pt")

    prompt = "<|system|> You are a helpful EAN assistant. <|user|> Explain how EAN differs from a Transformer. <|assistant|> <|answer|>"
    sample = generate(model, tok, prompt, device)
    (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (outdir / "sample_chat.txt").write_text(sample, encoding="utf-8")
    print(json.dumps({"final_concepts":len(model.ean.population), "final_entropy":model.concept_entropy(), "sample":sample}, indent=2))


if __name__ == "__main__":
    main()
