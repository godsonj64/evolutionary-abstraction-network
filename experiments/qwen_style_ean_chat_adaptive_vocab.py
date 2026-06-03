from __future__ import annotations

"""Adaptive-vocabulary Qwen-tokenized EAN chat pipeline.

This script keeps the modern Qwen/GPT byte-BPE tokenizer for segmentation, but
remaps only the corpus-observed token ids into a compact local vocabulary. This
is better for Colab-scale training than a randomly initialized 150k+ output head.

Pipeline:
    1. base LM pretraining
    2. instruction tuning
    3. chat-format fine-tuning

Run:
    python experiments/qwen_style_ean_chat_adaptive_vocab.py --device cuda --quick
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


class AdaptiveQwenTokenizer:
    """Qwen/GPT tokenizer segmentation with compact local id remapping."""

    def __init__(self, model_name: str, texts: list[str], max_local_vocab: int):
        self.base = self._load(model_name)
        if self.base.pad_token is None:
            self.base.pad_token = self.base.eos_token or self.base.unk_token
        self.base.add_special_tokens({"additional_special_tokens": [s for s in EAN_SPECIALS if s not in self.base.get_vocab()]})
        self.loaded_name = getattr(self, "loaded_name", model_name)
        special_ids = set()
        for tok in [self.base.pad_token, self.base.eos_token, self.base.bos_token, self.base.unk_token] + EAN_SPECIALS:
            if tok is not None:
                tid = self.base.convert_tokens_to_ids(tok)
                if tid is not None and tid != self.base.unk_token_id:
                    special_ids.add(int(tid))
        counts = Counter()
        for text in texts:
            counts.update(self.base.encode(text, add_special_tokens=True))
        ordered = list(special_ids)
        for tid, _ in counts.most_common(max_local_vocab * 2):
            tid = int(tid)
            if tid not in ordered:
                ordered.append(tid)
            if len(ordered) >= max_local_vocab:
                break
        self.global_to_local = {gid: i for i, gid in enumerate(ordered)}
        self.local_to_global = ordered
        self.unk_local = len(self.local_to_global)
        self.local_to_global.append(int(self.base.unk_token_id if self.base.unk_token_id is not None else self.base.eos_token_id))
        self.vocab_size = len(self.local_to_global)
        self.pad_id = self.global_to_local.get(int(self.base.pad_token_id), 0)
        self.eos_id = self.global_to_local.get(int(self.base.eos_token_id), self.pad_id)

    def _load(self, model_name: str):
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
        raise RuntimeError("Could not load tokenizer: " + " | ".join(errors))

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        gids = self.base.encode(text, add_special_tokens=add_special_tokens)
        return [self.global_to_local.get(int(g), self.unk_local) for g in gids]

    def decode(self, ids: list[int]) -> str:
        gids = []
        for lid in ids:
            lid = int(lid)
            if lid == self.pad_id:
                continue
            if 0 <= lid < len(self.local_to_global):
                gids.append(self.local_to_global[lid])
            if lid == self.eos_id:
                break
        return self.base.decode(gids, skip_special_tokens=False)

    def state(self) -> dict:
        return {"base_tokenizer": self.loaded_name, "adaptive_vocab_size": self.vocab_size, "pad_id": self.pad_id, "eos_id": self.eos_id, "ean_specials": EAN_SPECIALS}


class LMDataset(Dataset):
    def __init__(self, texts: list[str], tok: AdaptiveQwenTokenizer, block_size: int):
        self.samples = []
        self.pad_id = tok.pad_id
        self.block_size = block_size
        for text in texts:
            ids = tok.encode(text, add_special_tokens=True)
            if len(ids) < 4:
                continue
            for s in range(0, max(1, len(ids) - 1), block_size):
                chunk = ids[s:s + block_size + 1]
                if len(chunk) >= 4:
                    self.samples.append(chunk)
        if not self.samples:
            raise ValueError("No valid samples built.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx][:self.block_size + 1]
        ids = ids + [self.pad_id] * (self.block_size + 1 - len(ids))
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long).masked_fill(torch.tensor(ids[1:], dtype=torch.long) == self.pad_id, -100)
        return x, y


class EANChatLM(nn.Module):
    def __init__(self, cfg: ChatEANConfig):
        super().__init__(); self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_id)
        self.pos = nn.Embedding(cfg.block_size, cfg.emb_dim)
        self.norm = nn.LayerNorm(cfg.emb_dim)
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
        p = torch.arange(t, device=ids.device).unsqueeze(0).expand(b, t)
        x0 = self.emb(ids) + self.pos(p)
        valid = ids != self.cfg.pad_id
        evidence = (x0 * valid.float().unsqueeze(-1)).sum(1) / valid.float().sum(1, keepdim=True).clamp_min(1.0)
        e = self.ean(self.norm(evidence), store_memory=store_memory)
        x = x0 + self.h_proj(e["hidden"]).unsqueeze(1) + self.z_proj(e["latent"]).unsqueeze(1)
        causal = torch.triu(torch.full((t, t), float("-inf"), device=ids.device), diagonal=1)
        h = self.blocks(self.drop(x), mask=causal, src_key_padding_mask=~valid)
        return {**e, "logits": self.head(h)}

    @torch.no_grad()
    def concept_entropy(self) -> float:
        u = torch.tensor([max(float(getattr(c, "usage", torch.tensor(0.0)).detach().cpu()), 0.0) for c in self.ean.population])
        if u.sum() <= 0: return 0.0
        p = u / u.sum(); return float(-(p * p.clamp_min(1e-8).log()).sum())


def clean(x: str) -> str: return re.sub(r"\s+", " ", str(x)).strip()
def set_seed(s: int): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_pretrain(n):
    try:
        ds = load_dataset("roneneldan/TinyStories", split=f"train[:{n}]")
        return [clean(r["text"]) for r in ds if clean(r["text"])]
    except Exception:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        return [clean(r["text"]) for r in ds if clean(r["text"])][:n]


def load_instruct(n):
    try: ds = load_dataset("yahma/alpaca-cleaned", split=f"train[:{n}]")
    except Exception: ds = load_dataset("tatsu-lab/alpaca", split=f"train[:{n}]")
    rows = []
    for r in ds:
        inst, inp, out = clean(r.get("instruction", "")), clean(r.get("input", "")), clean(r.get("output", ""))
        if inst and out:
            rows.append(f"<|user|> {inst + (' ' + inp if inp else '')} <|assistant|> <|answer|> {out} <|end|>")
    return rows[:n]


def load_chat(n):
    try:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{n}]")
        rows = []
        for r in ds:
            parts = []
            for m in r.get("messages", []):
                role, content = m.get("role", "user"), clean(m.get("content", ""))
                if role in {"system", "user", "assistant"} and content:
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
    return lm + latent_weight * lat, lm.detach()


def train_stage(model, loader, opt, device, args, name):
    model.train(); total_lm = 0.0; total_tok = 0; events = {"born":0,"mutated":0,"merged":0,"pruned":0,"consolidated":0}
    for step, (x, y) in enumerate(tqdm(loader, desc=name, leave=False), 1):
        x = x.to(device); y = y.to(device)
        opt.zero_grad(set_to_none=True); out = model(x, store_memory=True)
        loss, lm = loss_fn(out, y, args.latent_weight); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt.step()
        n = int((y != -100).sum().cpu()); total_tok += n; total_lm += float(lm.cpu()) * n
        if (not args.no_evolution) and step % args.evolve_every == 0:
            ev = model.ean.evolve_from_outputs(out, next_latent_target=out["latent"].detach())
            for k, v in ev.items(): events[k] = events.get(k, 0) + int(v)
            if ev.get("born",0) or ev.get("merged",0) or ev.get("pruned",0):
                opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return {"ppl": math.exp(min(total_lm / max(total_tok, 1), 20.0)), "events": events, "concepts": len(model.ean.population), "entropy": model.concept_entropy()}, opt


def configure_evolution(model, args):
    ev = model.ean.evolution
    ev.min_concepts = min(args.min_concepts, len(model.ean.population))
    ev.merge_similarity_threshold = args.merge_threshold
    ev.birth_error_threshold = args.birth_error_threshold
    ev.novelty_threshold = args.novelty_threshold
    ev.min_age_before_merge = args.min_age_before_merge
    ev.min_age_before_prune = args.min_age_before_prune


@torch.no_grad()
def generate(model, tok, prompt, device, max_new=100, temperature=0.75, top_k=50, repetition_penalty=1.15):
    model.eval(); ids = tok.encode(prompt, add_special_tokens=True)[-model.cfg.block_size:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        logits = model(x[:, -model.cfg.block_size:])["logits"][:, -1, :] / max(temperature, 1e-5)
        logits[:, tok.pad_id] = -float("inf")
        recent = set(int(i) for i in x[0, -64:].tolist())
        for rid in recent:
            if 0 <= rid < logits.size(-1): logits[:, rid] /= repetition_penalty
        if top_k > 0:
            vals, idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            mask = torch.full_like(logits, -float("inf")); logits = mask.scatter(1, idx, vals)
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        x = torch.cat([x, nxt], 1)
        if int(nxt.item()) == tok.eos_id: break
    return tok.decode(x.squeeze(0).tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--quick", action="store_true")
    p.add_argument("--tokenizer-name", default="Qwen/Qwen3-0.6B"); p.add_argument("--adaptive-vocab-size", type=int, default=16000)
    p.add_argument("--block-size", type=int, default=128); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pretrain-epochs", type=int, default=3); p.add_argument("--instruct-epochs", type=int, default=3); p.add_argument("--chat-epochs", type=int, default=3)
    p.add_argument("--pretrain-samples", type=int, default=6000); p.add_argument("--instruct-samples", type=int, default=2500); p.add_argument("--chat-samples", type=int, default=2500)
    p.add_argument("--lr", type=float, default=4e-4); p.add_argument("--weight-decay", type=float, default=0.02); p.add_argument("--latent-weight", type=float, default=0.01); p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--evolve-every", type=int, default=100); p.add_argument("--min-concepts", type=int, default=10); p.add_argument("--merge-threshold", type=float, default=0.9999)
    p.add_argument("--birth-error-threshold", type=float, default=0.25); p.add_argument("--novelty-threshold", type=float, default=0.15); p.add_argument("--min-age-before-merge", type=int, default=100); p.add_argument("--min-age-before-prune", type=int, default=100); p.add_argument("--no-evolution", action="store_true")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--output-dir", default="outputs/qwen_style_ean_chat_adaptive_vocab")
    args = p.parse_args(); set_seed(args.seed)
    if args.quick:
        args.pretrain_samples = min(args.pretrain_samples, 2000); args.instruct_samples = min(args.instruct_samples, 800); args.chat_samples = min(args.chat_samples, 800)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    pre, inst, chat = load_pretrain(args.pretrain_samples), load_instruct(args.instruct_samples), load_chat(args.chat_samples)
    tok = AdaptiveQwenTokenizer(args.tokenizer_name, pre + inst + chat, max_local_vocab=args.adaptive_vocab_size)
    cfg = ChatEANConfig(tok.vocab_size, tok.pad_id, args.block_size)
    model = EANChatLM(cfg).to(device); configure_evolution(model, args)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(json.dumps({"device": str(device), "tokenizer": tok.state(), "config": asdict(cfg), "samples": {"pretrain": len(pre), "instruct": len(inst), "chat": len(chat)}}, indent=2))
    hist = []
    for stage, texts, epochs in [("pretrain", pre, args.pretrain_epochs), ("instruct", inst, args.instruct_epochs), ("chat", chat, args.chat_epochs)]:
        loader = make_loader(texts, tok, args.block_size, args.batch_size, True, device)
        for ep in range(1, epochs + 1):
            metrics, opt = train_stage(model, loader, opt, device, args, f"{stage} {ep}/{epochs}")
            row = {"stage": stage, "epoch": ep, **metrics}; hist.append(row); print(json.dumps(row))
        torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg), "tokenizer": tok.state(), "history": hist}, outdir / f"ean_{stage}.pt")
    prompt = "<|system|> You are a helpful EAN assistant. <|user|> Explain how EAN differs from a Transformer. <|assistant|> <|answer|>"
    sample = generate(model, tok, prompt, device)
    (outdir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    (outdir / "sample_chat.txt").write_text(sample, encoding="utf-8")
    print(json.dumps({"final_concepts": len(model.ean.population), "final_entropy": model.concept_entropy(), "sample": sample}, indent=2))


if __name__ == "__main__": main()
