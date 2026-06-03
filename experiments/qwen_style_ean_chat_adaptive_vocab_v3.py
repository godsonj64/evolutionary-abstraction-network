from __future__ import annotations

"""Adaptive-vocabulary Qwen-tokenized EAN chat pipeline v3.

This version reuses the v2 training stack but adds safer generation controls:
- configurable min/max new tokens;
- EOS suppression before the minimum length;
- configurable temperature, top-k, and repetition penalty;
- optional prompt override.

Run:
    python experiments/qwen_style_ean_chat_adaptive_vocab_v3.py --device cuda --quick
"""

import argparse, json, sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.qwen_style_ean_chat_adaptive_vocab_v2 import (
    AdaptiveQwenTokenizerV2,
    ChatEANConfig,
    EANChatLM,
    clean,
    configure_evolution,
    load_chat,
    load_instruct,
    load_pretrain,
    make_loader,
    set_seed,
    train_stage,
)


@torch.no_grad()
def generate_v3(
    model,
    tok,
    prompt: str,
    device,
    min_new_tokens: int = 24,
    max_new_tokens: int = 128,
    temperature: float = 0.85,
    top_k: int = 80,
    repetition_penalty: float = 1.35,
) -> str:
    model.eval()
    ids = tok.encode(prompt, append_eos=False)[-model.cfg.block_size:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for step in range(max_new_tokens):
        logits = model(x[:, -model.cfg.block_size:])["logits"][:, -1, :] / max(temperature, 1e-5)
        logits[:, tok.pad_id] = -float("inf")
        for bid in tok.banned_generation_ids:
            if 0 <= bid < logits.size(-1):
                logits[:, bid] = -float("inf")
        if step < min_new_tokens and 0 <= tok.eos_id < logits.size(-1):
            logits[:, tok.eos_id] = -float("inf")
        for rid in set(int(i) for i in x[0, -96:].tolist()):
            if 0 <= rid < logits.size(-1):
                logits[:, rid] /= repetition_penalty
        if top_k > 0:
            vals, idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            mask = torch.full_like(logits, -float("inf"))
            logits = mask.scatter(1, idx, vals)
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        x = torch.cat([x, nxt], dim=1)
        if step >= min_new_tokens and int(nxt.item()) == tok.eos_id:
            break
    return tok.decode(x.squeeze(0).tolist()[len(ids):], skip_control=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--tokenizer-name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--adaptive-vocab-size", type=int, default=8000)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pretrain-epochs", type=int, default=8)
    p.add_argument("--instruct-epochs", type=int, default=8)
    p.add_argument("--chat-epochs", type=int, default=8)
    p.add_argument("--pretrain-samples", type=int, default=6000)
    p.add_argument("--instruct-samples", type=int, default=2500)
    p.add_argument("--chat-samples", type=int, default=2500)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--latent-weight", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--evolve-every", type=int, default=100)
    p.add_argument("--min-concepts", type=int, default=10)
    p.add_argument("--merge-threshold", type=float, default=0.9999)
    p.add_argument("--birth-error-threshold", type=float, default=0.25)
    p.add_argument("--novelty-threshold", type=float, default=0.15)
    p.add_argument("--min-age-before-merge", type=int, default=100)
    p.add_argument("--min-age-before-prune", type=int, default=100)
    p.add_argument("--no-evolution", action="store_true")
    p.add_argument("--min-new-tokens", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=140)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=80)
    p.add_argument("--repetition-penalty", type=float, default=1.35)
    p.add_argument("--prompt", default="<|system|> You are a helpful EAN assistant. <|user|> Explain how EAN differs from a Transformer. <|assistant|> <|answer|>")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/qwen_style_ean_chat_adaptive_vocab_v3")
    args = p.parse_args()
    set_seed(args.seed)

    if args.quick:
        args.pretrain_samples = min(args.pretrain_samples, 2000)
        args.instruct_samples = min(args.instruct_samples, 800)
        args.chat_samples = min(args.chat_samples, 800)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    pre = load_pretrain(args.pretrain_samples)
    inst = load_instruct(args.instruct_samples)
    chat = load_chat(args.chat_samples)
    tok = AdaptiveQwenTokenizerV2(args.tokenizer_name, pre + inst + chat, max_local_vocab=args.adaptive_vocab_size)
    cfg = ChatEANConfig(tok.vocab_size, tok.pad_id, args.block_size)
    model = EANChatLM(cfg).to(device)
    configure_evolution(model, args)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(json.dumps({
        "device": str(device),
        "tokenizer": tok.state(),
        "config": asdict(cfg),
        "samples": {"pretrain": len(pre), "instruct": len(inst), "chat": len(chat)},
        "generation": {
            "min_new_tokens": args.min_new_tokens,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
        },
    }, indent=2))

    history = []
    for stage, texts, epochs in [("pretrain", pre, args.pretrain_epochs), ("instruct", inst, args.instruct_epochs), ("chat", chat, args.chat_epochs)]:
        loader = make_loader(texts, tok, args.block_size, args.batch_size, True, device)
        for ep in range(1, epochs + 1):
            metrics, opt = train_stage(model, loader, opt, device, args, f"{stage} {ep}/{epochs}")
            row = {"stage": stage, "epoch": ep, **metrics}
            history.append(row)
            print(json.dumps(row))
        torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg), "tokenizer": tok.state(), "history": history}, outdir / f"ean_{stage}.pt")

    sample = generate_v3(
        model,
        tok,
        args.prompt,
        device,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (outdir / "sample_chat.txt").write_text(sample, encoding="utf-8")
    print(json.dumps({"final_concepts": len(model.ean.population), "final_entropy": model.concept_entropy(), "sample": sample}, indent=2))


if __name__ == "__main__":
    main()
