from __future__ import annotations

"""Unsloth Qwen 4B chat SFT pipeline.

This script fine-tunes a real pretrained Qwen 4B base model with QLoRA using
Unsloth. It is intended for Colab GPUs and replaces the scratch chat-model path
when the goal is a usable chat-like model.

Recommended Colab command:
    python experiments/unsloth_qwen4b_chat_sft.py \
      --model-name unsloth/Qwen3-4B-Base-bnb-4bit \
      --output-dir outputs/qwen4b-ean-chat-sft \
      --ultrachat-samples 12000 \
      --alpaca-samples 5000 \
      --max-seq-length 2048 \
      --per-device-train-batch-size 2 \
      --gradient-accumulation-steps 8 \
      --max-steps 1200
"""

import argparse
import json
import os
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets, load_dataset

try:
    from unsloth import FastLanguageModel
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Unsloth is not installed. In Colab, install it before running this script."
    ) from exc

from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig


SYSTEM_PROMPT = (
    "You are a precise, helpful, and intellectually rigorous assistant. "
    "Give direct answers, explain reasoning clearly, and avoid fabricating facts."
)


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def build_messages(system: str, user: str, assistant: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": clean_text(system)},
            {"role": "user", "content": clean_text(user)},
            {"role": "assistant", "content": clean_text(assistant)},
        ]
    }


def load_ultrachat(n: int, seed: int) -> Dataset:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    if n > 0:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    def convert(row: dict[str, Any]) -> dict[str, Any]:
        messages = row.get("messages", [])
        normalized = []
        has_system = False
        for m in messages:
            role = m.get("role", "")
            content = clean_text(m.get("content", ""))
            if not content or role not in {"system", "user", "assistant"}:
                continue
            if role == "system":
                has_system = True
            normalized.append({"role": role, "content": content})
        if not has_system:
            normalized = [{"role": "system", "content": SYSTEM_PROMPT}] + normalized
        return {"messages": normalized, "source": "ultrachat"}

    ds = ds.map(convert, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["messages"]) >= 3 and x["messages"][-1]["role"] == "assistant")
    return ds


def load_alpaca(n: int, seed: int) -> Dataset:
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    if n > 0:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    def convert(row: dict[str, Any]) -> dict[str, Any]:
        instruction = clean_text(row.get("instruction", ""))
        input_text = clean_text(row.get("input", ""))
        output = clean_text(row.get("output", ""))
        user = instruction if not input_text else instruction + "\n\nInput:\n" + input_text
        result = build_messages(SYSTEM_PROMPT, user, output)
        result["source"] = "alpaca"
        return result

    ds = ds.map(convert, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["messages"][-1]["content"]) > 0)
    return ds


def make_dataset(args: argparse.Namespace) -> Dataset:
    parts = []
    if args.ultrachat_samples > 0:
        parts.append(load_ultrachat(args.ultrachat_samples, args.seed))
    if args.alpaca_samples > 0:
        parts.append(load_alpaca(args.alpaca_samples, args.seed))
    if not parts:
        raise ValueError("No datasets selected. Set --ultrachat-samples or --alpaca-samples above zero.")
    ds = concatenate_datasets(parts).shuffle(seed=args.seed)
    if args.max_train_samples > 0:
        ds = ds.select(range(min(args.max_train_samples, len(ds))))
    return ds


def ensure_chat_template(tokenizer: Any) -> None:
    if getattr(tokenizer, "chat_template", None):
        return
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
        "{% endif %}"
        "{% endfor %}"
    )


def format_dataset(ds: Dataset, tokenizer: Any, packing: bool) -> Dataset:
    ensure_chat_template(tokenizer)

    def apply_template(row: dict[str, Any]) -> dict[str, str]:
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    return ds.map(apply_template, remove_columns=ds.column_names)


def train(args: argparse.Namespace) -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    ensure_chat_template(tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=True,
        loftq_config=None,
    )

    raw_ds = make_dataset(args)
    train_ds = format_dataset(raw_ds, tokenizer, args.packing)
    print(json.dumps({
        "model_name": args.model_name,
        "train_examples": len(train_ds),
        "max_seq_length": args.max_seq_length,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "packing": args.packing,
        "output_dir": args.output_dir,
    }, indent=2))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=args.weight_decay,
            lr_scheduler_type="cosine",
            seed=args.seed,
            report_to="none",
            save_strategy="steps",
            save_steps=args.save_steps,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        ),
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.save_merged_16bit:
        merged_dir = args.output_dir.rstrip("/") + "-merged-16bit"
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    if args.save_gguf:
        gguf_dir = args.output_dir.rstrip("/") + "-gguf"
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=args.gguf_quant)

    run_sample(model, tokenizer, args)


def run_sample(model: Any, tokenizer: Any, args: argparse.Namespace) -> None:
    FastLanguageModel.for_inference(model)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.sample_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    _ = model.generate(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=args.sample_max_new_tokens,
        temperature=args.sample_temperature,
        top_p=args.sample_top_p,
        do_sample=True,
        repetition_penalty=args.sample_repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="unsloth/Qwen3-4B-Base-bnb-4bit")
    p.add_argument("--output-dir", default="outputs/qwen4b-chat-sft")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--ultrachat-samples", type=int, default=12000)
    p.add_argument("--alpaca-samples", type=int, default=5000)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--packing", action="store_true")
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=60)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--save-merged-16bit", action="store_true")
    p.add_argument("--save-gguf", action="store_true")
    p.add_argument("--gguf-quant", default="q4_k_m")
    p.add_argument("--sample-prompt", default="Explain how EAN differs from a Transformer in simple terms.")
    p.add_argument("--sample-max-new-tokens", type=int, default=256)
    p.add_argument("--sample-temperature", type=float, default=0.7)
    p.add_argument("--sample-top-p", type=float, default=0.9)
    p.add_argument("--sample-repetition-penalty", type=float, default=1.08)
    return p.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("WANDB_DISABLED", "true")
    train(parse_args())
