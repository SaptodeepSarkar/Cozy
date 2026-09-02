"""QLoRA SFT training for the VLA (UI-TARS-2B-SFT) on the 6 GB GPU.

Same 6 GB constraint as the planner SFT:

  * 4-bit NF4 base via bitsandbytes
  * Vision tower frozen
  * LoRA on q/k/v/o/gate/up/down_proj only
  * Gradient checkpointing on
  * per_device_train_batch_size=1
  * paged_adamw_8bit
  * bf16 mixed precision
  * max_seq_length=1024

The data is a JSONL of (screenshot, todo_item) -> (action_string) rows
collected by ``harness.cli collect``. The trainer teaches UI-TARS to
ground on your specific Pop!_OS COSMIC theme, window decorations,
and ydotool coordinate space.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


SYSTEM_PROMPT = """You are a GUI agent executing ONE todo item at a time on Pop!_OS / COSMIC.

You will be given a single, atomic todo item and the current screenshot. Output ONE action.

Allowed actions (one per turn):
  click(x, y)        Click at the given pixel coordinates.
  type(text)         Type the given text into the focused element.
  hotkey(k1, k2,..)  Press a hotkey combination.
  scroll(dx, dy)     Scroll.
  wait()             Wait briefly.
  finished()         The todo item is complete.

Output ONLY the action call."""


class VLASFTDataset(Dataset):
    """JSONL rows: {image, todo: {action, target, check}, assistant: "click(120,340)"}."""

    def __init__(self, jsonl_path, processor, max_seq_length=1024):
        self.path = Path(jsonl_path)
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.rows = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["image"]).convert("RGB")
        # cap to 512px to bound vision tokens
        w, h = img.size
        m = max(w, h)
        if m > 512:
            scale = 512 / m
            img = img.resize((int(w * scale), int(h * scale)))
        todo = row.get("todo", {})
        user_text = f"Todo: action={todo.get('action','')}, target={todo.get('target','')}\nNext action:"
        assistant = row.get("assistant", "wait()")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": user_text},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": assistant}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(
            text=[text], images=[img], return_tensors="pt",
            padding=False, truncation=True, max_length=self.max_seq_length,
        )
        out = {k: v[0] for k, v in inputs.items()}
        out["labels"] = out["input_ids"].clone()
        return out


@dataclass
class CollatorPad:
    processor: object
    pad_to_multiple_of: int = 8

    def __call__(self, batch):
        max_len = max(b["input_ids"].size(0) for b in batch)
        max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        pad_id = self.processor.tokenizer.pad_token_id or 0
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            n = b["input_ids"].size(0)
            pad = max_len - n
            out["input_ids"].append(torch.cat([b["input_ids"], torch.full((pad,), pad_id, dtype=b["input_ids"].dtype)]))
            out["attention_mask"].append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=b["attention_mask"].dtype)]))
            out["labels"].append(torch.cat([b["labels"], torch.full((pad,), -100, dtype=b["labels"].dtype)]))
        out["input_ids"] = torch.stack(out["input_ids"])
        out["attention_mask"] = torch.stack(out["attention_mask"])
        out["labels"] = torch.stack(out["labels"])
        if "pixel_values" in batch[0]:
            out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
        if "image_grid_thw" in batch[0]:
            out["image_grid_thw"] = torch.stack([b["image_grid_thw"] for b in batch])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=200)
    args = ap.parse_args()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    print("[train_sft_vla] loading UI-TARS-2B in NF4 4-bit ...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "2GiB", "cpu": "30GiB"},
    )

    if hasattr(model, "visual"):
        for p in model.visual.parameters():
            p.requires_grad = False
        print("[train_sft_vla] vision tower frozen")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=target_modules, lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = VLASFTDataset(args.data, processor, max_seq_length=args.max_seq)
    collator = CollatorPad(processor=processor)
    print(f"[train_sft_vla] dataset size = {len(dataset)}")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=dataset, data_collator=collator)
    trainer.train()
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"[train_sft_vla] saved VLA LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
