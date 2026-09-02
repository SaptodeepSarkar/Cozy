"""QLoRA SFT training for the VLM planner on the 6 GB RTX 3050.

Hardware budget (RTX 3050 6 GB Laptop GPU, sm_86):

  Full fine-tuning ............ 20-25 GB VRAM  -> NOT POSSIBLE
  Standard LoRA (16-bit) ...... 10-12 GB VRAM  -> NOT POSSIBLE
  QLoRA (4-bit) ............... 5.5-6 GB VRAM  -> TIGHT, this script

Strategy (per the in-chat hardware assessment):

  1. Load the base in 4-bit (NF4) via bitsandbytes. The Qwen2.5-VL-3B
     AWQ checkpoint is already int4, but we re-quantise to NF4 inside
     the trainer because AWQ weights are *frozen-for-inference* (their
     scales are baked in) and a different format makes LoRA injection
     easier.
  2. Freeze the vision tower (ViT) entirely. Train LoRA only on the
     language-model linear projections (q_proj, k_proj, v_proj,
     o_proj, gate_proj, up_proj, down_proj). This cuts activation
     memory by 30-40 % and lets the model see a 512x512 image.
  3. Gradient checkpointing ON. Mandatory on 6 GB.
  4. per_device_train_batch_size = 1. Use gradient_accumulation_steps
     = 8-16 to simulate a real batch.
  5. Paged AdamW 8-bit (paged_adamw_8bit) so optimiser state spikes
     land in CPU RAM instead of VRAM.
  6. Cap max_seq_length to 1024 and max image side to 512 px to keep
     vision-token count bounded.
  7. bf16 mixed precision (Ampere+). The RTX 3050 Laptop is sm_86 so
     bf16 is supported.
  8. Optional 8-bit optimiser via bitsandbytes; otherwise fall back to
     paged_adamw_32bit.

CLI:

    python -m harness.train_sft \
        --model-dir cozy-vision/models/qwen2.5-vl-3b \
        --data cozy-vision/data/sft_planner.jsonl \
        --out cozy-vision/checkpoints/planner-lora \
        --epochs 1 --lr 2e-4 --grad-accum 8 --max-seq 1024
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

HERE = Path(__file__).resolve().parent


class PlannerSFTDataset(Dataset):
    """A simple JSONL dataset of {messages, image_path} rows.

    Each row is::

        {"image": "screenshots/foo.png", "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "[''open firefox'', ''type github.com'']"},
        ]}
    """

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
        # Cap image to 512 px on the long side to bound vision tokens
        w, h = img.size
        m = max(w, h)
        if m > 512:
            scale = 512 / m
            img = img.resize((int(w * scale), int(h * scale)))
        msgs = row["messages"]
        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text],
            images=[img],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_seq_length,
        )
        out = {k: v[0] for k, v in inputs.items()}
        labels = out["input_ids"].clone()
        out["labels"] = labels
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
            out["input_ids"].append(
                torch.cat([b["input_ids"], torch.full((pad,), pad_id, dtype=b["input_ids"].dtype)])
            )
            out["attention_mask"].append(
                torch.cat([b["attention_mask"], torch.zeros(pad, dtype=b["attention_mask"].dtype)])
            )
            out["labels"].append(
                torch.cat([b["labels"], torch.full((pad,), -100, dtype=b["labels"].dtype)])
            )
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

    bnb_config = BitsAndBytesConfig(
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

    print("[train_sft] loading base model in 4-bit (NF4)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "5GiB", "cpu": "30GiB"},
    )

    if hasattr(model, "visual"):
        for p in model.visual.parameters():
            p.requires_grad = False
        print("[train_sft] vision tower frozen")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = PlannerSFTDataset(args.data, processor, max_seq_length=args.max_seq)
    collator = CollatorPad(processor=processor)
    print(f"[train_sft] dataset size = {len(dataset)}")

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

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"[train_sft] saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
