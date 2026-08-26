#!/usr/bin/env python3
"""LoRA fine-tune of Qwen3-0.6B on Cozy's function-calling dataset.

Run:  python sft_qwen.py
Out:  assistant/model/cozy-llm-v1  (merged, ready for llama.cpp/vLLM/HF)
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

HERE = Path(__file__).resolve().parent
BASE = "Qwen/Qwen3-0.6B"
OUT = HERE / "model" / "cozy-llm-v1"
DATA = HERE / "data"


def render(row, tok):
    return tok.apply_chat_template(
        row["messages"],
        tools=row["tools"],
        tokenize=False,
        add_generation_prompt=False,
    )


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE)

    def to_text(row):
        return {"text": render(row, tok)}

    ds = load_dataset(
        "json",
        data_files={
            "train": str(DATA / "sft_train.jsonl"),
            "validation": str(DATA / "sft_val.jsonl"),
        },
    )
    ds = ds.map(to_text)

    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model = model.to("cuda")  # explicit: never fall back to CPU silently
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir=str(HERE / "model" / "sft_runs"),
        num_train_epochs=3,
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=40,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        max_length=1024,
        report_to=[],
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
    )
    trainer.train()

    merged = trainer.model.merge_and_unload()
    OUT.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(OUT))
    tok.save_pretrained(str(OUT))
    print("saved merged model ->", OUT)

    # smoke test: does it emit a tool call for a seen-style command?
    merged.eval()
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_TEXT},
         {"role": "user", "content": "set volume to 30"}],
        tools=json.loads((HERE.parent / "team" / "tool_schema.json")
                         .read_text())["tools"],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok(prompt, return_tensors="pt").to(merged.device)
    out_ids = merged.generate(**ids, max_new_tokens=80, do_sample=False)
    text = tok.decode(out_ids[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    print("SMOKE OUTPUT:", text.strip()[:200])


SYSTEM_TEXT = (
    "You are Cozy, a voice assistant running fully offline on the user's "
    "laptop. Respond fast and short. When the user wants an action, call "
    "exactly one tool with compact JSON. For plain chat, answer briefly "
    "and warmly without tools."
)


if __name__ == "__main__":
    main()
