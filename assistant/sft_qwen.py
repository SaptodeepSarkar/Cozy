#!/usr/bin/env python3
"""LoRA fine-tune of Qwen3-0.6B on Cozy's function-calling dataset.

Run:  python sft_qwen.py
Out:  assistant/model/cozy-llm-v1  (merged, ready for llama.cpp/vLLM/HF)
"""
from __future__ import annotations

import argparse
import json
import os
import time
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--adapter-out", default=str(HERE / "model" / "cozy-llm-v1-adapter"))
    parser.add_argument("--run-dir", default=str(HERE / "model" / "sft_runs"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--resume", action="store_true", help="resume the latest SFT checkpoint")
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required for LLM SFT; run train.sh preflight for details")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def to_text(row):
        return {"text": render(row, tok)}

    ds = load_dataset(
        "json",
        data_files={
            "train": str(DATA / "sft_train.jsonl"),
            "validation": str(DATA / "sft_val.jsonl"),
        },
    )
    ds = ds.map(to_text, num_proc=args.workers, desc="Rendering chat templates")

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
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
    model.config.use_cache = False
    model = model.to("cuda")  # explicit: never fall back to CPU silently
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir=args.run_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=5,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        tf32=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=not args.no_grad_checkpoint,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        shuffle_dataset=True,
        dataloader_num_workers=args.workers,
        dataloader_persistent_workers=args.workers > 0,
        dataloader_prefetch_factor=2 if args.workers > 0 else None,
        report_to=[],
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
    )
    started = time.time()
    train_result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    eval_metrics = trainer.evaluate()

    # save the SMALL adapter first (committable to git)
    adapter_dir = Path(args.adapter_out)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    print("saved adapter ->", adapter_dir)

    merged = trainer.model.merge_and_unload()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use safe_serialization=True (default in transformers >= 4.x) and 
    # max_shard_size=2GB so the file definitely lands in cozy-llm-v1/.
    # Also explicitly print files to catch the case where save_pretrained
    # silently writes elsewhere.
    saved = merged.save_pretrained(str(output_dir), safe_serialization=True,
                                    max_shard_size="2GB")
    print(f"save_pretrained returned: {saved}")
    tok.save_pretrained(str(output_dir))
    print("saved merged model ->", output_dir)
    print("contents of OUT:", sorted(p.name for p in output_dir.iterdir()))
    metrics = {
        **train_result.metrics,
        **eval_metrics,
        "wall_time_s": round(time.time() - started, 2),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "effective_batch_size": args.batch_size * args.grad_accum,
        "dtype": str(dtype).removeprefix("torch."),
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")

    # smoke test: does it emit a tool call for a seen-style command?
    merged.eval()
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_TEXT},
         {"role": "user", "content": "set volume to 30"}],
        tools=json.loads((HERE.parent / "team" / "tool_schema.json")
                         .read_text())["tools"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tok(prompt, return_tensors="pt").to(merged.device)
    out_ids = merged.generate(**ids, max_new_tokens=200, do_sample=False)
    text = tok.decode(out_ids[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=False)
    print("SMOKE OUTPUT:", text.strip()[:400])


SYSTEM_TEXT = (
    "You are Cozy, a voice assistant running fully offline on the user's "
    "laptop. Respond fast and short. When the user wants an action, call "
    "exactly one tool with compact JSON. For plain chat, answer briefly "
    "and warmly without tools."
)


if __name__ == "__main__":
    main()
