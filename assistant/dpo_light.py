#!/usr/bin/env python3
"""Lightweight DPO trainer for Cozy.

The default TRL DPOTrainer in v1.12 needs ~5+ GB GPU for a 0.6B model
because it materializes [batch, seq, vocab] tensors for entropy metrics.
This script does the bare-minimum DPO loss by hand, fitting on 6 GB
GPUs (RTX 3050).

Run AFTER sft_qwen.py has produced cozy-llm-v1/model.safetensors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PAIRS = HERE / "data" / "dpo_pairs_short.jsonl"
OUT = HERE / "model" / "cozy-llm-v1-dpo"
SCHEMA = json.loads((REPO / "team" / "tool_schema.json").read_text())


def compute_logps(model, input_ids, labels):
    """Per-token log-probabilities of the labels under the model."""
    out = model(input_ids=input_ids)
    logits = out.logits[:, :-1, :]
    target = labels[:, 1:]
    # mask out padding
    mask = (target != -100).float()
    logps = F.log_softmax(logits.float(), dim=-1)
    per_token = logps.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (per_token * mask).sum(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(HERE / "model" / "cozy-llm-v1"))
    ap.add_argument("--pairs", default=str(PAIRS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=1200)
    args = ap.parse_args()
    started = time.perf_counter()
    print("[dpo] loading model...", flush=True)
    model_dir = Path(args.model)
    if not any(model_dir.glob("*.safetensors")):
        print(f"[dpo] no weights at {model_dir}", flush=True)
        sys.exit(1)
    if not torch.cuda.is_available():
        ap.error("CUDA is required for DPO")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=dtype,
                                                 attn_implementation="sdpa")
    base.config.use_cache = False
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(base, lora)
    model = model.to("cuda")
    model.print_trainable_parameters()

    pairs = [json.loads(l) for l in Path(args.pairs).read_text().splitlines() if l.strip()]
    print(f"[dpo] {len(pairs)} pairs", flush=True)

    # Pre-tokenize: for each pair, concatenate prompt+chosen and prompt+rejected
    # with the appropriate loss masking.
    schema = SCHEMA["tools"]
    def tokenize(messages, response):
        p = tok.apply_chat_template(messages, tools=schema, tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tok(p, return_tensors=None, add_special_tokens=False)["input_ids"]
        # response may be a list of messages or a string
        if isinstance(response, list):
            r = tok.apply_chat_template(messages + response, tools=schema,
                                         tokenize=False,
                                         add_generation_prompt=False)
            r_ids = tok(r, return_tensors=None, add_special_tokens=False)["input_ids"]
        else:
            r_ids = tok(response, return_tensors=None, add_special_tokens=False)["input_ids"]
        full = prompt_ids + r_ids[len(prompt_ids):] if r_ids[:len(prompt_ids)] == prompt_ids else prompt_ids + r_ids
        labels = [-100] * len(prompt_ids) + full[len(prompt_ids):]
        if len(full) > args.max_length:
            full, labels = full[-args.max_length:], labels[-args.max_length:]
        return full, labels

    print("[dpo] tokenizing pairs...", flush=True)
    chosen_data, rejected_data = [], []
    for p in pairs:
        c_ids, c_lab = tokenize(p["prompt"], p["chosen"])
        r_ids, r_lab = tokenize(p["prompt"], p["rejected"])
        chosen_data.append((c_ids, c_lab))
        rejected_data.append((r_ids, r_lab))

    # Pad to max length
    def pad(seqs, pad_id):
        maxlen = min(args.max_length, max(len(pair[0]) for pair in seqs))
        out_ids, out_lab = [], []
        for ids, lab in seqs:
            pad_n = maxlen - len(ids)
            out_ids.append(ids[:maxlen] + [pad_id] * max(0, pad_n))
            out_lab.append(lab[:maxlen] + [-100] * max(0, pad_n))
        return torch.tensor(out_ids, dtype=torch.long), torch.tensor(out_lab, dtype=torch.long)

    pad_id = tok.pad_token_id
    chosen_ids, chosen_lab = pad(chosen_data, pad_id)
    rejected_ids, rejected_lab = pad(rejected_data, pad_id)
    print(f"[dpo] max seq len: {chosen_ids.shape[1]}", flush=True)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                              fused=torch.cuda.is_available())
    beta = 0.1
    n_epochs = args.epochs
    n = len(pairs)
    bsz = args.batch_size
    if n == 0:
        print("[dpo] no preference pairs; nothing to train")
        return
    model.train()
    for ep in range(n_epochs):
        perm = torch.randperm(n)
        optim.zero_grad(set_to_none=True)
        for step, s in enumerate(range(0, n, bsz), 1):
            idx = perm[s:s+bsz]
            c_batch_ids = chosen_ids[idx].to("cuda")
            c_batch_lab = chosen_lab[idx].to("cuda")
            r_batch_ids = rejected_ids[idx].to("cuda")
            r_batch_lab = rejected_lab[idx].to("cuda")

            chosen_logps = compute_logps(model, c_batch_ids, c_batch_lab)
            with torch.no_grad():
                # Reference probabilities must come from the frozen base, not
                # the adapter currently being optimized.
                with model.disable_adapter():
                    ref_chosen_logps = compute_logps(model, c_batch_ids, c_batch_lab)
                    ref_rejected_logps = compute_logps(model, r_batch_ids, r_batch_lab)
            rejected_logps = compute_logps(model, r_batch_ids, r_batch_lab)

            logits = beta * ((chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps))
            loss = -F.logsigmoid(logits).mean()

            (loss / args.grad_accum).backward()
            if step % args.grad_accum == 0 or s + bsz >= n:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
            print(f"[dpo] ep{ep+1} step {step}/{(n + bsz - 1)//bsz} loss={loss.item():.4f} margin={((chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps)).mean().item():.3f}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tok.save_pretrained(str(out_dir))
    metrics = {"pairs": n, "epochs": n_epochs, "batch_size": bsz,
               "grad_accum": args.grad_accum, "elapsed_s": round(time.perf_counter() - started, 2),
               "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1)}
    (out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[dpo] saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
