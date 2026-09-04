#!/usr/bin/env python3
"""Lightweight DPO trainer for Cozy.

The default TRL DPOTrainer in v1.12 needs ~5+ GB GPU for a 0.6B model
because it materializes [batch, seq, vocab] tensors for entropy metrics.
This script does the bare-minimum DPO loss by hand, fitting on 6 GB
GPUs (RTX 3050).

Run AFTER sft_qwen.py has produced cozy-llm-v1/model.safetensors.
"""
from __future__ import annotations

import json
import os
import sys
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
    print("[dpo] loading model...", flush=True)
    model_dir = HERE / "model" / "cozy-llm-v1"
    if not (model_dir / "model.safetensors").exists():
        print(f"[dpo] no weights at {model_dir}", flush=True)
        sys.exit(1)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16)
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

    pairs = [json.loads(l) for l in open(PAIRS) if l.strip()]
    print(f"[dpo] {len(pairs)} pairs", flush=True)

    # Pre-tokenize: for each pair, concatenate prompt+chosen and prompt+rejected
    # with the appropriate loss masking.
    schema = SCHEMA["tools"]
    def tokenize(messages, response):
        p = tok.apply_chat_template(messages, tools=schema, tokenize=False,
                                    add_generation_prompt=True)
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
        maxlen = max(len(s) for s in seqs)
        out_ids, out_lab = [], []
        for ids, lab in seqs:
            pad_n = maxlen - len(ids)
            out_ids.append(ids + [pad_id] * pad_n)
            out_lab.append(lab + [-100] * pad_n)
        return torch.tensor(out_ids, dtype=torch.long), torch.tensor(out_lab, dtype=torch.long)

    pad_id = tok.pad_token_id
    chosen_ids, chosen_lab = pad(chosen_data, pad_id)
    rejected_ids, rejected_lab = pad(rejected_data, pad_id)
    print(f"[dpo] max seq len: {chosen_ids.shape[1]}", flush=True)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    beta = 0.1
    n_epochs = 2
    n = len(pairs)
    bsz = 1
    for ep in range(n_epochs):
        perm = torch.randperm(n)
        for s in range(0, n, bsz):
            idx = perm[s:s+bsz]
            c_batch_ids = chosen_ids[idx].to("cuda")
            c_batch_lab = chosen_lab[idx].to("cuda")
            r_batch_ids = rejected_ids[idx].to("cuda")
            r_batch_lab = rejected_lab[idx].to("cuda")

            chosen_logps = compute_logps(model, c_batch_ids, c_batch_lab)
            with torch.no_grad():
                ref_chosen_logps = compute_logps(model, c_batch_ids, c_batch_lab)
                ref_rejected_logps = compute_logps(model, r_batch_ids, r_batch_lab)
            rejected_logps = compute_logps(model, r_batch_ids, r_batch_lab)

            logits = beta * ((chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps))
            loss = -F.logsigmoid(logits).mean()

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            print(f"[dpo] ep{ep+1} step {s//bsz+1}/{n//bsz} loss={loss.item():.4f} margin={((chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps)).mean().item():.3f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT))
    tok.save_pretrained(str(OUT))
    print(f"[dpo] saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
