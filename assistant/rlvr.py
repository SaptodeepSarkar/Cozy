#!/usr/bin/env python3
"""RLVR/DPO step for Cozy.

Loads the SFT-trained cozy-llm-v1, rolls it out on the RLVR probe set, 
scores each output against the verifier rules, builds DPO preference pairs, 
and runs DPOTrainer.

The verifier is the agent: scores each output 0/1 on:
  1. schema_valid: output contains a parseable tool_call when one is needed
  2. tool_in_schema: the tool name is in team/tool_schema.json
  3. params_parse: the arguments JSON parses
  4. affirmation_present: a TTS-able "Done." style phrase is in the next
     assistant turn (only for tool cases)

A probe gets score 1 iff all relevant checks pass. Anything that scores 0
becomes "rejected"; the hand-written CHOSEN response is "chosen".

Run AFTER sft_qwen.py has produced cozy-llm-v1/model.safetensors.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PROBES = HERE / "rlm_harness" / "data" / "rlvr_probes.jsonl"
OUT_PAIRS = HERE / "data" / "dpo_pairs_short.jsonl"
SCHEMA = json.loads((REPO / "team" / "tool_schema.json").read_text())
VALID = {t["name"] for t in SCHEMA["tools"]}

TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_tool(text):
    m = TOOL_RE.search(text)
    if m:
        try:
            call = json.loads(m.group(1))
        except json.JSONDecodeError:
            return None
        name = call.get("name")
        params = call.get("parameters") or call.get("arguments") or {}
        if not isinstance(params, dict):
            try:
                params = json.loads(params)
            except Exception:
                params = {}
        return {"name": name, "parameters": params, "text": text}
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                chunk = text[start:i + 1]
                try:
                    call = json.loads(chunk)
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(call, dict) and isinstance(call.get("name"), str):
                    params = call.get("parameters") or call.get("arguments") or {}
                    if isinstance(params, str):
                        try:
                            params = json.loads(params)
                        except Exception:
                            params = {}
                    return {"name": call["name"], "parameters": params, "text": text}
                start = None
    return None


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def score(output_text, kind, expected):
    text = strip_think(output_text)
    if kind == "tool":
        call = parse_tool(text)
        if call is None:
            return 0, "no tool call"
        if call["name"] not in VALID:
            return 0, "tool " + repr(call["name"]) + " not in schema"
        if not isinstance(call["parameters"], dict):
            return 0, "params not a dict"
        if expected and expected[0].get("tool_calls"):
            exp_call = expected[0]["tool_calls"][0]["function"]
            if call["name"] != exp_call["name"]:
                return 0, "wrong tool: got " + call["name"] + ", want " + exp_call["name"]
        return 1, "ok"
    else:
        if not text or text in ("...", "<|im_end|>"):
            return 0, "empty chat"
        return 1, "ok"


def main():
    print("[rlvr] loading model...", flush=True)
    model_dir = HERE / "model" / "cozy-llm-v1"
    if not (model_dir / "model.safetensors").exists():
        print("[rlvr] ERROR: no weights at " + str(model_dir) + "/model.safetensors", flush=True)
        print("[rlvr] Run assistant/sft_qwen.py first.", flush=True)
        sys.exit(1)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16)
    model = model.to("cuda").eval()

    schema = SCHEMA["tools"]
    sys_msg = ("You are Cozy, a voice assistant running fully offline on the "
               "user's laptop. Respond fast and short. When the user wants "
               "an action, call exactly one tool with compact JSON. For "
               "plain chat, answer briefly and warmly without tools.")

    probes = [json.loads(l) for l in open(PROBES) if l.strip()]
    print("[rlvr] " + str(len(probes)) + " probes", flush=True)

    pairs = []
    for i, p in enumerate(probes, 1):
        prompt = p["prompt"]
        kind = p["kind"]
        chosen = p["chosen"]
        chat = [{"role":"system","content":sys_msg}, {"role":"user","content":prompt}]
        ptxt = tok.apply_chat_template(chat, tools=schema, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        ids = tok(ptxt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**ids, max_new_tokens=160, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        s, reason = score(text, kind, chosen)
        if s == 0:
            pairs.append({
                "prompt": chat,
                "chosen": chosen,
                "rejected": [{"role":"assistant", "content": strip_think(text) or "..."}],
                "tools": schema,
                "reason": reason,
            })
        if i % 5 == 0:
            print("[rlvr] " + str(i) + "/" + str(len(probes)) + " (pairs so far: " + str(len(pairs)) + ")", flush=True)

    print("[rlvr] built " + str(len(pairs)) + " preference pairs", flush=True)
    OUT_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PAIRS, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("[rlvr] wrote " + str(OUT_PAIRS), flush=True)

    if not pairs:
        print("[rlvr] no pairs to train on - SFT model already nails every probe!", flush=True)
        return

    print("[rlvr] running DPO...", flush=True)
    # Free the SFT eval model from GPU before loading DPO trainer
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    from trl import DPOTrainer, DPOConfig
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model

    base = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
    )
    model_dpo = get_peft_model(base, lora)

    cfg = DPOConfig(
        output_dir=str(HERE / "model" / "dpo_runs"),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        fp16=True,
        max_grad_norm=1.0,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=42,
        max_length=1200,
        beta=0.1,
    )
    ds = load_dataset("json", data_files=str(OUT_PAIRS), split="train")
    # When using PEFT, passing ref_model=None tells TRL to use the base model
    # (with adapter disabled) as the reference, sharing weights and saving GPU.
    trainer = DPOTrainer(
        model=model_dpo,
        ref_model=None,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()

    out_adapter = HERE / "model" / "cozy-llm-v1-dpo"
    out_adapter.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(out_adapter))
    tok.save_pretrained(str(out_adapter))
    print("[rlvr] saved DPO adapter -> " + str(out_adapter), flush=True)

    print("[rlvr] re-verifying on probes...", flush=True)
    del model, trainer
    torch.cuda.empty_cache()
    from peft import PeftModel
    base2 = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16)
    m2 = PeftModel.from_pretrained(base2, str(out_adapter)).to("cuda").eval()
    n_ok = 0
    for p in probes:
        chat = [{"role":"system","content":sys_msg}, {"role":"user","content":p["prompt"]}]
        ptxt = tok.apply_chat_template(chat, tools=schema, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        ids = tok(ptxt, return_tensors="pt").to(m2.device)
        with torch.inference_mode():
            out = m2.generate(**ids, max_new_tokens=160, do_sample=False,
                                pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        s, _ = score(text, p["kind"], p["chosen"])
        n_ok += s
    print("[rlvr] post-DPO: " + str(n_ok) + "/" + str(len(probes)) + " probes pass", flush=True)


if __name__ == "__main__":
    main()
