#!/usr/bin/env python3
"""Exact tool-selection benchmark for Cozy's current model artifacts.

Examples: `python eval_llm.py --model sft=assistant/model/cozy-llm-v1`
or `--model dpo=assistant/model/cozy-llm-v1:assistant/model/cozy-llm-v1-dpo`.
The optional `base:adapter` suffix evaluates a PEFT adapter on the base model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "assistant"))
from evaluation import load_probes, score_probe  # noqa: E402

OUT = REPO / "models" / "benchmarks" / "llm_eval.json"
PROBES = REPO / "assistant" / "data" / "rlvr_probes.jsonl"
SYSTEM = (
    "You are Cozy, a voice assistant running fully offline on the user's laptop. "
    "Respond fast and short. When the user wants an action, call exactly one tool "
    "with compact JSON. For plain chat, answer briefly and warmly without tools."
)


def evaluate(label: str, spec: str, probes: list[dict], schema: list[dict], batch_size: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parts = spec.split(":", 1)
    model_path, adapter_path = parts[0], parts[1] if len(parts) == 2 else None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.to("cuda").eval()
    valid = {tool["name"] for tool in schema}
    passed = 0
    details = []
    started = time.perf_counter()
    for offset in range(0, len(probes), batch_size):
        batch = probes[offset:offset + batch_size]
        rendered = [tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p["user"]}],
            tools=schema, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ) for p in batch]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
        width = inputs["input_ids"].shape[1]
        outputs = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=False)
        for probe, output in zip(batch, outputs):
            ok, reason = score_probe(output, probe, valid)
            passed += int(ok)
            details.append({"id": probe["id"], "passed": ok, "reason": reason})
    result = {
        "model": label, "spec": spec, "accuracy": passed / max(1, len(probes)),
        "passed": passed, "total": len(probes), "tool_total": sum(p["kind"] == "tool" for p in probes),
        "chat_total": sum(p["kind"] == "chat" for p in probes),
        "latency_s": round(time.perf_counter() - started, 3), "details": details,
    }
    del model
    torch.cuda.empty_cache()
    print(f"{label}: {passed}/{len(probes)} ({result['accuracy']:.1%})")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", metavar="LABEL=PATH[:ADAPTER]",
                        help="repeatable; defaults to the current SFT artifact")
    parser.add_argument("--probes", default=str(PROBES))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        parser.error("CUDA is required for LLM benchmarks")
    schema = json.loads((REPO / "team" / "tool_schema.json").read_text())["tools"]
    probes = load_probes(Path(args.probes))
    if args.limit:
        probes = probes[:args.limit]
    specs = args.model or [f"sft={REPO / 'assistant/model/cozy-llm-v1'}"]
    models = {}
    for item in specs:
        if "=" not in item:
            parser.error(f"--model must be LABEL=PATH[:ADAPTER], got {item!r}")
        label, spec = item.split("=", 1)
        models[label] = evaluate(label, spec, probes, schema, args.batch_size)
    result = {"probes": probes, "models": models, "generated_at": time.time(),
              "gpu": torch.cuda.get_device_name(0)}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", output)


if __name__ == "__main__":
    main()
