#!/usr/bin/env python3
"""Generate verifier-scored preference pairs from held-out Cozy probes.

This is the RLVR rollout stage. It does not train: failed SFT rollouts become
rejected responses paired with deterministic, verifier-approved responses for
the following DPO stage (`dpo_light.py`).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from evaluation import chosen_message, load_probes, score_probe, strip_thinking  # noqa: E402

SYSTEM = (
    "You are Cozy, a voice assistant running fully offline on the user's laptop. "
    "Respond fast and short. When the user wants an action, call exactly one tool "
    "with compact JSON. For plain chat, answer briefly and warmly without tools."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(HERE / "model" / "cozy-llm-v1"))
    parser.add_argument("--probes", default=str(HERE / "data" / "rlvr_probes.jsonl"))
    parser.add_argument("--out", default=str(HERE / "data" / "dpo_pairs_short.jsonl"))
    parser.add_argument("--metrics-out", default=str(HERE / "data" / "rlvr_metrics.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required for RLVR rollouts")
    model_dir = Path(args.model)
    if not any(model_dir.glob("*.safetensors")):
        parser.error(f"no model weights found in {model_dir}; run SFT first")

    schema = json.loads((REPO / "team" / "tool_schema.json").read_text())["tools"]
    valid_tools = {tool["name"] for tool in schema}
    probes = load_probes(Path(args.probes))
    if args.limit:
        probes = probes[:args.limit]

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to("cuda").eval()
    torch.backends.cuda.matmul.allow_tf32 = True

    pairs = []
    results = []
    started = time.perf_counter()
    for offset in range(0, len(probes), args.batch_size):
        batch = probes[offset:offset + args.batch_size]
        chats = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": probe["user"]}] for probe in batch]
        rendered = [tokenizer.apply_chat_template(
            chat, tools=schema, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ) for chat in chats]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_width = inputs["input_ids"].shape[1]
        outputs = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=False)
        for probe, chat, output in zip(batch, chats, outputs):
            passed, reason = score_probe(output, probe, valid_tools)
            results.append({"id": probe["id"], "passed": passed, "reason": reason, "output": strip_thinking(output)})
            if not passed:
                pairs.append({
                    "prompt": chat,
                    "chosen": [chosen_message(probe)],
                    "rejected": [{"role": "assistant", "content": strip_thinking(output) or "..."}],
                    "tools": schema,
                    "probe_id": probe["id"],
                    "reason": reason,
                })
        print(f"[rlvr] {min(offset + len(batch), len(probes))}/{len(probes)} · failures={len(pairs)}", flush=True)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    tmp.replace(output_path)
    metrics = {
        "model": str(model_dir), "n_probes": len(probes), "passed": len(probes) - len(pairs),
        "failed": len(pairs), "accuracy": (len(probes) - len(pairs)) / max(1, len(probes)),
        "elapsed_s": round(time.perf_counter() - started, 3), "results": results,
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[rlvr] accuracy={metrics['accuracy']:.1%}; wrote {len(pairs)} pairs to {output_path}")


if __name__ == "__main__":
    main()
