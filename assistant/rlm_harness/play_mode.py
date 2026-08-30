"""Play mode: drive the model on every task and record its decisions.

This is the "evaluation" half of the harness. It produces a JSONL where
each line is::

    {"task_id": ..., "task_text": ..., "prediction": {...},
     "messages_rendered": [...], "tools": [...]}

Use cases:

* Smoke-test the current LoRA before shipping.
* Generate SFT data from a *stronger* teacher model running on the same
  hardware.
* Build DPO preference pairs by running the model on each task and
  flagging failures that the human later corrects.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .trace import Trace, append_jsonl, write_jsonl
from .tasks import load_tasks
from .harness import load_tools_schema, ModelBackend, RuleBackend, Backend


def play(backend: Backend, tasks, out_path: Path,
         also_sft: bool = False) -> int:
    tools = load_tools_schema()
    n = 0
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        trace = Trace(task_id=task.id, task_text=task.text,
                      tools_schema=tools,
                      meta={"category": task.category,
                            "difficulty": task.difficulty,
                            **task.meta,
                            "backend": backend.name})
        trace.add_user(task.text)
        messages = [{"role": "system", "content": "You are Cozy."},
                    {"role": "user", "content": task.text}]
        try:
            decision = backend.decide(messages, tools)
        except Exception as exc:
            decision = {"text": f"(error: {exc})", "tool": None}

        trace.meta["prediction"] = decision
        if decision.get("tool"):
            t = decision["tool"]
            trace.add_assistant_tool_call(t["name"], t.get("parameters", {}),
                                          producer=backend.name)
        elif decision.get("text"):
            trace.add_assistant_text(decision["text"], producer=backend.name)

        append_jsonl(out_path, trace.to_jsonl_dict())
        if also_sft:
            sft_path = out_path.with_name(out_path.stem + ".sft.jsonl")
            append_jsonl(sft_path, trace.to_sft_record())
        n += 1
        if i % 20 == 0 or i == len(tasks):
            rate = n / max(1e-6, time.time() - t0)
            print(f"[play] {i}/{len(tasks)}  ({rate:.1f}/s)")
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rlm-harness play")
    p.add_argument("--source", default="seed")
    p.add_argument("--out", type=Path,
                   default=Path("assistant/rlm_harness/traces/play.jsonl"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--difficulties", nargs="*", default=None)
    p.add_argument("--backend", choices=["model", "rule"], default="model")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--adapter-dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--also-sft", action="store_true",
                   help="also dump SFT-shaped records next to the trace file")
    args = p.parse_args(argv)

    tasks = load_tasks(args.source, limit=args.limit,
                       categories=args.categories,
                       difficulties=args.difficulties)
    if not tasks:
        print("[rlm-harness] no tasks loaded")
        return 1

    if args.backend == "rule":
        backend = RuleBackend()
    else:
        backend = ModelBackend(model_dir=args.model_dir,
                               adapter_dir=args.adapter_dir,
                               device=args.device)

    n = play(backend, tasks, args.out, also_sft=args.also_sft)
    print(f"[rlm-harness] wrote {n} play traces -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
