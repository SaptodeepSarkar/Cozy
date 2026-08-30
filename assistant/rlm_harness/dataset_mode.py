"""Dataset mode: the human (or another AI) plays the oracle.

For every task, the loop is:

    1. Print the task.
    2. Show a hint of what tool the model *would* call (optional).
    3. Wait for the oracle to type either:
         a. plain text          -> assistant text reply
         b. ``tool name k=v``   -> assistant emits that tool call
         c. ``exec``            -> if you typed a tool call, also run it
                                  and feed the result back as a tool message
         d. ``skip``            -> discard this task
         e. ``quit`` / ``done`` -> stop

The completed trace is appended to the output JSONL in the same schema
as ``assistant/data/sft_train.jsonl`` so it drops straight into SFT.

The loop also supports an "AI oracle" mode: if you pass ``--oracle-ai``,
this script calls a child LLM (or the rule backend) to produce the
"ground truth" tool call. Use this when the user wants to bulk-generate
training data without manual typing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .trace import Trace, append_jsonl
from .tasks import load_tasks
from .harness import load_tools_schema, RuleBackend, ModelBackend, Backend


HELP_TEXT = (
    "Commands:\n"
    "  <text>            plain assistant reply (no tool)\n"
    "  tool NAME k=v     emit a tool call, e.g.  tool app.open name=chrome\n"
    "  exec              run the previous tool call and feed the result back\n"
    "  hint              show the model's prediction for this task\n"
    "  skip              discard this task\n"
    "  done / quit       stop the session\n"
)


def parse_tool_call(line):
    """Parse ``tool NAME k=v k=v ...`` into a tool-call dict."""
    line = line.strip()
    if not line.startswith("tool"):
        return None
    body = line[4:].strip()
    if not body:
        return None
    parts = body.split()
    name = parts[0]
    args = {}
    for tok in parts[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                v = json.loads(v)
            except Exception:
                pass
            args[k] = v
    return {"name": name, "parameters": args}


def maybe_run_tool(name, args):
    """Try to execute the tool via the real executor and return the result."""
    try:
        from executor import execute as exec_tool
    except Exception as exc:
        return f"(executor unavailable: {exc})"
    res = exec_tool(name, args or {})
    out = res.get("output", "")
    return ("OK: " if res.get("ok") else "ERR: ") + str(out)


def _ask(prompt):
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        return "done"


def run_human_oracle(tasks, out_path, harness, show_hints=True):
    """Interactive loop. Returns the number of traces written."""
    n = 0
    print(f"\n[rlm-harness] dataset mode: {len(tasks)} tasks queued.")
    print(f"[rlm-harness] output: {out_path}")
    print(HELP_TEXT)

    for i, task in enumerate(tasks, 1):
        print("\n" + "=" * 70)
        print(f"[task {i}/{len(tasks)}] {task}")
        if task.hint:
            print(f"  hint: {task.hint}")
        if show_hints and harness is not None:
            t_probe = Trace(task_id=task.id, task_text=task.text,
                            tools_schema=harness.tools)
            t_probe.add_user(task.text)
            pred = harness.decide(t_probe)
            print(f"  model says: {pred}")

        while True:
            raw = _ask("oracle> ")
            cmd = raw.strip().lower()
            if cmd in {"done", "quit", "exit", "q"}:
                print(f"[rlm-harness] stopping after {n} traces")
                return n
            if cmd == "skip":
                print("  skipped")
                break
            if cmd == "help":
                print(HELP_TEXT)
                continue
            if cmd == "hint" and harness is not None:
                t_probe = Trace(task_id=task.id, task_text=task.text,
                                tools_schema=harness.tools)
                t_probe.add_user(task.text)
                print("  model says:", harness.decide(t_probe))
                continue
            if cmd == "exec":
                print("  (nothing to execute - type a tool call first)")
                continue

            trace = Trace(task_id=task.id, task_text=task.text,
                          tools_schema=harness.tools if harness else
                          load_tools_schema(),
                          meta={"category": task.category,
                                "difficulty": task.difficulty,
                                **task.meta})
            trace.add_user(task.text)

            tc = parse_tool_call(raw)
            if tc is not None:
                trace.add_assistant_tool_call(tc["name"], tc["parameters"],
                                              producer="human")
                print(f"  -> tool {tc['name']}({tc['parameters']})")
                run_it = _ask("    run it? [y/N] > ").strip().lower()
                if run_it == "y":
                    out = maybe_run_tool(tc["name"], tc["parameters"])
                    print("    result:", out)
                    trace.add_tool_result(tc["name"], out)
            elif raw.strip():
                trace.add_assistant_text(raw.strip(), producer="human")
                print(f"  -> text: {raw.strip()[:80]}")

            while True:
                more = _ask("more turns? [user/tool/text/n] > ").strip()
                lm = more.lower()
                if not lm or lm == "n":
                    break
                if lm == "user":
                    u = _ask("  user> ")
                    if u.strip():
                        trace.add_user(u.strip())
                elif lm == "tool":
                    raw2 = _ask("  tool NAME k=v > ")
                    # accept both "NAME k=v" and "tool NAME k=v"
                    if not raw2.lstrip().startswith("tool"):
                        raw2 = "tool " + raw2.lstrip()
                    tc2 = parse_tool_call(raw2)
                    if tc2:
                        trace.add_assistant_tool_call(tc2["name"],
                                                      tc2["parameters"],
                                                      producer="human")
                elif lm == "text":
                    t = _ask("  assistant> ")
                    if t.strip():
                        trace.add_assistant_text(t.strip(), producer="human")
                else:
                    print("    (user | tool | text | n)")

            append_jsonl(out_path, trace.to_sft_record())
            verbose = out_path.with_suffix(".traces.jsonl")
            append_jsonl(verbose, trace.to_jsonl_dict())
            n += 1
            print(f"  [saved]  (total {n})")
            break
    return n


def oracle_tools_schema(backend):
    return load_tools_schema()


def run_ai_oracle(tasks, out_path, oracle,
                  harness_for_eval=None, max_per_task=4):
    """Use another model (or the rule backend) to produce traces."""
    n = 0
    for i, task in enumerate(tasks, 1):
        trace = Trace(task_id=task.id, task_text=task.text,
                      tools_schema=oracle_tools_schema(oracle),
                      meta={"category": task.category,
                            "difficulty": task.difficulty,
                            **task.meta})
        trace.add_user(task.text)
        msgs = [{"role": "system", "content": "You are Cozy."},
                {"role": "user", "content": task.text}]
        decision = oracle.decide(msgs, trace.tools_schema)
        if decision.get("tool"):
            t = decision["tool"]
            trace.add_assistant_tool_call(t["name"], t.get("parameters", {}),
                                          producer="ai_oracle")
            out = maybe_run_tool(t["name"], t.get("parameters", {}))
            trace.add_tool_result(t["name"], out)
        else:
            trace.add_assistant_text(decision.get("text", "..."),
                                      producer="ai_oracle")

        if harness_for_eval is not None:
            try:
                probe_msgs = [{"role": "system",
                               "content": "You are Cozy."},
                              {"role": "user", "content": task.text}]
                student = harness_for_eval.decide(probe_msgs,
                                                   trace.tools_schema)
                trace.meta["student_prediction"] = student
            except Exception as exc:
                trace.meta["student_error"] = str(exc)

        append_jsonl(out_path, trace.to_sft_record())
        verbose = out_path.with_suffix(".traces.jsonl")
        append_jsonl(verbose, trace.to_jsonl_dict())
        n += 1
        if i % 20 == 0:
            print(f"[rlm-harness] ai-oracle: {i}/{len(tasks)}")
    return n


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="rlm-harness dataset",
        description="Collect SFT traces for Cozy's tool-calling LLM.")
    p.add_argument("--source", default="seed")
    p.add_argument("--out", type=Path,
                   default=Path("assistant/data/sft_extra.jsonl"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--difficulties", nargs="*", default=None)
    p.add_argument("--no-hint", action="store_true")
    p.add_argument("--oracle-ai", action="store_true")
    p.add_argument("--oracle-rule", action="store_true")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--adapter-dir", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    tasks = load_tasks(args.source, limit=args.limit,
                       categories=args.categories,
                       difficulties=args.difficulties)
    if not tasks:
        print("[rlm-harness] no tasks loaded")
        return 1

    from .harness import load_tools_schema, ModelBackend, RuleBackend
    tools = load_tools_schema()
    if args.oracle_rule:
        oracle = RuleBackend()
    elif args.oracle_ai:
        oracle = ModelBackend(model_dir=args.model_dir,
                              adapter_dir=args.adapter_dir,
                              device=args.device)
    else:
        oracle = None

    if oracle is not None:
        n = run_ai_oracle(tasks, args.out, oracle)
        print(f"[rlm-harness] wrote {n} ai-oracle traces -> {args.out}")
        return 0

    n = run_human_oracle(tasks, args.out, oracle, show_hints=not args.no_hint)
    print(f"[rlm-harness] wrote {n} traces -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
