"""``python -m assistant.rlm_harness`` - the CLI front door.

Subcommands:

* ``dataset``  - collect SFT traces (human oracle or AI oracle)
* ``play``     - run the model and record predictions
* ``serve``    - REPL: type a user message, the model replies, log the trace
* ``info``     - dump task stats and current tool schema
* ``merge``    - fold collected traces into assistant/data/sft_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ASSISTANT = Path(__file__).resolve().parent.parent
REPO = ASSISTANT.parent


def cmd_dataset(a):
    from . import dataset_mode
    argv = [
        "--source", a.source,
        "--out", str(a.out),
    ]
    if a.limit is not None:
        argv += ["--limit", str(a.limit)]
    if a.categories:
        argv += ["--categories", *a.categories]
    if a.difficulties:
        argv += ["--difficulties", *a.difficulties]
    if a.no_hint:
        argv += ["--no-hint"]
    if a.oracle_ai:
        argv += ["--oracle-ai"]
    if a.oracle_rule:
        argv += ["--oracle-rule"]
    if a.model_dir:
        argv += ["--model-dir", a.model_dir]
    if a.adapter_dir:
        argv += ["--adapter-dir", a.adapter_dir]
    if a.device:
        argv += ["--device", a.device]
    return dataset_mode.main(argv)


def cmd_play(a):
    from . import play_mode
    argv = [
        "--source", a.source,
        "--out", str(a.out),
        "--backend", a.backend,
        "--device", a.device,
    ]
    if a.limit is not None:
        argv += ["--limit", str(a.limit)]
    if a.categories:
        argv += ["--categories", *a.categories]
    if a.difficulties:
        argv += ["--difficulties", *a.difficulties]
    if a.model_dir:
        argv += ["--model-dir", a.model_dir]
    if a.adapter_dir:
        argv += ["--adapter-dir", a.adapter_dir]
    if a.also_sft:
        argv += ["--also-sft"]
    return play_mode.main(argv)


def cmd_serve(a):
    """Interactive REPL. The model answers, you can correct it, every
    exchange is logged. The corrected turn becomes SFT data on the fly."""
    from .harness import (load_tools_schema, ModelBackend, RuleBackend,
                           SYSTEM_PROMPT)
    from .trace import Trace, append_jsonl

    tools = load_tools_schema()
    if a.backend == "rule":
        backend = RuleBackend()
    else:
        backend = ModelBackend(model_dir=a.model_dir,
                               adapter_dir=a.adapter_dir,
                               device=a.device)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[rlm-harness] serve mode. logging to {out}")
    print("Type 'quit' to exit, 'correction: <text>' to override the model.")
    while True:
        try:
            u = input("\nuser> ")
        except (EOFError, KeyboardInterrupt):
            break
        if u.strip().lower() in {"quit", "exit", "q"}:
            break
        if not u.strip():
            continue
        trace = Trace(task_id="repl", task_text=u, tools_schema=tools)
        trace.add_user(u)
        decision = backend.decide(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": u}],
            tools,
        )
        if decision.get("tool"):
            t = decision["tool"]
            print(f"cozy> tool {t['name']}({t.get('parameters', {})})")
            try:
                from executor import execute as exec_tool
                res = exec_tool(t["name"], t.get("parameters") or {})
                print(f"        -> {'OK' if res.get('ok') else 'ERR'}: "
                      f"{res.get('output', '')}")
            except Exception as exc:
                print(f"        (no executor: {exc})")
            trace.add_assistant_tool_call(t["name"],
                                           t.get("parameters", {}),
                                           producer=backend.name)
        else:
            print(f"cozy> {decision.get('text', '...')}")
            trace.add_assistant_text(decision.get("text", "..."),
                                      producer=backend.name)
        append_jsonl(out, trace.to_jsonl_dict())
    return 0


def cmd_info(a):
    from .tasks import SEED_TASKS_PATH, load_tasks
    print(f"[rlm-harness] seed file: {SEED_TASKS_PATH}")
    tasks = load_tasks("seed")
    by_cat = {}
    by_diff = {}
    for t in tasks:
        by_cat[t.category] = by_cat.get(t.category, 0) + 1
        by_diff[t.difficulty] = by_diff.get(t.difficulty, 0) + 1
    print(f"  total seed tasks: {len(tasks)}")
    print(f"  by category:      {by_cat}")
    print(f"  by difficulty:    {by_diff}")

    schema = json.loads((REPO / "team" / "tool_schema.json").read_text())
    print(f"  tools in schema:  {len(schema['tools'])}")
    for t in schema["tools"]:
        print(f"    - {t['name']}")
    return 0


def cmd_merge(a):
    """Append collected traces to assistant/data/sft_train.jsonl."""
    src = Path(a.source)
    dst = Path(a.destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        print(f"[rlm-harness] no such file: {src}")
        return 1
    n = 0
    with src.open() as fin, dst.open("a") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"[rlm-harness] merged {n} rows from {src} into {dst}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="rlm-harness",
                                 description="Cozy RLM harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("dataset", help="collect SFT traces")
    pd.add_argument("--source", default="seed")
    pd.add_argument("--out", type=Path,
                    default=Path("assistant/data/sft_extra.jsonl"))
    pd.add_argument("--limit", type=int, default=None)
    pd.add_argument("--categories", nargs="*", default=None)
    pd.add_argument("--difficulties", nargs="*", default=None)
    pd.add_argument("--no-hint", action="store_true")
    pd.add_argument("--oracle-ai", action="store_true")
    pd.add_argument("--oracle-rule", action="store_true")
    pd.add_argument("--model-dir", default=None)
    pd.add_argument("--adapter-dir", default=None)
    pd.add_argument("--device", default="cuda")
    pd.set_defaults(func=cmd_dataset)

    pp = sub.add_parser("play", help="run the model, record predictions")
    pp.add_argument("--source", default="seed")
    pp.add_argument("--out", type=Path,
                    default=Path("assistant/rlm_harness/traces/play.jsonl"))
    pp.add_argument("--limit", type=int, default=None)
    pp.add_argument("--categories", nargs="*", default=None)
    pp.add_argument("--difficulties", nargs="*", default=None)
    pp.add_argument("--backend", choices=["model", "rule"], default="model")
    pp.add_argument("--model-dir", default=None)
    pp.add_argument("--adapter-dir", default=None)
    pp.add_argument("--device", default="cuda")
    pp.add_argument("--also-sft", action="store_true")
    pp.set_defaults(func=cmd_play)

    ps = sub.add_parser("serve", help="REPL: talk to the model, log traces")
    ps.add_argument("--out", type=Path,
                    default=Path("assistant/rlm_harness/traces/serve.jsonl"))
    ps.add_argument("--backend", choices=["model", "rule"], default="model")
    ps.add_argument("--model-dir", default=None)
    ps.add_argument("--adapter-dir", default=None)
    ps.add_argument("--device", default="cuda")
    ps.set_defaults(func=cmd_serve)

    pi = sub.add_parser("info", help="show task stats and tool schema")
    pi.set_defaults(func=cmd_info)

    pm = sub.add_parser("merge",
                        help="append collected sft rows to sft_train.jsonl")
    pm.add_argument("--source", required=True, type=Path)
    pm.add_argument("--destination", type=Path,
                    default=Path("assistant/data/sft_train.jsonl"))
    pm.set_defaults(func=cmd_merge)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
