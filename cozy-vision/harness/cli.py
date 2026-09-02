"""Command-line interface for the Cozy-Vision agent.

Subcommands:

  run GOAL              Plan + execute a single goal end-to-end.
  ask QUESTION          Visual Q&A: answer a question about the screen.
  plan GOAL             Plan only, no execution. Prints the todo list.
  collect --tasks N     Collect SFT traces by running N tasks.
  train                 QLoRA SFT on the planner.
  train-vla             QLoRA SFT on the VLA (grounder).
  smoke                 Load both models, run a synthetic inference.
  list-tasks            List seed tasks.
  seed                  Dump seed task JSONL.
  snapshot              Print the live OS context.
"""
from __future__ import annotations

import argparse
import logging
import os

# Quiet transformers progress bars and weights loading noise
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
import json
import sys
import time
from pathlib import Path

from .context import OSContextCollector
from .driver import PopOSDriver
from .grounder import Grounder
from .planner import Planner
from .runner import VisionRunner, TRACE_DIR
from . import tasks


def _banner(msg: str) -> None:
    bar = "=" * (len(msg) + 4)
    print(f"\n{bar}\n  {msg}\n{bar}")


def cmd_run(args) -> int:
    driver = PopOSDriver()
    planner = Planner()
    grounder = Grounder()
    runner = VisionRunner(
        planner=planner, grounder=grounder, driver=driver,
        trace_name=f"cli_run_{int(time.time())}.jsonl",
    )
    _banner(f"RUN: {args.goal!r}")
    try:
        runner.warmup()
        result = runner.run(args.goal)
    finally:
        runner.shutdown()
    print(f"\n  success: {result.success}")
    print(f"  score:   {result.score:.2f}")
    print(f"  reason:  {result.reward.reason}")
    print(f"  todo:    {len(result.todo)} items, {sum(1 for a in result.actions if 'action' in a)} actions")
    print(f"  time:    {result.duration_s:.1f}s")
    if result.error:
        print(f"  error:   {result.error}")
    return 0 if result.success else 1


def cmd_ask(args) -> int:
    driver = PopOSDriver(dryrun=True)
    planner = Planner()
    _banner(f"ASK: {args.question!r}")
    try:
        planner.load()
        ctx = OSContextCollector().snapshot(screen_size=driver.screen_size)
        shot = driver.screenshot()
        ans = planner.answer(args.question, shot, ctx)
    finally:
        planner.free()
    print(f"\n  {ans.text}")
    return 0


def cmd_plan(args) -> int:
    driver = PopOSDriver(dryrun=True)
    planner = Planner()
    _banner(f"PLAN: {args.goal!r}")
    try:
        planner.load()
        shot = driver.screenshot()
        ctx = OSContextCollector().snapshot(screen_size=driver.screen_size)
        plan = planner.plan(args.goal, shot, ctx)
    finally:
        planner.free()
    print("\n  OS context:")
    print("  " + ctx.to_prompt().replace("\n", "\n  "))
    print("\n  Todo list:")
    for i, t in enumerate(plan.todo, 1):
        print(f"    {i}. [{t.action}] target={t.target!r}")
        if t.check:
            print(f"       check: {t.check}")
    print(f"\n  raw: {plan.raw[:200]!r}")
    return 0


def cmd_task(args) -> int:
    """Run a single desktop task end-to-end with streamed progress."""
    from .runner import TaskSession
    _banner(f"TASK: {args.goal!r}")
    with TaskSession() as session:
        result = session.task(args.goal)
    print(f"\n  success: {result.success}")
    print(f"  score:   {result.score:.2f}")
    print(f"  reason:  {result.reward.reason}")
    print(f"  time:    {result.duration_s:.1f}s")
    return 0 if result.success else 1


def cmd_session(args) -> int:
    """Interactive REPL: keep the models loaded, accept one task per line."""
    from .runner import TaskSession
    _banner("Cozy-Vision session — type a task, blank line to quit")
    with TaskSession() as session:
        while True:
            try:
                goal = input("\ntask> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not goal:
                break
            session.task(goal)
    _banner("session ended")
    return 0


def cmd_collect(args) -> int:
    driver = PopOSDriver()
    planner = Planner()
    grounder = Grounder()
    runner = VisionRunner(
        planner=planner, grounder=grounder, driver=driver,
        trace_name=f"collect_{int(time.time())}.jsonl",
    )
    _banner(f"COLLECT: {args.tasks} tasks")
    runner.warmup()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    collected = 0
    for t in tasks.iter_tasks():
        if collected >= args.tasks:
            break
        print(f"\n  [{collected+1}/{args.tasks}] {t.text}")
        result = runner.run(t.text, task=t.to_dict())
        row = {
            "task_id": t.id,
            "category": t.category,
            "user_text": t.text,
            "todo": result.todo,
            "score": result.score,
            "actions": result.actions,
            "contexts": result.contexts,
            "duration_s": result.duration_s,
        }
        with out.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        collected += 1
    runner.shutdown()
    print(f"\n  wrote {collected} rows to {out}")
    return 0


def cmd_train(args) -> int:
    from . import train_sft
    argv = [
        "--model-dir", args.model_dir,
        "--data", args.data,
        "--out", args.out,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--grad-accum", str(args.grad_accum),
        "--max-seq", str(args.max_seq),
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
    ]
    if args.max_steps > 0:
        argv += ["--max-steps", str(args.max_steps)]
    sys.argv = ["train_sft", *argv]
    train_sft.main()
    return 0


def cmd_train_vla(args) -> int:
    from . import train_sft_vla
    argv = [
        "--model-dir", args.model_dir,
        "--data", args.data,
        "--out", args.out,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--grad-accum", str(args.grad_accum),
        "--max-seq", str(args.max_seq),
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
    ]
    if args.max_steps > 0:
        argv += ["--max-steps", str(args.max_steps)]
    sys.argv = ["train_sft_vla", *argv]
    train_sft_vla.main()
    return 0


def cmd_smoke(args) -> int:
    _banner("SMOKE TEST")
    driver = PopOSDriver()
    planner = Planner()
    grounder = Grounder()
    print("loading planner (Qwen2.5-VL-3B NF4) ...")
    planner.load()
    print("loading grounder (UI-TARS-2B NF4) ...")
    grounder.load()
    print("synthetic 512x512 random RGB image ...")
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)
    img = Image.fromarray((rng.random((512, 512, 3)) * 255).astype("uint8"))
    ctx = OSContextCollector().snapshot(screen_size=driver.screen_size)
    plan = planner.plan("click the centre of the image", img, ctx)
    print(f"  planner raw: {plan.raw[:200]!r}")
    print(f"  todo items:  {len(plan.todo)}")
    for t in plan.todo:
        print(f"    - {t.action!r} target={t.target!r} check={t.check!r}")
    if plan.todo:
        action = grounder.step(plan.todo[0], img)
    else:
        action = grounder.step(type("X", (), {"action": "click centre", "target": "", "check": "", "params": {}})(), img)
    print(f"  grounder:    {action}")
    # Round-trip the parser
    print("\nparser sanity:")
    for fake in ["click(123, 456)", "type('hello world')", "hotkey('ctrl', 'c')",
                 "scroll(0, 100)", "wait()", "finished('done')"]:
        a = Grounder._parse(fake)
        print(f"  {fake!r:50s} -> {type(a).__name__} {a}")
    planner.free()
    grounder.free()
    _banner("OK")
    return 0


def cmd_list_tasks(args) -> int:
    rows = list(tasks.iter_tasks())
    print(f"{len(rows)} seed tasks:")
    for t in rows:
        print(f"  [{t.difficulty:6s}] {t.category:10s} {t.text}")
    return 0


def cmd_seed(args) -> int:
    p = tasks.save_seed_jsonl(args.out)
    print(f"wrote {p}")
    return 0


def cmd_snapshot(args) -> int:
    driver = PopOSDriver()
    ctx = OSContextCollector().snapshot(screen_size=driver.screen_size)
    print(ctx.to_prompt())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="cozy-vision")
    ap.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto",
                    help="cuda (default) = both models on the dGPU; "
                         "cpu = both models on CPU RAM (slow but no VRAM); "
                         "auto = try cuda, fall back to cpu if OOM")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="plan + execute a goal end-to-end (alias for task)")
    p_run.add_argument("goal")
    p_run.set_defaults(func=cmd_task)

    p_task = sub.add_parser("task", help="plan + execute one desktop task with live progress")
    p_task.add_argument("goal")
    p_task.set_defaults(func=cmd_task)

    p_session = sub.add_parser("session", help="interactive REPL — type tasks one per line")
    p_session.set_defaults(func=cmd_session)

    p_ask = sub.add_parser("ask", help="visual Q&A about the current screen")
    p_ask.add_argument("question")
    p_ask.set_defaults(func=cmd_ask)

    p_plan = sub.add_parser("plan", help="plan only, no execution")
    p_plan.add_argument("goal")
    p_plan.set_defaults(func=cmd_plan)

    p_col = sub.add_parser("collect", help="collect SFT traces from N tasks")
    p_col.add_argument("--tasks", type=int, default=5)
    p_col.add_argument("--out", default="cozy-vision/data/sft_planner_raw.jsonl")
    p_col.set_defaults(func=cmd_collect)

    p_tr = sub.add_parser("train", help="QLoRA SFT on the planner (VLM)")
    p_tr.add_argument("--model-dir", default="cozy-vision/models/qwen2.5-vl-3b")
    p_tr.add_argument("--data", default="cozy-vision/data/sft_planner.jsonl")
    p_tr.add_argument("--out", default="cozy-vision/checkpoints/planner-lora")
    p_tr.add_argument("--epochs", type=int, default=1)
    p_tr.add_argument("--lr", type=float, default=2e-4)
    p_tr.add_argument("--grad-accum", type=int, default=8)
    p_tr.add_argument("--max-seq", type=int, default=1024)
    p_tr.add_argument("--lora-r", type=int, default=16)
    p_tr.add_argument("--lora-alpha", type=int, default=32)
    p_tr.add_argument("--max-steps", type=int, default=-1)
    p_tr.set_defaults(func=cmd_train)

    p_trv = sub.add_parser("train-vla", help="QLoRA SFT on the VLA (grounder)")
    p_trv.add_argument("--model-dir", default="cozy-vision/models/ui-tars-2b-sft")
    p_trv.add_argument("--data", default="cozy-vision/data/sft_vla.jsonl")
    p_trv.add_argument("--out", default="cozy-vision/checkpoints/vla-lora")
    p_trv.add_argument("--epochs", type=int, default=1)
    p_trv.add_argument("--lr", type=float, default=2e-4)
    p_trv.add_argument("--grad-accum", type=int, default=8)
    p_trv.add_argument("--max-seq", type=int, default=1024)
    p_trv.add_argument("--lora-r", type=int, default=16)
    p_trv.add_argument("--lora-alpha", type=int, default=32)
    p_trv.add_argument("--max-steps", type=int, default=-1)
    p_trv.set_defaults(func=cmd_train_vla)

    p_sm = sub.add_parser("smoke", help="load both models + run synthetic inference")
    p_sm.set_defaults(func=cmd_smoke)

    p_ls = sub.add_parser("list-tasks", help="list seed tasks")
    p_ls.set_defaults(func=cmd_list_tasks)

    p_sd = sub.add_parser("seed", help="dump seed task JSONL")
    p_sd.add_argument("--out", default="cozy-vision/data/tasks_seed.jsonl")
    p_sd.set_defaults(func=cmd_seed)

    p_sn = sub.add_parser("snapshot", help="print the live OS context")
    p_sn.set_defaults(func=cmd_snapshot)

    args = ap.parse_args()
    if args.device != "cuda":
        os.environ["COZY_VISION_DEVICE"] = args.device
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
