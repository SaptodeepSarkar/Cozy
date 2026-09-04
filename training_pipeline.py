#!/usr/bin/env python3
"""Reproducible Cozy training pipeline.

One command coordinates data validation, LLM SFT → RLVR → DPO, STT LoRA,
exports, and exact benchmarks. Every stage is logged and marked complete only
after a zero exit status, so `--resume` is safe after an interrupted run.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "artifacts" / "training_runs"


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def gpu_info() -> dict[str, object]:
    try:
        out = subprocess.check_output([python_for("assistant"), "-c", (
            "import json,torch; print(json.dumps({'available':torch.cuda.is_available(),"
            "'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
            "'bf16':torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}))"
        )], cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=15)
        return json.loads(out.strip().splitlines()[-1])
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(path)


def python_for(folder: str) -> str:
    candidate = ROOT / folder / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def stage_commands(profile: str, components: set[str]) -> list[tuple[str, list[str]]]:
    assistant = python_for("assistant")
    stt = python_for("stt-finetune")
    smoke = profile == "smoke"
    llm_sft = [assistant, "assistant/sft_qwen.py", "--epochs", "1" if smoke else ("5" if profile == "quality" else "3"), "--workers", "1" if smoke else "4"]
    if smoke:
        llm_sft += ["--max-steps", "2", "--eval-steps", "2", "--batch-size", "1", "--grad-accum", "1"]
    stt_out = ROOT / "stt-finetune" / "checkpoints" / "lora_cozy_pipeline"
    stt_sft = [stt, "stt-finetune/scripts/train_lora.py", "--epochs", "1" if smoke else ("5" if profile == "quality" else "3"), "--workers", "0" if smoke else "2", "--out", str(stt_out)]
    if smoke:
        stt_sft += ["--max-steps", "2", "--batch-size", "1", "--grad-accum", "1", "--eval-steps", "2"]
    commands: list[tuple[str, list[str]]] = []
    if "llm" in components:
        commands += [
            ("llm-data", [assistant, "assistant/make_dataset.py", "--val-fraction", "0.10"]),
            ("llm-sft", llm_sft),
            ("llm-rlvr", [assistant, "assistant/rlvr.py", "--limit", "4" if smoke else "0"]),
            ("llm-dpo", [assistant, "assistant/dpo_light.py", "--epochs", "1" if smoke else "2"]),
            ("llm-benchmark", [assistant, "models/benchmarks/eval_llm.py", "--limit", "4" if smoke else "0", "--model", f"sft={ROOT / 'assistant/model/cozy-llm-v1'}"]),
        ]
    if "stt" in components:
        commands += [
            ("stt-data", [stt, "stt-finetune/scripts/prepare_data.py"]),
            ("stt-baseline", [stt, "stt-finetune/scripts/baseline_eval.py", "--limit", "2" if smoke else "0", "--tag", "baseline"]),
            ("stt-sft", stt_sft),
            ("stt-export", [stt, "stt-finetune/scripts/export_overlay.py", "--adapter", str(stt_out / "adapter")]),
            ("stt-benchmark", [stt, "stt-finetune/scripts/baseline_eval.py", "--hf", str(ROOT / "stt-finetune/output/hf_finetuned_v4layout"), "--limit", "2" if smoke else "0", "--tag", "cozy-lora"]),
        ]
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "standard", "quality"), default="standard")
    parser.add_argument("--components", default="llm,stt", help="comma-separated: llm, stt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--from-stage", metavar="NAME")
    parser.add_argument("--only", metavar="NAME", help="run one stage (for debugging)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    components = {item.strip() for item in args.components.split(",") if item.strip()}
    unknown = components - {"llm", "stt"}
    if unknown:
        parser.error(f"unknown components: {', '.join(sorted(unknown))}")
    commands = stage_commands(args.profile, components)
    if args.only:
        commands = [item for item in commands if item[0] == args.only]
        if not commands:
            parser.error(f"unknown stage {args.only!r}")
    if args.from_stage:
        names = [name for name, _ in commands]
        if args.from_stage not in names:
            parser.error(f"unknown stage {args.from_stage!r}")
        commands = commands[names.index(args.from_stage):]

    run_dir = RUNS / args.run_id
    state_path = run_dir / "state.json"
    old = json.loads(state_path.read_text()) if args.resume and state_path.exists() else {"stages": {}}
    manifest = {"run_id": args.run_id, "profile": args.profile, "components": sorted(components),
                "revision": git_revision(), "python": platform.python_version(), "gpu": gpu_info(),
                "started_at": old.get("started_at", time.time())}
    atomic_json(run_dir / "manifest.json", manifest)
    if args.dry_run:
        for name, command in commands:
            print(name, " ".join(command))
        return 0
    if not manifest["gpu"].get("available"):
        print("CUDA is unavailable in the selected assistant environment; refusing to start training.", file=sys.stderr)
        return 2
    state = {"stages": old.get("stages", {}), "manifest": manifest}
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    for name, command in commands:
        if args.resume and state["stages"].get(name, {}).get("status") == "done":
            print(f"[skip] {name} (resume)")
            continue
        log_path = run_dir / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {name} ===\n$ {' '.join(command)}", flush=True)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            code = process.wait()
        entry = {"status": "done" if code == 0 else "failed", "exit_code": code,
                 "elapsed_s": round(time.perf_counter() - started, 2), "log": str(log_path)}
        state["stages"][name] = entry
        atomic_json(state_path, state)
        if code != 0:
            print(f"[stop] {name} failed; log: {log_path}", file=sys.stderr)
            return code
    state["finished_at"] = time.time()
    atomic_json(state_path, state)
    print(f"\nPipeline complete. Run manifest: {run_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
