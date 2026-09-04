#!/usr/bin/env python3
"""Cozy STT — one-command personalization for new users.

    python3 scripts/train_me.py

Walks you through: environment check -> asset download -> recording your
voice (incl. Hinglish) -> training (live loss) -> evaluation -> export.
Safe to re-run: finished steps are detected and skipped.

Options:
    --skip-record    use existing recordings only
    --skip-baseline  don't measure the stock model first
    --epochs N       training epochs (default 4)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"
VENV = ROOT / ".venv"


def banner(msg):
    print(f"\n{'='*66}\n  {msg}\n{'='*66}", flush=True)


def step_ok(name):
    print(f"  ✔ {name}", flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=str(ROOT), **kw)


# ----------------------------------------------------------------- steps
def check_env():
    banner("STEP 1/7 · Environment")
    ok = True
    for tool in ("ffmpeg", "arecord"):
        if shutil.which(tool) is None:
            print(f"  ✖ missing: {tool}  (Arch: sudo pacman -S ffmpeg alsa-utils  |  Debian/Ubuntu: sudo apt install ffmpeg alsa-utils)")
            ok = False
        else:
            step_ok(tool)
    if not VENV.exists():
        print("  · creating .venv (reuses your system CUDA torch) ...")
        r = run(["uv", "venv", str(VENV), "--system-site-packages",
                 "--python", "python3"])
        if r.returncode != 0:
            r = run([sys.executable, "-m", "venv", str(VENV),
                     "--system-site-packages"])
        run([str(VENV / "bin" / "pip"), "install", "-q", "--upgrade", "pip"])
    step_ok(".venv present")
    need = ["transformers", "peft", "datasets", "soundfile", "librosa",
            "jiwer", "accelerate", "silero-vad", "sounddevice"]
    missing = []
    for mod in need:
        r = run([str(PY), "-c", f"import {mod.split('-')[0]}"],
                capture_output=True)
        if r.returncode != 0:
            missing.append(mod)
    if missing:
        print(f"  · installing {missing} ...")
        r = run(["uv", "pip", "install", "--python", str(PY)] +
                ["transformers", "datasets", "accelerate", "peft", "jiwer",
                 "soundfile", "librosa", "ctranslate2", "faster-whisper",
                 "silero-vad", "sounddevice", "rich"] + missing,
                capture_output=True)
        if r.returncode != 0:
            print(r.stderr.decode()[-800:])
            sys.exit("dependency install failed - see above")
    step_ok("python deps")
    gpu = run([str(PY), "-c",
               "import torch;assert torch.cuda.is_available()"],
              capture_output=True)
    if gpu.returncode != 0:
        sys.exit("  ✖ CUDA GPU not visible to torch - cannot train here")
    step_ok("CUDA GPU")
    if not ok:
        sys.exit("fix missing system tools and re-run")


def assets():
    banner("STEP 2/7 · Base model + Indian-English corpus (first run ~5 GB)")
    env = dict(os.environ, HF_HOME=str(ROOT / ".hf_cache"))
    marker_m = ROOT / ".hf_cache" / ".base_done"
    if not marker_m.exists():
        run([str(PY), "-c",
             "from common import *;"
             "from huggingface_hub import snapshot_download as s;"
             "print(s('openai/whisper-small', allow_patterns=['*.json','*.txt',"
             "'*.model','pytorch_model.bin']))"],
            env=env).check_returncode()
        marker_m.touch()
    step_ok("whisper-small cached")
    if not (ROOT / "data" / "cv_indian" / ".done").exists():
        run([str(PY), "scripts/build_indian_corpus.py"], env=env)\
            .check_returncode()
    step_ok("Indian-English corpus ready")


def record(skip):
    banner("STEP 3/7 · Record YOUR voice (Hinglish included)")
    if skip:
        n = sum(1 for _ in RECORDINGS_DIR.glob("session_*/*.wav"))
        print(f"  skipped ({n} clips exist)")
        return
    while True:
        run([sys.executable, "scripts/record_voice.py"])
        done = all_sessions_complete()
        if done or input("Record more sessions later? [y/N] ").strip().lower() == "n":
            break


def all_sessions_complete():
    data = json.loads((ROOT / "data" / "prompts.json").read_text())
    rec = ROOT / "recordings"
    for s in data["sessions"]:
        wavs = list((rec / f"session_{s['id']}").glob("*.wav")) \
            if (rec / f"session_{s['id']}").exists() else []
        if len(wavs) < len(s["lines"]):
            return False
    return True


def process_audio():
    banner("STEP 4/7 · Audio processing (silence strip + word alignment)")
    run([sys.executable, "scripts/clean_recordings.py"]).check_returncode()
    run([sys.executable, "scripts/align_words.py"]).check_returncode()
    run([str(PY), "scripts/prepare_data.py"]).check_returncode()
    step_ok("manifests built")


def baseline():
    banner("STEP 5/7 · Stock-model baseline WER (~4 min)")
    run([str(PY), "scripts/baseline_eval.py", "--tag", "baseline"])


def train(epochs):
    banner(f"STEP 6/7 · LoRA finetuning ({epochs} epochs — watch the loss!)")
    out = str(ROOT / "checkpoints" / "lora_personal")
    r = run([str(PY), "scripts/train_lora.py", "--epochs", str(epochs),
             "--lr", "1e-4", "--eval-steps", "50", "--out", out])
    if r.returncode != 0:
        sys.exit("training failed - scroll up for the traceback")


def export_and_report():
    banner("STEP 7/7 · Export + report")
    adapter = str(ROOT / "checkpoints" / "lora_personal" / "adapter")
    r = run([str(PY), "scripts/export_overlay.py", "--adapter", adapter,
             "--hf-out", str(ROOT / "output" / "hf_finetuned"),
             "--ct2-out", str(ROOT / "output" / "cozy_stt_v1_ct2_int8")])
    if r.returncode != 0:
        print("  · CT2 fast-export unavailable (known quirk) - "
              "the transformers model still works everywhere.")
    print("""
  ─────────────────────────────────────────────────────
  YOUR MODEL IS READY 🎉
    live mic :  .venv/bin/python scripts/tui.py
    files    :  .venv/bin/python scripts/infer.py --wav x.wav
  Accuracy engine (Hinglish): stt-finetune/output/hf_finetuned
  Fast engine (English)     : output/cozy_stt_v1_ct2_int8
  ─────────────────────────────────────────────────────""")


RECORDINGS_DIR = ROOT / "recordings"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-record", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()

    os.chdir(ROOT)
    check_env()
    assets()
    record(args.skip_record)
    process_audio()
    if not args.skip_baseline:
        baseline()
    train(args.epochs)
    export_and_report()
