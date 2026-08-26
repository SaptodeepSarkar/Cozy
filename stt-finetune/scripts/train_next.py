#!/usr/bin/env python3
"""One command: rebuild data -> train v6 -> export -> summarise.

Use this for any major retraining (44-session dataset etc.).
Existing v5 in output/hf_finetuned/ is preserved in v5_backup/.

    python3 scripts/train_next.py --epochs 3

Estimated on RTX 3050 6 GB:
  * 44 sessions x 15 lines x 3 repeats = ~1,900 user samples + 1,425 corpus
    = ~3,350 train rows  ->  ~30 minutes training  +  ~3 min eval
  * training VRAM peak: ~3.5 GB
  * final merged model on disk: 642 MB
  * inference VRAM (hf_fp16):  ~1.2-1.5 GB
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=str(ROOT), **kw)


def banner(t):
    print("\n" + "=" * 66, flush=True)
    print("  " + t, flush=True)
    print("=" * 66, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--user-repeat", type=int, default=3)
    ap.add_argument("--out", default="checkpoints/lora_cozy_v6",
                    help="LoRA checkpoint directory (next version)")
    ap.add_argument("--skip-data", action="store_true")
    args = ap.parse_args()

    t0 = time.time()

    if not args.skip_data:
        banner("STEP 1/3 · rebuild data manifests from all 44 sessions")
        r = run([str(PY), "scripts/prepare_data.py",
                 "--eval-sessions", "6", "--user-repeat", str(args.user_repeat)])
        if r.returncode != 0:
            sys.exit("prepare_data failed")

    banner(f"STEP 2/3 · train v6 (LoRA, {args.epochs} epochs, lr {args.lr})")
    print("  expected: ~30 min on RTX 3050, peak VRAM ~3.5 GB", flush=True)
    print("  per-step loss prints inline, eval WER every "
          f"{args.eval_steps} steps.\n", flush=True)
    r = run([str(PY), "scripts/train_lora.py",
             "--epochs", str(args.epochs),
             "--lr", str(args.lr),
             "--eval-steps", str(args.eval_steps),
             "--out", args.out])
    if r.returncode != 0:
        sys.exit("training failed - scroll up for the traceback")

    banner("STEP 3/3 · export to transformers + sanity check")
    # back up current v5 (if it exists) and promote v6
    cur = ROOT / "output" / "hf_finetuned"
    bak = ROOT / "output" / "hf_finetuned_v5_backup"
    if cur.exists() and not bak.exists():
        print(f"  · backing up current hf_finetuned -> {bak.name}")
        shutil.copytree(cur, bak)
    elif cur.exists() and bak.exists():
        print(f"  · {bak.name} already exists, leaving as-is")

    r = run([str(PY), "scripts/export_overlay.py",
             "--adapter", str(Path(args.out) / "adapter"),
             "--hf-out", str(cur)])
    elapsed = time.time() - t0
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print(f"\n  total wall clock: {mins}m {secs}s")
    print("  · merged model size:", end=" ")
    p = cur / "pytorch_model.bin"
    if p.exists():
        print(f"{p.stat().st_size / 1e6:.0f} MB  (fp16)")
    print("  · inference VRAM (hf_fp16, batch 1):  ~1.2-1.5 GB")
    print("  · test it now:")
    print(f"      {PY} scripts/tui.py")


if __name__ == "__main__":
    main()
