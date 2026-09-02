"""One-shot model downloader.

Downloads the two pretrained checkpoints to ``cozy-vision/models/``::

    ui-tars-2b-sft/         # ByteDance-Seed/UI-TARS-2B-SFT (VLA)
    qwen2.5-vl-3b/          # Qwen/Qwen2.5-VL-3B-Instruct (VLM)

Uses ``aria2c`` for the large safetensors files (4 connections per
file, more reliable on this mirror than 8) and ``huggingface-cli``
for the small config / tokenizer files. Both can be skipped with
``--skip-large`` / ``--skip-small``.

If the model directories already contain the right safetensors, the
script will not re-download them.

Note on the Qwen download: Hugging Face redirects to a CDN
(us.aws.cdn.hf.co). Aria2c with ``-x 8`` is **unreliable** on this
mirror — chunks fail intermittently and the file is truncated
silently. Use ``-x 2 -s 2`` (which is what this script does) or
fall back to ``hf download`` if aria2c fails.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

MODELS_TO_GET = [
    {
        "name": "ui-tars-2b-sft",
        "repo": "ByteDance-Seed/UI-TARS-2B-SFT",
        "large": [
            ("model-00001-of-00002.safetensors", 4.98e9),
            ("model-00002-of-00002.safetensors", 4.79e9),
        ],
        "small": [
            "config.json", "preprocessor_config.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json", "merges.txt",
            "chat_template.json", "special_tokens_map.json",
            "added_tokens.json", "generation_config.json", "README.md",
        ],
        "cdn": "https://huggingface.co",
    },
    {
        "name": "qwen2.5-vl-3b",
        "repo": "Qwen/Qwen2.5-VL-3B-Instruct",
        "large": [
            ("model-00001-of-00002.safetensors", 3.98e9),
            ("model-00002-of-00002.safetensors", 3.53e9),
        ],
        "small": [
            "config.json", "preprocessor_config.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json", "merges.txt",
            "chat_template.json", "special_tokens_map.json",
            "added_tokens.json", "generation_config.json",
            "README.md", "LICENSE", "model.safetensors.index.json",
        ],
        "cdn": "https://huggingface.co",
    },
]


def have_aria2() -> bool:
    return shutil.which("aria2c") is not None


def run_aria2(url: str, dest_dir: Path, dest_name: str, conns: int = 2) -> bool:
    if not have_aria2():
        return False
    cmd = [
        "aria2c", "-x", str(conns), "-s", str(conns), "-k", "1M",
        "--file-allocation=none", "--auto-file-renaming=false",
        "--max-tries=0", "--retry-wait=5", "--console-log-level=error",
        "--timeout=60", "--connect-timeout=30",
        "-d", str(dest_dir), "-o", dest_name, url,
    ]
    print(f"  $ aria2c -x {conns} {dest_name}")
    return subprocess.run(cmd).returncode == 0


def run_hf(repo: str, files, dest_dir: Path) -> bool:
    cmd = ["hf", "download", repo, "--local-dir", str(dest_dir), *files]
    print(f"  $ hf download {repo} ({len(files)} small files)")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-400:])
        return False
    return True


def check_complete(model: dict) -> bool:
    out_dir = MODELS / model["name"]
    for fname, _ in model["large"]:
        if not (out_dir / fname).is_file():
            return False
    for fname in model["small"]:
        if not (out_dir / fname).is_file():
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-large", action="store_true")
    ap.add_argument("--skip-small", action="store_true")
    ap.add_argument("--only", choices=[m["name"] for m in MODELS_TO_GET], default=None)
    ap.add_argument("--connections", type=int, default=2,
                    help="aria2c connections per file (default 2; do not exceed 4 on this mirror)")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    for model in MODELS_TO_GET:
        if args.only and model["name"] != args.only:
            continue
        out_dir = MODELS / model["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {model['name']} ({model['repo']}) ===")
        if check_complete(model):
            print("  already complete, skipping")
            continue

        # Large safetensors via aria2c (HF CDN)
        if not args.skip_large:
            for fname, _ in model["large"]:
                target = out_dir / fname
                if target.is_file() and target.stat().st_size > 1_000_000_000:
                    print(f"  {fname}: already on disk ({target.stat().st_size/1e9:.2f} GB)")
                    continue
                url = f"{model['cdn']}/{model['repo']}/resolve/main/{fname}"
                ok = run_aria2(url, out_dir, fname, conns=args.connections)
                if not ok:
                    print("  aria2c failed, falling back to hf download")
                    subprocess.run(["hf", "download", model["repo"], "--local-dir", str(out_dir), fname])

        # Small files via hf
        if not args.skip_small:
            missing = [f for f in model["small"] if not (out_dir / f).is_file()]
            if missing:
                run_hf(model["repo"], missing, out_dir)

        # Drop any leftover .cache or .aria2
        for leftover in (out_dir / ".cache",):
            if leftover.is_dir():
                shutil.rmtree(leftover)
        for f in out_dir.iterdir():
            if f.name.endswith(".aria2"):
                f.unlink()

        ok = check_complete(model)
        print(f"  complete: {ok}")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
