#!/usr/bin/env python
"""Download the base STT model + Indian English corpus into stt-finetune/.hf_cache
and data/fleurs (all project-local, nothing outside the folder).

Run:  source env.sh && .venv/bin/python scripts/download_assets.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_MODEL, FLEURS_DIR, ROOT  # noqa: E402


def download_model():
    from huggingface_hub import snapshot_download
    print(f"== Downloading {BASE_MODEL} ...")
    p = snapshot_download(BASE_MODEL, allow_patterns=["*.json", "*.txt", "*.model", "pytorch_model.bin"])
    print(f"   -> {p}")


def download_fleurs():
    from datasets import load_dataset
    if FLEURS_DIR.exists() and (FLEURS_DIR / "dataset_info.json").exists():
        print(f"== FLEURS en_in already present at {FLEURS_DIR}, skipping")
        return
    print("== Downloading Google FLEURS 'en_in' (Indian English, ~1-2 GB) ...")
    ds = load_dataset("google/fleurs", "en_in")
    for split in ("train", "validation", "test"):
        print(f"   {split}: {len(ds[split])} utterances")
    FLEURS_DIR.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(FLEURS_DIR))
    print(f"   -> saved to {FLEURS_DIR}")


if __name__ == "__main__":
    download_model()
    download_fleurs()
    print("\nAll assets downloaded. Total size:")
    import subprocess
    subprocess.run(["du", "-sh", str(ROOT / ".hf_cache"), str(ROOT / "data" / "fleurs")])
