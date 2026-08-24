#!/usr/bin/env python
"""Download the base STT model + Google FLEURS 'en_in' (Indian English) archives,
extract them under data/fleurs_raw/, all inside stt-finetune/.

Run:  source env.sh && .venv/bin/python scripts/download_assets.py
"""
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_MODEL, ROOT  # noqa: E402

FLEURS_RAW = ROOT / "data" / "fleurs_raw" / "en_in"
SPLITS = ["train", "validation", "test"]


def download_model():
    from huggingface_hub import snapshot_download
    print(f"== Downloading {BASE_MODEL} ...")
    p = snapshot_download(BASE_MODEL,
                          allow_patterns=["*.json", "*.txt", "*.model", "*.bin"])
    print(f"   -> {p}")


def download_fleurs():
    from huggingface_hub import hf_hub_download
    FLEURS_RAW.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        marker = FLEURS_RAW / f".{split}_done"
        if marker.exists():
            print(f"== FLEURS {split}: already extracted")
            continue
        fname = f"audio/en_in~{split}.tar.gz"
        print(f"== FLEURS en_in {split}: downloading {fname} ...")
        arch = hf_hub_download(repo_id="google/fleurs", repo_type="dataset",
                               filename=fname)
        print(f"   extracting -> {FLEURS_RAW}")
        with tarfile.open(arch) as tf:
            tf.extractall(FLEURS_RAW)
        marker.touch()
        Path(arch).unlink(missing_ok=True)  # save disk; re-downloadable


if __name__ == "__main__":
    download_model()
    download_fleurs()
    print("\n== Asset sizes inside this project:")
    import subprocess
    subprocess.run(["du", "-sh", str(ROOT / ".hf_cache"), str(ROOT / "data")])
