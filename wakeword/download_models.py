#!/usr/bin/env python3
"""Downloads every model the Cozy wake-word pipeline needs - BULLETPROOF.

Resilience features (so a flaky connection can never break the pipeline):
  * resumable downloads   - partial files are kept as *.part and continued
                            with HTTP Range requests
  * automatic retries     - 6 attempts per URL with exponential backoff
  * mirror fallback       - Hugging Face URLs fall back to COZY_HF_MIRROR
                            (default: https://hf-mirror.com)
  * size sanity checks    - zero-byte or truncated files are retried

Downloads (all cached under wakeword/work/, git-ignored):
  * Piper TTS voices (US / UK / IN accents) -> work/models/voices/
  * LibriTTS-R multi-speaker sample generator -> work/models/
  * openWakeWord feature models (melspectrogram, embedding_model)

Usage:
    python download_models.py                    # everything
    python download_models.py --only-openwakeword
    python download_models.py --no-multispeaker  # voices only (smaller)
"""
from __future__ import annotations

import argparse
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
VOICES_DIR = WORK / "models" / "voices"

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en"
HF_MIRROR = os.environ.get("COZY_HF_MIRROR", "https://hf-mirror.com")

MAX_ATTEMPTS = 6
BACKOFF_BASE = 5      # seconds; doubles each attempt, capped at 60

# voice file name                          (hf locale, speaker, quality)
PIPER_VOICES = {
    "en_US-lessac-medium": ("en_US", "lessac", "medium"),
    "en_US-amy-medium": ("en_US", "amy", "medium"),
    "en_US-ryan-medium": ("en_US", "ryan", "medium"),
    "en_GB-jenny-medium": ("en_GB", "jenny", "medium"),
    "en_GB-alan-medium": ("en_GB", "alan", "medium"),
    "en_GB-northern_english_male-medium": (
        "en_GB", "northern_english_male", "medium"),
    "en_IN-neupane-medium": ("en_IN", "neupane", "medium"),  # may 404
}

MULTISPEAKER_NAME = "en_US-libritts_r-medium.pt"
MULTISPEAKER_URL = (
    "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/"
    + MULTISPEAKER_NAME
)

# piper-sample-generator imports piper_train, which ships only inside the
# piper repository (src/python). We vendor those sources automatically.
VENDOR_DIR = WORK / "vendor"
PIPER_SRC_URL = "https://github.com/rhasspy/piper/archive/refs/heads/master.tar.gz"


def voice_url(locale, speaker, quality, voice):
    return HF_BASE + "/" + locale + "/" + speaker + "/" + quality + "/" \
        + voice + ".onnx"


def hf_candidates(url):
    """Primary URL first, then mirrors for Hugging Face links."""
    urls = [url]
    if url.startswith("https://huggingface.co/") and HF_MIRROR:
        urls.append(url.replace("https://huggingface.co/", HF_MIRROR + "/"))
    return urls


def _mb(n):
    return format(n / 1e6, ".0f") + " MB"


def _download_once(url, part):
    """Streams url into part, resuming from existing bytes when possible."""
    resume_from = part.stat().st_size if part.exists() else 0
    headers = {}
    if resume_from > 0:
        headers["Range"] = "bytes=" + str(resume_from) + "-"
    with requests.get(url, stream=True, timeout=(15, 90),
                      headers=headers) as resp:
        if resume_from > 0 and resp.status_code == 200:
            resume_from = 0          # server ignored Range -> start over
        resp.raise_for_status()
        declared = int(resp.headers.get("content-length", 0)) + resume_from
        mode = "ab" if resume_from > 0 else "wb"
        with open(part, mode) as fh, tqdm(
                total=declared or None,
                initial=resume_from,
                unit="B", unit_scale=True,
                desc="      " + part.name,
                leave=False) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
        size_now = part.stat().st_size
        if declared and size_now < declared:
            raise IOError("truncated: got " + str(size_now) + " of "
                          + str(declared))


def download(urls, dest, optional=False, label=""):
    """Download the first working candidate URL into dest. Resumable."""
    if dest.exists() and dest.stat().st_size > 0:
        print("  = " + label + dest.name + " ("
              + _mb(dest.stat().st_size) + ", cached)")
        return True

    unique_urls = []
    for u in urls:
        if u not in unique_urls:
            unique_urls.append(u)

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last_error = "unknown"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        for url in unique_urls:
            try:
                _download_once(url, part)
                if part.stat().st_size <= 0:
                    raise IOError("empty response")
                os.replace(part, dest)
                print("  + " + label + dest.name + " ("
                      + _mb(dest.stat().st_size) + ")")
                return True
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if part.exists() and part.stat().st_size == 0:
                    part.unlink()
        if attempt < MAX_ATTEMPTS:
            delay = min(60, BACKOFF_BASE * (2 ** (attempt - 1)))
            delay += random.uniform(0, 3)
            print("  .. attempt " + str(attempt) + " failed for "
                  + dest.name + " - retrying in "
                  + format(delay, ".0f") + "s (" + last_error[:80] + ")")
            time.sleep(delay)

    part.unlink(missing_ok=True)
    if optional:
        print("  ! skipped " + dest.name + ": " + last_error[:120])
        return False
    raise SystemExit("[download_models] giving up on " + dest.name
                     + " after " + str(MAX_ATTEMPTS) + " attempts: "
                     + last_error[:200]
                     + " | rerun later - finished parts resume automatically")


def vendor_piper_python():
    """Ensures work/vendor/piper/src/python (with piper_train) exists."""
    target = VENDOR_DIR / "piper" / "src" / "python"
    if (target / "piper_train").is_dir():
        print("  = vendored piper_train sources (cached)")
        return str(target)
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = VENDOR_DIR / "piper_master.tar.gz"
    download([PIPER_SRC_URL], tar_path, label="")
    import tarfile
    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if "/src/python/" in m.name]
        tf.extractall(VENDOR_DIR, members=members, filter="data")
    if not (VENDOR_DIR / "piper-master" / "src" / "python").is_dir():
        raise SystemExit("[download_models] unexpected piper archive layout")
    os.replace(VENDOR_DIR / "piper-master", VENDOR_DIR / "piper")
    print("  = vendored piper_train sources -> " + str(target))
    return str(target)


def fetch_openwakeword_features():
    try:
        import openwakeword
    except ImportError as exc:
        raise SystemExit("[download_models] openWakeWord is not installed. "
                         "Run bash setup.sh first.") from exc

    # 1) skip the network entirely when the files are already on disk
    res = Path(openwakeword.__file__).parent / "resources" / "models"
    needed = ["melspectrogram.onnx", "embedding_model.onnx"]
    if all((res / n).exists() for n in needed):
        print("  = feature models already on disk")
        return

    # 2) otherwise download inside a HARD-TIMEOUT subprocess: a stalled
    #    socket can slow a retry but can never wedge the whole pipeline
    child = (
        "from openwakeword.utils import download_models as d\n"
        "try:\n"
        "    d(model_names=['melspectrogram', 'embedding_model'])\n"
        "except TypeError:\n"
        "    d()\n"
    )
    for attempt in range(1, 4):
        try:
            proc = subprocess.run([sys.executable, "-c", child],
                                  timeout=240)
            if proc.returncode == 0:
                print("  = openWakeWord feature models ready")
                return
            print("  .. downloader exit " + str(proc.returncode)
                  + ", attempt " + str(attempt) + "/3")
        except subprocess.TimeoutExpired:
            print("  .. downloader timed out after 240 s, attempt "
                  + str(attempt) + "/3")
        time.sleep(10 * attempt)
    raise SystemExit("[download_models] could not fetch openWakeWord feature "
                     "models - rerun later; finished parts are cached")


def main():
    socket.setdefaulttimeout(30)  # no network call may ever stall forever
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-openwakeword", action="store_true",
                        help="only fetch the openWakeWord feature models")
    parser.add_argument("--no-multispeaker", action="store_true",
                        help="skip the LibriTTS-R multi-speaker generator")
    args = parser.parse_args()

    if not args.only_openwakeword:
        print("[1/4] piper_train Python sources (vendored via PYTHONPATH)")
        vendor_path = vendor_piper_python()
        (VENDOR_DIR / "PYTHONPATH.txt").write_text(vendor_path)

        print("[2/4] Piper TTS voices")
        for voice, meta in PIPER_VOICES.items():
            url = voice_url(*meta, voice)
            ok = download([url], VOICES_DIR / (voice + ".onnx"),
                          optional=True)
            if ok:
                download([url + ".json"],
                         VOICES_DIR / (voice + ".onnx.json"),
                         optional=True)

        if args.no_multispeaker:
            print("[3/4] LibriTTS-R multi-speaker generator: SKIPPED")
        else:
            print("[3/4] LibriTTS-R multi-speaker generator (~250 MB)")
            download([MULTISPEAKER_URL],
                     WORK / "models" / MULTISPEAKER_NAME,
                     optional=True)

    print("[4/4] openWakeWord feature models")
    fetch_openwakeword_features()

    print("")
    print("All models ready. Next: python generate_data.py --mode smoke")


if __name__ == "__main__":
    main()
