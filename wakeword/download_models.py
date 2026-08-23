#!/usr/bin/env python3
"""Downloads every model the Cozy wake-word pipeline needs.

  * Piper TTS voices (US / UK / IN accents)        -> work/models/voices/
  * LibriTTS-R multi-speaker sample generator      -> work/models/
  * openWakeWord feature models (melspectrogram,
    embedding_model)                               -> openWakeWord cache

Everything is cached under wakeword/work/ (git-ignored). Re-running is safe:
files that already exist are skipped.

Usage:
    python download_models.py                    # everything
    python download_models.py --only-openwakeword
    python download_models.py --no-multispeaker  # voices only (smaller)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
VOICES_DIR = WORK / "models" / "voices"

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en"

# voice file name          (hf locale, speaker, quality)
PIPER_VOICES = {
    "en_US-lessac-medium": ("en_US", "lessac", "medium"),
    "en_US-amy-medium": ("en_US", "amy", "medium"),
    "en_US-ryan-medium": ("en_US", "ryan", "medium"),
    "en_GB-jenny-medium": ("en_GB", "jenny", "medium"),
    "en_GB-alan-medium": ("en_GB", "alan", "medium"),
    "en_GB-northern_english_male-medium": ("en_GB", "northern_english_male", "medium"),
    "en_IN-neupane-medium": ("en_IN", "neupane", "medium"),  # optional; may 404
}

MULTISPEAKER_NAME = "en_US-libritts_r-medium.pt"
MULTISPEAKER_URL = (
    "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/"
    + MULTISPEAKER_NAME
)


def voice_url(locale: str, speaker: str, quality: str, voice: str) -> str:
    return f"{HF_BASE}/{locale}/{speaker}/{quality}/{voice}.onnx"


def download(url: str, dest: Path, optional: bool = False) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        mb = dest.stat().st_size / 1e6
        print(f"  = {dest.relative_to(HERE)} ({mb:.0f} MB, cached)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as fh, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=f"  + {dest.name}",
                leave=False,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as exc:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        if optional:
            print(f"  ! skipped {dest.name}: {exc}")
            return False
        msg = ("[download_models] required download failed: "
               + str(url) + " | " + str(exc))
        raise SystemExit(msg) from None


def fetch_openwakeword_features() -> None:
    try:
        from openwakeword.utils import download_models as oww_download
    except ImportError as exc:
        raise SystemExit(
            "[download_models] openWakeWord is not installed. "
            "Run bash setup.sh first."
        ) from exc
    try:
        oww_download(model_names=["melspectrogram", "embedding_model"])
    except TypeError:  # older signature
        oww_download()
    print("  = openWakeWord feature models ready")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-openwakeword", action="store_true",
                        help="only fetch the openWakeWord feature models")
    parser.add_argument("--no-multispeaker", action="store_true",
                        help="skip the LibriTTS-R multi-speaker generator")
    args = parser.parse_args()

    if not args.only_openwakeword:
        print("[1/3] Piper TTS voices")
        for voice, meta in PIPER_VOICES.items():
            url = voice_url(*meta, voice)
            if download(url, VOICES_DIR / f"{voice}.onnx", optional=True):
                download(url + ".json",
                         VOICES_DIR / f"{voice}.onnx.json", optional=True)

        if args.no_multispeaker:
            print("[2/3] LibriTTS-R multi-speaker generator: SKIPPED "
                  "(--no-multispeaker)")
        else:
            print("[2/3] LibriTTS-R multi-speaker generator (~250 MB, gives "
                  "900+ speaker variety)")
            download(MULTISPEAKER_URL, WORK / "models" / MULTISPEAKER_NAME,
                     optional=True)

    print("[3/3] openWakeWord feature models")
    fetch_openwakeword_features()

    print("\nAll models ready. Next: python generate_data.py --mode smoke")


if __name__ == "__main__":
    main()
