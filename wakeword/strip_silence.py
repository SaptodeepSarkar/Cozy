"""Strip silence from all existing recordings using Silero VAD.

Runs over data/cozy and data/similar, replaces each WAV with a tighter
version containing only voiced regions. Preserves the filename.

Usage:
    python strip_silence.py             # process all data/
    python strip_silence.py data/cozy   # one folder
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
SR = 16000
BLOCK = 512
VAD_THRESHOLD = 0.5
MIN_SPEECH_SEC = 0.3
PAD_BEFORE = 0.2
PAD_AFTER = 0.2


def load_vad():
    return torch.hub.load(
        "snakers4/silero-vad", "silero_vad",
        trust_repo=True, force_reload=False)[0]


def voiced_mask(pcm: np.ndarray, vad) -> np.ndarray:
    x = torch.from_numpy(pcm.astype(np.float32) / 32768.0)
    speech = np.zeros(len(pcm), dtype=bool)
    for i in range(0, len(x) - BLOCK, BLOCK):
        if float(vad(x[i:i + BLOCK], SR).item()) >= VAD_THRESHOLD:
            speech[i:i + BLOCK] = True
    return speech


def trim(pcm: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return pcm
    idx = np.flatnonzero(mask)
    start = max(0, idx[0] - int(PAD_BEFORE * SR))
    end = min(len(pcm), idx[-1] + int(PAD_AFTER * SR))
    return pcm[start:end]


def process_file(path: Path, vad) -> bool:
    pcm, sr = sf.read(str(path), dtype="int16")
    if sr != SR:
        return False
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1).astype(np.int16)
    mask = voiced_mask(pcm, vad)
    out = trim(pcm, mask)
    if len(out) == len(pcm) and (out == pcm).all():
        return False  # no change
    if len(out) / SR < MIN_SPEECH_SEC:
        return False
    sf.write(str(path), out, SR, subtype="PCM_16")
    return True


def main() -> None:
    targets = [Path(a) for a in sys.argv[1:]] or [
        HERE / "data" / "cozy",
        HERE / "data" / "similar",
    ]
    print("loading Silero VAD...")
    vad = load_vad()

    n_total = n_changed = 0
    for folder in targets:
        if not folder.exists():
            print("skip (missing):", folder)
            continue
        for wav in sorted(folder.rglob("*.wav")):
            n_total += 1
            before = wav.stat().st_size
            if process_file(wav, vad):
                after = wav.stat().st_size
                print(f"  trimmed {wav.name}: {before} -> {after} bytes")
                n_changed += 1

    print(f"\nprocessed {n_changed}/{n_total} files")


if __name__ == "__main__":
    main()
