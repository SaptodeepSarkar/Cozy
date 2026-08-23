#!/usr/bin/env python3
"""Record YOUR voice saying the wake word into data/cozy/.

Examples:
    python record_samples.py                # 15 guided recordings
    python record_samples.py --num 30
    python record_samples.py --seconds 3

Clips are stored as 16 kHz mono WAV - exactly what training expects.
More clips (20+) with varied tone/speed/distance noticeably improve accuracy.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError as exc:
    raise SystemExit(
        "Missing microphone dependencies. Inside wakeword/:\n"
        "  source .venv/bin/activate && pip install sounddevice soundfile\n"
        "(Linux OS package: sudo apt install libportaudio2)"
    ) from exc

HERE = Path(__file__).resolve().parent
SR = 16000


def trim_silence(pcm: np.ndarray, floor: int = 250) -> np.ndarray:
    mag = np.abs(pcm).astype(np.int32)
    idx = np.where(mag > floor)[0]
    if idx.size == 0:
        return pcm
    pad = SR // 4  # keep a little breathing room
    start = max(0, int(idx[0]) - pad)
    end = min(len(pcm), int(idx[-1]) + pad)
    return pcm[start:end]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num", type=int, default=15)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=HERE / "data" / "cozy")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frames = int(args.seconds * SR)
    existing = sorted(args.out.glob("recording_*.wav"))
    start_index = len(existing) + 1

    print(f"Recording {args.num} clips -> {args.out}")
    print("Say 'cozy' naturally once the countdown finishes. Ctrl-C to stop.\n")

    for i in range(start_index, start_index + args.num):
        input(f"[{i}] press ENTER, then say 'cozy'...")
        for tick in (3, 2, 1):
            print(f"  {tick}", flush=True)
        print("  ● REC", flush=True)
        pcm = sd.rec(frames, samplerate=SR, channels=1, dtype="int16")
        sd.wait()
        clipped = trim_silence(pcm[:, 0])
        dest = args.out / f"recording_{i:03d}.wav"
        sf.write(dest, clipped, SR, subtype="PCM_16")
        print(f"  saved {dest.name} ({clipped.shape[0] / SR:.2f}s)")

    print(f"\nDone. Now retrain so the model learns your voice:")
    print("  python generate_data.py --mode full && python train_wakeword.py")


if __name__ == "__main__":
    main()
