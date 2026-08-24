#!/usr/bin/env python3
"""Record YOUR voice for Cozy - the wake word AND its lookalikes.

Wake word (positives, saved to data/cozy):
    python record_samples.py                     # 15 clips
    python record_samples.py --num 30

Similar-sounding words (hard negatives, saved to data/similar):
    python record_samples.py --similar           # 6 takes of every word
    python record_samples.py --similar --per-word 10

The similar-word list lives in config.yaml (similar_words), so you can add
new lookalikes any time and rerun - already-recorded words are skipped.
Clips are stored as 16 kHz mono WAV automatically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError as exc:
    raise SystemExit(
        "Missing microphone dependencies. Inside wakeword/:"
        "  source .venv/bin/activate && pip install sounddevice soundfile"
        "  (Linux OS package: sudo apt install libportaudio2)"
    ) from exc

HERE = Path(__file__).resolve().parent
SR = 16000


def trim_silence(pcm, floor=250):
    mag = np.abs(pcm).astype(np.int32)
    idx = np.where(mag > floor)[0]
    if idx.size == 0:
        return pcm
    pad = SR // 4
    start = max(0, int(idx[0]) - pad)
    end = min(len(pcm), int(idx[-1]) + pad)
    return pcm[start:end]


def record_clip(seconds):
    frames = int(seconds * SR)
    pcm = sd.rec(frames, samplerate=SR, channels=1, dtype="int16")
    sd.wait()
    return trim_silence(pcm[:, 0])


def slug(text):
    clean = "".join(ch if ch.isalnum() else "_" for ch in text.lower())
    return clean.strip("_")


def next_index(out_dir, prefix):
    existing = sorted(out_dir.glob(prefix + "_*.wav"))
    return len(existing) + 1


def countdown():
    for tick in (3, 2, 1):
        print("   " + str(tick), flush=True)
    print("   REC...", flush=True)


def run_set(label, out_dir, prefix, total, seconds):
    out_dir.mkdir(parents=True, exist_ok=True)
    i = next_index(out_dir, prefix)
    for n in range(1, total + 1):
        prompt = "[" + label + " " + str(n) + "/" + str(total) + "]"
        input(prompt + " press ENTER, then say it...")
        countdown()
        pcm = record_clip(seconds)
        dest = out_dir / (prefix + "_" + format(i, "03d") + ".wav")
        sf.write(str(dest), pcm, SR, subtype="PCM_16")
        print("   saved " + dest.name + " ("
              + format(pcm.shape[0] / SR, ".2f") + "s)", flush=True)
        i += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num", type=int, default=15,
                        help="clips of the wake word (default 15)")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=None,
                        help="override output folder for wake word clips")
    parser.add_argument("--similar", action="store_true",
                        help="record the similar_words from config.yaml instead")
    parser.add_argument("--per-word", type=int, default=6, dest="per_word",
                        help="takes per similar word (default 6)")
    args = parser.parse_args()

    cfg = yaml.safe_load((HERE / "config.yaml").read_text())

    if args.similar:
        words = list(cfg["similar_words"])
        print("Recording " + str(args.per_word) + " takes of "
              + str(len(words)) + " similar words -> data/similar")
        print("Say each word clearly and naturally.")
        print("(Words come from config.yaml:similar_words)\n")
        for word in words:
            prefix = "recording_" + slug(word)
            out_dir = HERE / "data" / "similar"
            done = next_index(out_dir, prefix) - 1
            remaining = max(0, args.per_word - done)
            if remaining == 0:
                print("[skip] " + word + " - already have "
                      + str(done) + " takes")
                continue
            print("\n== " + word.upper() + " (" + str(remaining)
                  + " to go) ==")
            run_set(label=word, out_dir=out_dir, prefix=prefix,
                    total=remaining, seconds=args.seconds)
        print("\nAll similar words recorded. Retrain so they count:")
        print("  tell the agent: 'retrain cozy'")
    else:
        out_dir = args.out or (HERE / "data" / "cozy")
        phrase = str(cfg["wake_word"]["text"])
        print("Recording " + str(args.num) + " clips of '" + phrase
              + "' -> " + str(out_dir.relative_to(HERE)))
        print("Vary tone, speed and distance from the mic.\n")
        run_set(label=phrase, out_dir=out_dir, prefix="recording",
                total=args.num, seconds=args.seconds)
        print("\nDone! Retrain so the model learns your voice:")
        print("  tell the agent: 'retrain cozy'")


if __name__ == "__main__":
    main()
