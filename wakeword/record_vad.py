"""Record wake-word samples with Silero VAD - only saves speech parts.

Usage:
    python record_vad.py positive           # record "hey cozy" / "cozy" takes
    python record_vad.py positive --bare    # record bare "cozy" only
    python record_vad.py similar --word rosy
    python record_vad.py auto               # auto-cycle through wake + lookalikes

Press Enter to start a take. Speak. Recording stops automatically
when Silero VAD detects ~1.2s of silence after your voice.
"""
from __future__ import annotations

import argparse
import sys
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

HERE = Path(__file__).resolve().parent
SR = 16000
CHANNELS = 1
DTYPE = "int16"
BLOCK = 512  # 32ms frames - required for Silero VAD at 16kHz
VAD_THRESHOLD = 0.5
MIN_SPEECH_SEC = 0.3    # ignore blips shorter than this
SILENCE_END_SEC = 1.2   # stop recording after this much post-speech silence
PRE_SPEECH_PAD = 0.3    # keep this much audio before speech onset
MAX_DURATION = 6.0      # hard cap per take


def load_vad():
    model, _ = torch.hub.load(
        "snakers4/silero-vad", "silero_vad",
        trust_repo=True, force_reload=False)
    return model


def next_index(out_dir: Path, prefix: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    used = {int(p.stem.split("_")[-1]) for p in out_dir.glob(prefix + "_*.wav")}
    i = 1
    while i in used:
        i += 1
    return i


def save_wav(path: Path, pcm: np.ndarray) -> None:
    pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


def take(vad, out_dir: Path, prefix: str, prompt: str) -> Path | None:
    idx = next_index(out_dir, prefix)
    out_path = out_dir / f"{prefix}_{idx:03d}.wav"

    print(f"\n>>> {prompt}")
    print(f"    saving to: {out_path.name}")
    input("    press Enter when ready, then speak... ")

    pre_buffer: deque = deque(maxlen=int(PRE_SPEECH_PAD * SR / BLOCK))
    frames: list[np.ndarray] = []
    started = False
    silence_run = 0.0
    speech_run = 0.0
    started_at = 0.0

    def callback(indata, _frames, _t, _status):
        nonlocal silence_run, started, speech_run
        block = indata[:, 0].copy()
        # Silero wants float32 in [-1, 1]
        x = torch.from_numpy(block.astype(np.float32) / 32768.0)
        prob = float(vad(x, SR).item())

        if not started:
            pre_buffer.append(block)
            if prob >= VAD_THRESHOLD:
                started = True
                speech_run = MIN_SPEECH_SEC
                frames.extend(pre_buffer)
                pre_buffer.clear()
                return
        else:
            frames.append(block)
            if prob >= VAD_THRESHOLD:
                speech_run += BLOCK / SR
                silence_run = 0.0
            else:
                silence_run += BLOCK / SR

    with sd.InputStream(samplerate=SR, channels=CHANNELS, dtype=DTYPE,
                        blocksize=BLOCK, callback=callback):
        started_at = time.time()
        while True:
            time.sleep(0.05)
            if not started and (time.time() - started_at) > 0.5:
                pass
            if started and silence_run >= SILENCE_END_SEC:
                break
            if time.time() - started_at > MAX_DURATION:
                print("    (max duration, stopping)")
                break

    if not frames:
        print("    no speech detected, skipped")
        return None
    pcm = np.concatenate(frames)
    dur = len(pcm) / SR
    if dur < MIN_SPEECH_SEC:
        print(f"    too short ({dur:.2f}s), skipped")
        return None

    peak = int(np.abs(pcm).max())
    if peak > 32000:
        print(f"    WARNING: clipping (peak {peak}), consider lower mic gain")
    elif peak < 700:
        print(f"    WARNING: very quiet (peak {peak}), speak louder")

    save_wav(out_path, pcm)
    print(f"    saved {dur:.2f}s, peak={peak}")
    return out_path


POSITIVE_PROMPTS = [
    "hey cozy", "cozy",
    "hey cozy open firefox",
    "hey cozy set volume to fifty",
]

LOOKALIKE_PROMPTS = {
    "rosy": ["hey rosy", "rosy"],
    "nosy": ["hey nosy", "nosy"],
    "josie": ["hey josie", "josie"],
    "noisy": ["hey noisy", "noisy"],
    "dozy": ["hey dozy", "dozy"],
    "cosy": ["hey cosy", "cosy"],
    "cozey": ["hey cozey", "cozey"],
    "coz": ["hey coz", "coz"],
    "posey": ["hey posey", "posey"],
    "cosi": ["hey cosi", "cosi"],
    "osie": ["hey osie", "osie"],
    "frosty": ["hey frosty", "frosty"],
    "toasty": ["hey toasty", "toasty"],
    "most": ["hey most", "most"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["positive", "similar", "auto"])
    ap.add_argument("--bare", action="store_true",
                    help="positive mode: record bare 'cozy' only")
    ap.add_argument("--word", help="similar mode: which lookalike word")
    ap.add_argument("--count", type=int, default=20,
                    help="auto mode: target count per category")
    args = ap.parse_args()

    print("loading Silero VAD...")
    vad = load_vad()
    print("ready\n")

    if args.mode == "positive":
        out_dir = HERE / "data" / "cozy"
        prompts = ["cozy"] if args.bare else POSITIVE_PROMPTS
        prefix = "bare" if args.bare else "recording"
        while True:
            prompt = prompts[len(list(out_dir.glob(prefix + "_*.wav"))) % len(prompts)]
            take(vad, out_dir, prefix, prompt)
            if input("    another? (y/n) ").lower().strip() != "y":
                break

    elif args.mode == "similar":
        if not args.word or args.word not in LOOKALIKE_PROMPTS:
            print("pick a word:", ", ".join(LOOKALIKE_PROMPTS.keys()))
            sys.exit(1)
        out_dir = HERE / "data" / "similar"
        prefix = "recording_hey_" + args.word
        while True:
            take(vad, out_dir, prefix, args.word)
            if input("    another? (y/n) ").lower().strip() != "y":
                break

    elif args.mode == "auto":
        # record N positive + N lookalikes in a cycle
        for word in [None] + list(LOOKALIKE_PROMPTS.keys()):
            target = args.count
            while True:
                if word is None:
                    out_dir = HERE / "data" / "cozy"
                    n = len(list(out_dir.glob("recording_*.wav")))
                    if n >= target:
                        break
                    prompt = POSITIVE_PROMPTS[n % len(POSITIVE_PROMPTS)]
                    prefix = "recording"
                else:
                    out_dir = HERE / "data" / "similar"
                    n = len(list(out_dir.glob(f"recording_hey_{word}_*.wav")))
                    if n >= target:
                        break
                    prompt = LOOKALIKE_PROMPTS[word][n % 2]
                    prefix = f"recording_hey_{word}"
                print(f"\n=== {word or 'positive'} ({n+1}/{target}) ===")
                take(vad, out_dir, prefix, prompt)


if __name__ == "__main__":
    main()
