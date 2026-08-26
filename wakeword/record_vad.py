"""Record 'hey cozy' / 'cozy' / lookalike takes with Silero VAD + energy filter.
Only the voiced portion is kept - silence before, between, and after is stripped live.

Usage:
  python record_vad.py hey_cozy           # "hey cozy" / "hey cosy" takes
  python record_vad.py hey_cozy --bare    # bare "cozy" only
  python record_vad.py lookalike --word rosy
  python record_vad.py lookalike --word nosy
  python record_vad.py lookalike --word josie
  python record_vad.py lookalike --word noisy
  python record_vad.py lookalike --word dozy
  python record_vad.py lookalike --word cosy

Press Enter to start a take. Speak. Recording stops automatically when VAD
detects ~1.2s of post-speech silence.
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
BLOCK = 512
VAD_THRESHOLD = 0.5
ENERGY_FLOOR = 200
MIN_SPEECH_SEC = 0.3
SILENCE_END_SEC = 1.2
PRE_SPEECH_PAD = 0.3
MAX_DURATION = 6.0


def load_vad():
    return torch.hub.load(
        "snakers4/silero-vad", "silero_vad",
        trust_repo=True, force_reload=False)[0]


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


def is_voiced(block_pcm: np.ndarray, block_f: torch.Tensor, vad) -> bool:
    prob = float(vad(block_f, SR).item())
    energy = float(np.abs(block_pcm).max())
    return prob >= VAD_THRESHOLD or energy >= ENERGY_FLOOR


def take(vad, out_dir: Path, prefix: str, prompt: str) -> Path | None:
    idx = next_index(out_dir, prefix)
    out_path = out_dir / f"{prefix}_{idx:03d}.wav"

    print(f"\n>>> {prompt}")
    print(f"    -> {out_path.name}")
    input("    press Enter, then speak... ")

    pre_buffer: deque = deque(maxlen=int(PRE_SPEECH_PAD * SR / BLOCK))
    frames: list[np.ndarray] = []
    started = False
    silence_run = 0.0
    started_at = time.time()

    def callback(indata, _frames, _t, _status):
        nonlocal silence_run, started
        block = indata[:, 0].copy()
        block_f = torch.from_numpy(block.astype(np.float32) / 32768.0)
        voiced = is_voiced(block, block_f, vad)

        if not started:
            pre_buffer.append(block)
            if voiced:
                started = True
                frames.extend(pre_buffer)
                pre_buffer.clear()
        else:
            frames.append(block)
            if voiced:
                silence_run = 0.0
            else:
                silence_run += BLOCK / SR

    with sd.InputStream(samplerate=SR, channels=CHANNELS, dtype=DTYPE,
                        blocksize=BLOCK, callback=callback):
        while True:
            time.sleep(0.05)
            if started and silence_run >= SILENCE_END_SEC:
                break
            if time.time() - started_at > MAX_DURATION:
                print("    (max duration reached)")
                break

    if not frames:
        print("    no voice detected, skipped")
        return None
    pcm = np.concatenate(frames)
    dur = len(pcm) / SR
    if dur < MIN_SPEECH_SEC:
        print(f"    too short ({dur:.2f}s), skipped")
        return None

    peak = int(np.abs(pcm).max())
    if peak > 32000:
        print(f"    ! clipping (peak {peak}) - lower mic gain")
    elif peak < 700:
        print(f"    ! very quiet (peak {peak}) - speak louder")

    save_wav(out_path, pcm)
    print(f"    saved {dur:.2f}s, peak={peak}")
    return out_path


LOOKALIKES = {
    "rosy": "hey rosy",
    "nosy": "hey nosy",
    "josie": "hey josie",
    "noisy": "hey noisy",
    "dozy": "hey dozy",
    "cosy": "hey cosy",
    "cozey": "hey cozey",
    "coz": "hey coz",
    "frosty": "hey frosty",
    "toasty": "hey toasty",
}

HEY_COZY_PROMPTS = [
    "hey cozy",
    "hey cosy",
    "hey cozy",
    "cozy",
    "hey kozie",
    "hey Cozy",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["hey_cozy", "lookalike"])
    ap.add_argument("--bare", action="store_true",
                    help="hey_cozy mode: record bare 'cozy' only")
    ap.add_argument("--word", required=False,
                    help="lookalike mode: which lookalike word")
    args = ap.parse_args()

    print("loading Silero VAD...")
    vad = load_vad()
    print("ready\n")

    if args.mode == "hey_cozy":
        out_dir = HERE / "data" / "cozy"
        prefix = "bare" if args.bare else "recording"
        prompts = ["cozy"] if args.bare else HEY_COZY_PROMPTS
        while True:
            n = len(list(out_dir.glob(prefix + "_*.wav")))
            prompt = prompts[n % len(prompts)]
            take(vad, out_dir, prefix, prompt)
            if input("    another? (y/n) ").lower().strip() != "y":
                break

    elif args.mode == "lookalike":
        if not args.word or args.word not in LOOKALIKES:
            print("pick a word:", ", ".join(LOOKALIKES.keys()))
            sys.exit(1)
        out_dir = HERE / "data" / "similar"
        prefix = f"recording_hey_{args.word}"
        while True:
            take(vad, out_dir, prefix, LOOKALIKES[args.word])
            if input("    another? (y/n) ").lower().strip() != "y":
                break


if __name__ == "__main__":
    main()
