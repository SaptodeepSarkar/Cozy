#!/usr/bin/env python3
"""Re-extract user-voice training data from stt-finetune/recordings/ + old git history.

This script generates:
  wakeword/user_voice/user_cozy/        (32 wavs from old "cozy" recordings)
  wakeword/user_voice/user_pos_stt/      (106 2s windows from STT "cozy" recordings)
  wakeword/user_voice/user_neg_stt/      (2068 2s windows from non-cozy STT recordings)

Source data:
  - stt-finetune/recordings/session_*/*.wav + .txt
  - wakeword/data/cozy/recording_*.wav (extracted from old git commit e2a4dda)

Re-run with:
    cd wakeword
    .venv/bin/python extract_user_voice.py
"""
from __future__ import annotations
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

WW = Path(__file__).resolve().parent
STT = WW.parent / "stt-finetune" / "recordings"
OUT = WW / "user_voice"

SR = 16000
WIN = 2.0  # 2s windows (matches livekit-wakeword clip_duration)


def save_int16_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(pcm, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def read_16k_int16(path: Path) -> np.ndarray:
    pcm, sr = sf.read(str(path), dtype="int16")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1).astype(np.int16)
    if sr != SR:
        pcm = resample_poly(pcm.astype(np.int16), SR, sr).astype(np.int16)
    return pcm.astype(np.float32) / 32768.0


def slice_into_windows(pcm: np.ndarray, win_s: float = WIN, hop_s: float = 0.5):
    win = int(win_s * SR)
    hop = int(hop_s * SR)
    if len(pcm) < win:
        yield np.pad(pcm, (0, win - len(pcm)))
        return
    for s in range(0, len(pcm) - win + 1, hop):
        yield pcm[s:s + win]


def restore_old_user_cozy():
    """Restore 32 'cozy' recordings from git commit e2a4dda."""
    out_dir = OUT / "user_cozy"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in range(1, 33):
        src_path = f"e2a4dda:wakeword/data/cozy/recording_{i:03d}.wav"
        dst_path = out_dir / f"recording_{i:03d}.wav"
        if dst_path.exists():
            continue
        r = subprocess.run(["git", "show", src_path],
                           cwd=str(WW.parent), capture_output=True)
        if r.returncode == 0:
            dst_path.write_bytes(r.stdout)
            n += 1
    print(f"  Restored {n} user_cozy recordings from git history")


def extract_stt_pos_neg():
    """Slice STT recordings: cozy -> 2s positive windows, others -> 2s negative windows."""
    pos_dir = OUT / "user_pos_stt"
    neg_dir = OUT / "user_neg_stt"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    n_pos = n_neg = 0
    for wav in sorted(STT.glob("session_*/*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue
        is_pos = "cozy" in txt.read_text().lower()
        out_dir = pos_dir if is_pos else neg_dir
        prefix = f"stt_{wav.parent.name}_{wav.stem}"
        pcm = read_16k_int16(wav)
        for idx, seg in enumerate(slice_into_windows(pcm)):
            out_path = out_dir / f"{prefix}_{idx:02d}.wav"
            if not out_path.exists():
                save_int16_wav(out_path, seg)
        if is_pos:
            n_pos += 1
        else:
            n_neg += 1
    print(f"  Sliced {n_pos} STT positives and {n_neg} STT negatives")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Restoring user_cozy from git history ===")
    restore_old_user_cozy()
    print("\n=== Slicing STT recordings into 2s windows ===")
    extract_stt_pos_neg()
    print("\nDone.")
    print(f"Output: {OUT}")
    for d in ["user_cozy", "user_pos_stt", "user_neg_stt"]:
        n = len(list((OUT / d).glob("*.wav")))
        print(f"  {d}/: {n} files")


if __name__ == "__main__":
    main()
