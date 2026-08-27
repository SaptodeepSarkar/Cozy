#!/usr/bin/env python3
"""Re-extract user voice data using 1.0s windows centered on cozy. v3 fixed."""
from __future__ import annotations
import json, random, wave
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

WW = Path("/home/saptodeepsarkar/Projects/Cozy/wakeword")
STT = Path("/home/saptodeepsarkar/Projects/Cozy/stt-finetune/recordings")
WORK = WW / "work"
SR = 16000
WIN = int(1.0 * SR)
random.seed(42)


def save_int16(path: Path, pcm: np.ndarray) -> None:
    # pcm is expected to be in [-1, 1] float; scale to int16 range
    pcm = np.clip(pcm * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def read_16k_int16(path: Path) -> tuple[np.ndarray, int]:
    """Read any wav as 16kHz int16 mono."""
    pcm, sr = sf.read(str(path), dtype="int16")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1).astype(np.int16)
    if sr != SR:
        pcm = resample_poly(pcm.astype(np.int16), SR, sr).astype(np.int16)
    return pcm, SR


def extract_positive(wav_path: Path, txt: str, out_dir: Path, idx: int) -> list[Path]:
    pcm, sr = read_16k_int16(wav_path)
    pcm = pcm.astype(np.float32) / 32768.0  # normalize to [-1, 1]

    text_lower = txt.lower().strip()
    out = []

    # Strategy 1: leading 1.0s (covers "Hey Cozy," at start)
    if len(pcm) >= WIN:
        seg = pcm[:WIN]
        out_path = out_dir / f"{wav_path.parent.name}_{wav_path.stem}_{idx:02d}_lead.wav"
        save_int16(out_path, seg)
        out.append(out_path)

    # Strategy 2: mid window around cozy phrase
    if "cozy" in text_lower:
        words = text_lower.translate(str.maketrans("", "", ",.!?;:")).split()
        try:
            cozy_idx = next(i for i, w in enumerate(words) if "cozy" in w)
            pos_frac = cozy_idx / max(1, len(words))
        except StopIteration:
            pos_frac = 0.0
        cozy_time = pos_frac * (len(pcm) / SR)
        for offset_s, suffix in [(-0.1, "a"), (0.0, "b"), (0.1, "c")]:
            start_s = max(0.0, cozy_time - 0.5 + offset_s)
            start = int(start_s * SR)
            if start + WIN <= len(pcm):
                seg = pcm[start:start + WIN]
                out_path = out_dir / f"{wav_path.parent.name}_{wav_path.stem}_{idx:02d}_{suffix}.wav"
                save_int16(out_path, seg)
                out.append(out_path)
    return out


def slice_negatives(wav_path: Path, out_dir: Path) -> list[Path]:
    pcm, sr = read_16k_int16(wav_path)
    pcm = pcm.astype(np.float32) / 32768.0
    out = []
    if len(pcm) < WIN:
        seg = np.pad(pcm, (0, WIN - len(pcm)))
        out_path = out_dir / f"neg_{wav_path.parent.name}_{wav_path.stem}.wav"
        save_int16(out_path, seg)
        out.append(out_path)
    else:
        step = int(0.5 * SR)
        idx = 0
        for s in range(0, len(pcm) - WIN + 1, step):
            seg = pcm[s:s + WIN]
            out_path = out_dir / f"neg_{wav_path.parent.name}_{wav_path.stem}_{idx:02d}.wav"
            save_int16(out_path, seg)
            out.append(out_path)
            idx += 1
    return out


def main() -> None:
    for d in ["user_pos", "val_pos", "user_neg", "val_neg"]:
        target = WORK / d
        if target.exists():
            for p in target.glob("*.wav"):
                p.unlink()
    user_pos = WORK / "user_pos"; user_pos.mkdir(parents=True, exist_ok=True)
    user_neg = WORK / "user_neg"; user_neg.mkdir(parents=True, exist_ok=True)
    val_pos = WORK / "val_pos"; val_pos.mkdir(parents=True, exist_ok=True)
    val_neg = WORK / "val_neg"; val_neg.mkdir(parents=True, exist_ok=True)

    # 1) data/cozy - copy as-is
    cozy_pos = []
    for p in sorted((WW / "data" / "cozy").glob("recording_*.wav")):
        pcm, sr = read_16k_int16(p)
        pcm = pcm.astype(np.float32) / 32768.0
        if len(pcm) < WIN:
            pcm = np.pad(pcm, (0, WIN - len(pcm)))
        else:
            pcm = pcm[:WIN]
        out = user_pos / p.name
        save_int16(out, pcm)
        cozy_pos.append(out)
    print(f"data/cozy: {len(cozy_pos)}")

    # 2) STT positives
    stt_pos = [p for p in sorted(STT.glob("session_*/*.wav"))
               if (p.with_suffix(".txt").exists()
                   and "cozy" in p.with_suffix(".txt").read_text().lower())]
    all_stt_pos_files = []
    for i, p in enumerate(stt_pos):
        outs = extract_positive(p, p.with_suffix(".txt").read_text(), user_pos, i)
        all_stt_pos_files.extend(outs)
    print(f"STT positives: {len(stt_pos)} files -> {len(all_stt_pos_files)} crops")

    # 3) Hold out 8 STT-positive crops
    random.shuffle(all_stt_pos_files)
    held = all_stt_pos_files[:8]
    for p in held:
        (val_pos / p.name).write_bytes(p.read_bytes())
        p.unlink()
    print(f"train_pos: {len(list(user_pos.glob('*.wav')))}  val_pos: {len(held)}")

    # 4) Extract negatives
    stt_neg = [p for p in sorted(STT.glob("session_*/*.wav"))
               if (p.with_suffix(".txt").exists()
                   and "cozy" not in p.with_suffix(".txt").read_text().lower())]
    n_neg_slices = 0
    for p in stt_neg:
        for out in slice_negatives(p, user_neg):
            n_neg_slices += 1
    print(f"user_neg slices: {n_neg_slices}")

    # 5) Hold out 150 negatives
    user_neg_files = sorted(user_neg.glob("*.wav"))
    random.shuffle(user_neg_files)
    held_neg = user_neg_files[:150]
    for p in held_neg:
        (val_neg / p.name).write_bytes(p.read_bytes())
        p.unlink()
    user_neg_files = sorted(user_neg.glob("*.wav"))
    random.shuffle(user_neg_files)
    keep = user_neg_files[:1500]
    for p in user_neg_files[1500:]:
        p.unlink()
    print(f"train_neg_user: {len(list(user_neg.glob('*.wav')))}  val_neg: {len(held_neg)}")

    print("DONE_USER_EXTRACT_V3")


if __name__ == "__main__":
    main()
