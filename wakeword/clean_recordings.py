#!/usr/bin/env python3
"""Strips silence from every user recording in data/cozy and data/similar.

- trims leading/trailing quiet below 3% of the clip's peak
- squeezes internal silent gaps longer than 0.8 s down to 0.3 s
- leaves ~50 ms of breathing room on each edge
Rewrites files in place (16 kHz mono PCM_16). Safe to run repeatedly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
SR = 16000
GAP_KEEP = int(0.30 * SR)
GAP_TRIM = int(0.80 * SR)
EDGE_KEEP = int(0.05 * SR)


def clean(pcm: np.ndarray) -> np.ndarray | None:
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    peak = int(np.abs(pcm).max()) if len(pcm) else 0
    if peak == 0:
        return None
    floor = max(150, int(peak * 0.03))
    loud = np.abs(pcm) > floor
    idx = np.flatnonzero(loud)
    if idx.size == 0:
        return None
    start = max(0, int(idx[0]) - EDGE_KEEP)
    end = min(len(pcm), int(idx[-1]) + 1 + EDGE_KEEP)
    body = pcm[start:end]

    silent = ~loud[start:end]
    d = np.diff(silent.astype(np.int8))
    gap_starts = np.flatnonzero(d == 1) + 1
    gap_ends = np.flatnonzero(d == -1) + 1
    if silent[0]:
        gap_starts = np.concatenate([[0], gap_starts])
    if silent[-1]:
        gap_ends = np.concatenate([gap_ends, [len(silent)]])

    keep = np.ones(len(body), dtype=bool)
    for g0, g1 in zip(gap_starts, gap_ends):
        if g1 - g0 > GAP_TRIM:
            keep[g0 + GAP_KEEP:g1] = False
    trimmed = body[keep]
    # normalize to a healthy level (-3 dBFS target) so inconsistent mic
    # gain can never produce whisper-quiet or clipped features again
    peak = int(np.abs(trimmed).max()) if len(trimmed) else 0
    if 0 < peak < 15000:
        gain = min(22000 / peak, 8.0)
        trimmed = (trimmed.astype(np.float32) * gain).clip(-32767, 32767)
        trimmed = trimmed.astype(np.int16)
    elif peak > 30000:
        trimmed = (trimmed.astype(np.float32)
                   * (28000.0 / peak)).clip(-32767, 32767).astype(np.int16)
    head = np.zeros(EDGE_KEEP, dtype=trimmed.dtype)
    tail = np.zeros(int(0.08 * SR), dtype=trimmed.dtype)
    return np.concatenate([head, trimmed, tail])


def main() -> None:
    total, changed = 0, 0
    for folder in ("data/cozy", "data/similar"):
        for wav in sorted((HERE / folder).glob("*.wav")):
            if wav.name.startswith("synth_"):
                continue
            pcm, sr = sf.read(str(wav), dtype="int16")
            if sr != SR:
                print("  ! skip (wrong rate) " + wav.name)
                continue
            total += 1
            new = clean(pcm)
            if new is None:
                continue
            orig_peak = int(np.abs(pcm).max()) if len(pcm) else 0
            new_peak = int(np.abs(new).max()) if len(new) else 0
            length_changed = abs(len(new) - len(pcm)) >= int(0.02 * SR)
            level_bad = orig_peak < 8000 or orig_peak > 30000
            if not length_changed and not level_bad and \
                    abs(new_peak - orig_peak) < 500:
                continue
            sf.write(str(wav), new.astype(np.int16), SR, subtype="PCM_16")
            changed += 1
            print("  cleaned " + folder + "/" + wav.name + " "
                  + format(len(pcm) / SR, ".2f") + "s -> "
                  + format(len(new) / SR, ".2f") + "s")
    print("[clean] " + str(changed) + "/" + str(total)
          + " recordings needed cleaning")


if __name__ == "__main__":
    main()
