#!/usr/bin/env python
"""Stage 1 audio processing for user recordings:
  1. strip leading/trailing silence (energy-based, keeps 150ms breathing room)
  2. peak-normalise to -3 dBFS so level never saturates the mel features
Rewrites session wavs in place (originals are re-recordable) and writes a
cleaning report. Idempotent: skips clips already processed (marker file).

Run: python3 scripts/clean_recordings.py
"""
import array
import json
import math
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REC = ROOT / "recordings"
WIN_MS = 20          # energy window
THRESH_RATIO = 0.06  # window RMS below 6% of clip peak => silence
PAD_MS = 150         # keep this much quiet around speech edges


def process(wav_path: Path):
    with wave.open(str(wav_path), "rb") as w:
        sr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    assert sr == 16000 and nch == 1 and sw == 2, f"unexpected format {wav_path}"
    samples = array.array("h")
    samples.frombytes(frames)
    if not len(samples):
        return None

    win = int(sr * WIN_MS / 1000)
    n_windows = len(samples) // win
    if n_windows == 0:
        return None
    rms = [math.sqrt(sum(s * s for s in samples[i * win:(i + 1) * win]) / win)
           for i in range(n_windows)]
    peak_rms = max(rms)
    thr = max(peak_rms * THRESH_RATIO, 80.0)

    voiced = [i for i, r in enumerate(rms) if r >= thr]
    if not voiced:
        return {"skipped": "no speech detected"}
    first, last = voiced[0], voiced[-1]
    pad_w = int(sr * PAD_MS / 1000)
    start = max(0, first * win - pad_w)
    end = min(len(samples), (last + 1) * win + pad_w)

    trimmed = samples[start:end]
    # peak normalise to ~-3 dBFS (23000-ish amplitude target)
    peak_amp = max(abs(s) for s in trimmed) or 1
    gain = min(23000 / peak_amp, 4.0)   # never boost more than 4x
    out = array.array("h", (int(s * gain) for s in trimmed))

    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(out.tobytes())
    return {
        "orig_s": round((len(samples)) / sr, 2),
        "new_s": round(len(out) / sr, 2),
        "cut_head_s": round(start / sr, 2),
        "cut_tail_s": round((len(samples) - end) / sr, 2),
        "gain": round(gain, 2),
    }


def main():
    report = {}
    for wav in sorted(REC.glob("session_*/*.wav")):
        marker = wav.with_suffix(".clean")
        if marker.exists():
            continue
        res = process(wav)
        report[str(wav.relative_to(REC))] = res
        if res and "skipped" not in res:
            marker.write_text(json.dumps(res))
    n = len(report)
    print(f"cleaned {n} clips")
    if n:
        heads = sum(v.get("cut_head_s", 0) for v in report.values() if isinstance(v, dict))
        tails = sum(v.get("cut_tail_s", 0) for v in report.values() if isinstance(v, dict))
        print(f"total silence removed: head {heads:.1f}s + tail {tails:.1f}s")
    bad = {k: v for k, v in report.items() if isinstance(v, dict) and "skipped" in v}
    if bad:
        print("!! skipped (re-record these):", bad)
    (REC / "clean_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
