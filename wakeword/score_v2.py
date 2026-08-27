#!/usr/bin/env python3
"""Score a folder of wavs with cozynet_v2 in PURE EVAL mode (the way
runtime.py will use it). Print score distribution + threshold sweep.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

WW = Path("/home/saptodeepsarkar/Projects/Cozy/wakeword")
import json
meta = json.loads((WW / "models" / "cozynet_v2_meta.json").read_text())
T = meta["time_frames"]
BINS = meta["mel_bins"]
MEAN = np.array(meta["mel_mean"], dtype=np.float32)
STD = np.array(meta["mel_std"], dtype=np.float32)
SAFE = float(meta.get("safe_threshold_zero_fpr") or 0.7)
print(f"meta: T={T} BINS={BINS} safe_thr={SAFE} val_auc={meta['val_auc_real_voice']}")

# Use openWakeWord's mel extractor (matches training exactly)
from openwakeword.model import AudioFeatures
_af = AudioFeatures()

sess = ort.InferenceSession(str(WW / "models" / "cozynet_v2.onnx"),
                            providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
out = sess.get_outputs()[0].name
print(f"onnx: in={inp} out={out}")


def mel_of(pcm_int16: np.ndarray) -> np.ndarray:
    if pcm_int16.ndim > 1:
        pcm_int16 = pcm_int16.mean(axis=1)
    pcm_int16 = pcm_int16.astype(np.int16)
    out = _af._get_melspectrogram_batch(pcm_int16[None, :], batch_size=8, ncpu=4)
    return np.asarray(out)[0].astype(np.float32)


def score_wav(path: Path) -> float:
    """Score a wav with the production-style sliding window flow."""
    pcm, sr = sf.read(str(path), dtype="int16")
    if sr != 16000:
        from scipy.signal import resample_poly
        pcm = resample_poly(pcm.astype(np.int16), 16000, sr).astype(np.int16)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1).astype(np.int16)
    if len(pcm) < 16000:
        pcm = np.pad(pcm, (0, 16000 - len(pcm)))
    m = mel_of(pcm)
    if m.shape[0] < T:
        m = np.pad(m, ((0, T - m.shape[0]), (0, 0)))
    best = 0.0
    for s0 in range(0, max(1, m.shape[0] - T + 1), 19):  # HOP=19
        win = m[s0:s0 + T]
        if win.shape[0] < T:
            win = np.pad(win, ((0, T - win.shape[0]), (0, 0)))
        win = (win - MEAN) / STD
        xb = win[None, None].astype(np.float32)
        logit = float(sess.run([out], {inp: xb})[0].reshape(-1)[0])
        s = 1.0 / (1.0 + np.exp(-logit))  # sigmoid, model exports raw logits
        if s > best:
            best = s
    return best


def score_folder(folder: Path, label: str, limit: int | None = None):
    files = sorted(folder.glob("*.wav"))
    if limit:
        files = files[:limit]
    scores = []
    for p in files:
        s = score_wav(p)
        scores.append(s)
    scores = np.array(scores)
    print(f"\n[{label}] {len(scores)} files from {folder}")
    if len(scores) == 0:
        return scores
    print(f"  min={scores.min():.3f}  median={np.median(scores):.3f}  "
          f"max={scores.max():.3f}  mean={scores.mean():.3f}")
    for thr in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        fires = (scores >= thr).sum()
        rate = fires / len(scores)
        print(f"  thr={thr}: fires={fires}/{len(scores)} ({rate*100:.1f}%)")
    return scores


if __name__ == "__main__":
    val_pos = WW / "work" / "val_pos"
    val_neg = WW / "work" / "val_neg"
    user_pos = WW / "work" / "user_pos"  # train positives (sanity check only)
    similar = WW / "data" / "similar"
    work_similar = WW / "work" / "similar"
    negative = WW / "work" / "negative"

    pos_scores = score_folder(val_pos, "VAL POS (real voice, held out)")
    neg_scores = score_folder(val_neg, "VAL NEG (real voice, held out)")
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        # Find the threshold that gives zero FPR on val
        max_neg = float(neg_scores.max())
        print(f"\nMax neg score on val: {max_neg:.3f}")
        print(f"To guarantee zero FPR, set threshold > {max_neg:.3f}")
        # What TPR would we get at the best safe threshold?
        # Try several
        print("\nTPR vs threshold (val only):")
        for thr_offset in [0.01, 0.02, 0.05, 0.10]:
            thr = min(0.99, max_neg + thr_offset)
            tpr = float((pos_scores >= thr).mean()) if len(pos_scores) else 0.0
            fpr = float((neg_scores >= thr).mean())
            print(f"  thr={thr:.3f}  TPR={tpr*100:.1f}%  FPR={fpr*100:.1f}%")

    # Also score the data/similar seed (your voice on similar words)
    print("\n=== Hard negative seed files (your voice) ===")
    seed_files = sorted((WW / "data" / "similar").glob("*.wav"))
    if seed_files:
        for p in seed_files[:20]:
            s = score_wav(p)
            print(f"  {p.name}: {s:.3f}")
        # Group by word
        for word in ["hey_rosy", "hey_nosy", "hey_josy", "hey_ozzie", "hey_dozy",
                     "hey_posy", "hey_noisy", "rosy", "nosy", "dozy", "posy",
                     "noisy", "josie", "ozzie", "cause_he", "cozy_dash"]:
            matches = [p for p in seed_files if word in p.name][:3]
            for p in matches:
                s = score_wav(p)
                print(f"  {p.name}: {s:.3f}")
