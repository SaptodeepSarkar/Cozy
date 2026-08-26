"""Strip silence from ALL recordings across the dataset using Silero VAD
+ energy filter, and plot a chart showing how much silence was removed.

Processes:
  - data/cozy/                  (your hey-cozy takes)
  - data/similar/               (lookalike negatives)
  - data/archive_bare_cozy/cozy (archived takes)
  - data/archive_bare_cozy/similar (archived lookalikes)

Usage:
  python strip_all.py           # process everything, write plot
  python strip_all.py --inplace # rewrite files in place (default)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SR = 16000
BLOCK = 512  # 32ms - required for silero at 16kHz
VAD_THRESHOLD = 0.5
ENERGY_FLOOR = 200  # int16 amplitude floor (filters hiss/breath)
PAD_BEFORE = 0.15   # keep 150ms before speech
PAD_AFTER = 0.20    # keep 200ms after speech
MIN_KEEP = 0.20     # never trim below 200ms total


def load_vad():
    return torch.hub.load(
        "snakers4/silero-vad", "silero_vad",
        trust_repo=True, force_reload=False)[0]


def vad_mask(pcm: np.ndarray, vad) -> np.ndarray:
    """Return boolean mask of speech frames.
    Speech = (vad prob >= threshold) OR (energy >= floor)."""
    n = len(pcm)
    speech = np.zeros(n, dtype=bool)
    x = torch.from_numpy(pcm.astype(np.float32) / 32768.0)
    for i in range(0, n - BLOCK, BLOCK):
        block_pcm = pcm[i:i + BLOCK]
        block_f = x[i:i + BLOCK]
        vad_prob = float(vad(block_f, SR).item())
        energy = float(np.abs(block_pcm).max())
        if vad_prob >= VAD_THRESHOLD or energy >= ENERGY_FLOOR:
            speech[i:i + BLOCK] = True
    # tail block
    tail = n - (n // BLOCK) * BLOCK
    if tail > 0:
        s = (n // BLOCK) * BLOCK
        block_pcm = pcm[s:]
        energy = float(np.abs(block_pcm).max()) if len(block_pcm) else 0
        if energy >= ENERGY_FLOOR:
            speech[s:] = True
    return speech


def trim(pcm: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        # nothing detected - keep at least MIN_KEEP seconds of the
        # loudest section so we don't delete the file entirely
        return pcm[: int(MIN_KEEP * SR)] if len(pcm) > int(MIN_KEEP * SR) else pcm
    idx = np.flatnonzero(mask)
    start = max(0, idx[0] - int(PAD_BEFORE * SR))
    end = min(len(pcm), idx[-1] + int(PAD_AFTER * SR))
    out = pcm[start:end]
    if len(out) / SR < MIN_KEEP:
        # too aggressive trim - extend back to MIN_KEEP
        center = (start + end) // 2
        target = int(MIN_KEEP * SR)
        new_start = max(0, center - target // 2)
        out = pcm[new_start:new_start + target]
    return out


def process_file(path: Path, vad) -> tuple[int, int]:
    """Returns (original_samples, trimmed_samples)."""
    try:
        pcm, sr = sf.read(str(path), dtype="int16")
    except Exception:
        return 0, 0
    if sr != SR:
        return len(pcm), len(pcm)  # skip non-16k
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1).astype(np.int16)
    mask = vad_mask(pcm, vad)
    out = trim(pcm, mask)
    if not np.array_equal(out, pcm) and len(out) > 0:
        sf.write(str(path), out, SR, subtype="PCM_16")
    return len(pcm), len(out)


FOLDERS = [
    ("data/cozy", "cozy"),
    ("data/similar", "similar"),
    ("data/archive_bare_cozy/cozy", "archive_bare_cozy/cozy"),
    ("data/archive_bare_cozy/similar", "archive_bare_cozy/similar"),
]


def main() -> None:
    print("loading Silero VAD...")
    vad = load_vad()

    stats = []  # (folder_label, n_files, total_orig_s, total_new_s)
    for rel, label in FOLDERS:
        folder = HERE / rel
        if not folder.exists():
            print(f"skip (missing): {rel}")
            continue
        files = sorted(folder.rglob("*.wav"))
        n = len(files)
        orig_total = new_total = 0
        if n == 0:
            print(f"{label}: 0 files")
            stats.append((label, 0, 0, 0))
            continue
        for i, wav in enumerate(files, 1):
            o, n_ = process_file(wav, vad)
            orig_total += o
            new_total += n_
            if i % 25 == 0 or i == n:
                print(f"  {label}: {i}/{n}")
        stats.append((label, n, orig_total, new_total))

    # ---- plot ----
    labels = [s[0] for s in stats]
    origs = np.array([s[2] for s in stats], dtype=float) / SR
    news = np.array([s[3] for s in stats], dtype=float) / SR
    silences = origs - news

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    w = 0.6
    bars_new = ax.bar(x, news, w, label="speech kept", color="#2a9d8f")
    bars_sil = ax.bar(x, silences, w, bottom=news, label="silence removed",
                      color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n({s[1]} files)" for l, s in zip(labels, stats)],
                        fontsize=9)
    ax.set_ylabel("seconds (combined)")
    ax.set_title("Silence removed across Cozy wake-word dataset\n"
                 "(Silero VAD + energy floor)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # annotate totals
    total_orig = origs.sum()
    total_new = news.sum()
    total_sil = silences.sum()
    pct = (total_sil / total_orig * 100) if total_orig > 0 else 0
    ax.text(0.02, 0.97,
             f"TOTAL: {total_orig/60:.1f} min -> {total_new/60:.1f} min "
             f"({pct:.0f}% silence removed)",
             transform=ax.transAxes, va="top", ha="left",
             fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    out = HERE / "silence_removed.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"\nplot: {out}")

    # CSV summary
    csv = HERE / "silence_report.csv"
    with open(csv, "w") as f:
        f.write("folder,n_files,orig_seconds,new_seconds,silence_seconds,pct\n")
        for lbl, n, o, n_ in stats:
            f.write(f"{lbl},{n},{o/SR:.2f},{n_/SR:.2f},"
                    f"{(o-n_)/SR:.2f},{(o-n_)/o*100 if o else 0:.1f}\n")
        f.write(f"TOTAL,{sum(s[1] for s in stats)},"
                f"{total_orig:.2f},{total_new:.2f},"
                f"{total_sil:.2f},{pct:.1f}\n")
    print(f"report: {csv}")
    print(f"\nTOTAL: {total_orig/60:.1f} min -> {total_new/60:.1f} min "
          f"({pct:.0f}% silence)")


if __name__ == "__main__":
    main()
