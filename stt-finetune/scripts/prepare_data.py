#!/usr/bin/env python
"""Build train/eval manifests combining FLEURS en_in (Indian English) with YOUR
recorded voice (recordings/session_*/), up-sampling your clips so they get a
healthy share of gradient steps without letting 90 clips swamp 2,500.

Outputs (data/manifests/):
    train.jsonl   {audio_path | fleurs_id, text, source}
    eval.jsonl    held-out FLEURS val/test + held-out user sessions

Run:  .venv/bin/python scripts/prepare_data.py [--user-repeat auto] [--eval-sessions 6]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FLEURS_DIR, MANIFEST_DIR, RECORDINGS_DIR  # noqa: E402

random.seed(42)


def collect_user_clips(eval_sessions: set):
    """Returns (train_clips, eval_clips): [(wav_path, text)]"""
    train, evalc = [], []
    if not RECORDINGS_DIR.exists():
        return train, evalc
    for sdir in sorted(RECORDINGS_DIR.glob("session_*")):
        sid = int(sdir.name.split("_")[1])
        for wav in sorted(sdir.glob("*.wav")):
            txt = wav.with_suffix(".txt").read_text().strip()
            if not txt:
                continue
            (evalc if sid in eval_sessions else train).append((str(wav), txt))
    return train, evalc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-repeat", default="auto",
                    help="times each user clip repeats per epoch ('auto' balances to ~12%%)")
    ap.add_argument("--eval-sessions", type=int, nargs="*", default=None,
                    help="session ids reserved entirely for evaluation "
                         "(default: highest recorded session)")
    ap.add_argument("--fleurs-eval-per-split", type=int, default=150)
    args = ap.parse_args()

    from datasets import load_from_disk

    ds = load_from_disk(str(FLEURS_DIR))
    norm_rows = []
    for split in ("validation", "test"):
        take = ds[split].select(range(min(args.fleurs_eval_per_split, len(ds[split]))))
        for ex in take:
            # FLEURS provides both raw and normalised transcripts; use raw.
            text = ex["transcription"].strip()
            norm_rows.append({
                "fleurs_split": split, "fleurs_index": ex["id"],
                "audio_path": None, "text": text, "source": f"fleurs_{split}",
            })

    all_sessions = sorted(int(d.name.split("_")[1])
                          for d in RECORDINGS_DIR.glob("session_*")) or []
    eval_sessions = set(args.eval_sessions) if args.eval_sessions else (
        {max(all_sessions)} if len(all_sessions) >= 2 else set())
    user_train, user_eval = collect_user_clips(eval_sessions)
    print(f"User voice: {len(user_train)} train / {len(user_eval)} holdout clips "
          f"(holdout sessions: {sorted(eval_sessions)})")

    train_rows = []
    for ex in ds["train"]:
        train_rows.append({
            "fleurs_split": "train", "fleurs_index": ex["id"],
            "audio_path": None, "text": ex["transcription"].strip(),
            "source": "fleurs_train",
        })

    n_user_repeats = 1
    if args.user_repeat == "auto":
        if user_train:
            target_frac = 0.12
            # repeats so that user samples ≈ target_frac of one epoch
            n_user_repeats = max(1, min(25, round(
                target_frac / (1 - target_frac) * len(train_rows) / len(user_train))))
    else:
        n_user_repeats = int(args.user_repeat)

    for i in range(n_user_repeats):
        for j, (path, text) in enumerate(user_train):
            train_rows.append({"audio_path": path, "text": text,
                               "source": f"user_r{i}", "user_key": j})
    random.shuffle(train_rows)

    eval_rows = list(norm_rows)
    for path, text in user_eval:
        eval_rows.append({"audio_path": path, "text": text, "source": "user_holdout"})
    random.shuffle(eval_rows)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train_rows), ("eval.jsonl", eval_rows)):
        out = MANIFEST_DIR / name
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        srcs = {}
        for r in rows:
            key = r["source"].split("_r")[0]
            srcs[key] = srcs.get(key, 0) + 1
        print(f"{name}: {len(rows)} rows  {srcs}")
    print(f"user clip repeat factor: {n_user_repeats}x")


if __name__ == "__main__":
    main()
