#!/usr/bin/env python
"""Build train/eval manifests combining FLEURS en_in (Indian English) with YOUR
recorded voice (recordings/session_*/), up-sampling your clips so they get a
healthy share of gradient steps without letting ~90 clips swamp ~2,500.

Outputs (data/manifests/):
    train.jsonl   {audio_path, text, source}
    eval.jsonl    held-out FLEURS val/test + held-out user session(s)

Run:  .venv/bin/python scripts/prepare_data.py
      optional: --eval-sessions 6  (reserve a whole session as personal holdout)
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MANIFEST_DIR, RECORDINGS_DIR, ROOT  # noqa: E402

random.seed(42)
FLEURS_RAW = ROOT / "data" / "fleurs_raw" / "en_in"


def parse_fleurs_tsv(tsv: Path):
    """FLEURS tsv columns: id <tab> filename <tab> raw_transcription
    <tab> transcription <tab> num_samples <tab> gender"""
    wav_dir = tsv.parent / (tsv.stem.split("~")[-1]) if False else None
    rows = []
    with open(tsv, encoding="utf-8") as f:
        header = f.readline()  # column names line? FLEURS tsv has no header; keep robust
        f.seek(0)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4 or parts[0] == "id":
                continue
            _id, fname, raw_text = parts[0], parts[1], parts[2]
            rows.append((_id, fname, raw_text.strip()))
    return rows


def find_split_dir(split: str) -> Path:
    """Locate the extracted dir containing *.tsv + wavs for this split."""
    hits = list(FLEURS_RAW.rglob(f"*{split}*.tsv"))
    if not hits:
        raise FileNotFoundError(f"No FLEURS {split} tsv under {FLEURS_RAW} — run download_assets.py")
    return hits[0].parent


def fleurs_rows(split: str, limit=None):
    d = find_split_dir(split)
    tsv = next(f for f in d.glob("*.tsv"))
    out = []
    for _id, fname, text in parse_fleurs_tsv(d / tsv.name):
        wav = d / fname.split("/")[-1]
        if not wav.exists():
            # some layouts nest wavs one level deeper
            cand = list(d.rglob(fname.split("/")[-1]))
            if not cand:
                continue
            wav = cand[0]
        out.append({"audio_path": str(wav), "text": text, "source": f"fleurs_{split}"})
    random.shuffle(out)
    return out[:limit] if limit else out


def collect_user_clips(eval_sessions: set):
    train, evalc = [], []
    if not RECORDINGS_DIR.exists():
        return train, evalc
    for sdir in sorted(RECORDINGS_DIR.glob("session_*")):
        sid = int(sdir.name.split("_")[1])
        for wav in sorted(sdir.glob("*.wav")):
            txt_file = wav.with_suffix(".txt")
            if not txt_file.exists():
                continue
            txt = txt_file.read_text().strip()
            if txt:
                (evalc if sid in eval_sessions else train).append(
                    {"audio_path": str(wav), "text": txt, "source": "user"})
    return train, evalc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-repeat", default="auto",
                    help="repeats per user clip ('auto' balances to ~12%% of an epoch)")
    ap.add_argument("--eval-sessions", type=int, nargs="*", default=None,
                    help="session ids reserved entirely for evaluation "
                         "(default: newest recorded session, if >=2 sessions exist)")
    ap.add_argument("--fleurs-eval-per-split", type=int, default=150)
    args = ap.parse_args()

    all_sessions = sorted(int(d.name.split("_")[1])
                          for d in RECORDINGS_DIR.glob("session_*")) or []
    eval_sessions = set(args.eval_sessions) if args.eval_sessions else (
        {max(all_sessions)} if len(all_sessions) >= 2 else set())
    user_train, user_eval = collect_user_clips(eval_sessions)
    print(f"User voice: {len(user_train)} train / {len(user_eval)} holdout clips "
          f"(holdout sessions: {sorted(eval_sessions)})")

    train_rows = fleurs_rows("train") + user_train
    n_repeats = 1
    if args.user_repeat == "auto" and user_train:
        n_fleurs = len(train_rows) - len(user_train)
        frac = 0.12
        n_repeats = max(1, min(25, round(frac / (1 - frac) * n_fleurs / len(user_train))))
        train_rows += [dict(r) for _ in range(n_repeats - 1) for r in user_train]
    elif isinstance(args.user_repeat, str) and args.user_repeat.isdigit():
        n_repeats = int(args.user_repeat)
        train_rows += [dict(r) for _ in range(n_repeats - 1) for r in user_train]
    random.shuffle(train_rows)

    eval_rows = (fleurs_rows("validation", args.fleurs_eval_per_split)
                 + fleurs_rows("test", args.fleurs_eval_per_split) + user_eval)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train_rows), ("eval.jsonl", eval_rows)):
        out = MANIFEST_DIR / name
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        srcs = {}
        for r in rows:
            srcs[r["source"]] = srcs.get(r["source"], 0) + 1
        print(f"{name}: {len(rows)} rows  {srcs}")
    print(f"user clip repeat factor: {n_repeats}x")


if __name__ == "__main__":
    main()
