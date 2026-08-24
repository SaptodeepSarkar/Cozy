#!/usr/bin/env python
"""Build train/eval manifests combining the Indian-English corpora with YOUR
recordings, up-sampling your voice so it gets a healthy share of gradient steps.

Run:  .venv/bin/python scripts/prepare_data.py [--user-repeat auto]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, MANIFEST_DIR, RECORDINGS_DIR  # noqa: E402

random.seed(42)
POOLS = ["cv_indian", "nptel_indian"]
EVAL_TAKE = {"cv_indian": 120, "nptel_indian": 60}


def load_pool(name):
    man = DATA_DIR / name / "manifest.jsonl"
    if not man.exists():
        return []
    rows = [json.loads(l) for l in open(man) if l.strip()]
    random.shuffle(rows)
    n_eval = min(EVAL_TAKE.get(name, 50), max(1, len(rows) // 10))
    return rows[n_eval:], rows[:n_eval]   # train_part, eval_part


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
                    help="repeats per user clip ('auto' ~12%% of epoch, or int)")
    ap.add_argument("--eval-sessions", type=int, nargs="*", default=None,
                    help="session ids reserved entirely for evaluation "
                         "(default: newest session, if >=2 sessions exist)")
    args = ap.parse_args()

    all_sessions = sorted(int(d.name.split("_")[1])
                          for d in RECORDINGS_DIR.glob("session_*")) or []
    eval_sessions = set(args.eval_sessions) if args.eval_sessions else (
        {max(all_sessions)} if len(all_sessions) >= 2 else set())
    user_train, user_eval = collect_user_clips(eval_sessions)
    print(f"User voice: {len(user_train)} train / {len(user_eval)} holdout clips "
          f"(holdout sessions: {sorted(eval_sessions)})")

    train_rows, eval_rows = [], []
    for pool in POOLS:
        tr, ev = load_pool(pool)
        print(f"{pool}: {len(tr)} train / {len(ev)} held-out")
        train_rows += tr
        eval_rows += ev
    eval_rows += user_eval

    base_n = len(train_rows)
    if user_train:
        if args.user_repeat == "auto":
            frac = 0.12
            n_repeats = max(1, min(25, round(frac / (1 - frac) * base_n / len(user_train))))
        else:
            n_repeats = int(args.user_repeat)
        train_rows += [dict(r) for _ in range(n_repeats - 1) for r in user_train]
        train_rows += user_train
    else:
        n_repeats = 0
    random.shuffle(train_rows)

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
