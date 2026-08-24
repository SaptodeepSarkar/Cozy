#!/usr/bin/env python
"""Build the Indian-English corpus from two ungated HF datasets into plain
wav files + local manifests under data/:

  1. kaushalgawri/indian_accent_en_train  (4,489 clips tagged accent='indian',
     Common-Voice style votes)                       -> data/cv_indian/
  2. skbose/indian-english-nptel-v0-tags-gender-accent
     (NPTEL lecture speech by Indian professors)     -> data/nptel_indian/

Each output dir gets manifest.jsonl rows: {audio_path,text,source}.
Run:  source env.sh && .venv/bin/python scripts/build_indian_corpus.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR  # noqa: E402

random.seed(42)

CV_REPO = "kaushalgawri/indian_accent_en_train"
NPTEL_REPO = "skbose/indian-english-nptel-v0-tags-gender-accent"


def decode_audio(cell):
    """audio cell may be a JSON string {'array': [...], 'sampling_rate': r}
    or an arrow struct with the same keys."""
    import numpy as np
    if isinstance(cell, (str, bytes)):
        d = json.loads(cell)
    else:
        try:
            d = cell.as_py() if hasattr(cell, "as_py") else dict(cell)
        except Exception:
            return None, None
    arr = np.asarray(d["array"], dtype=np.float32)
    return arr, int(d.get("sampling_rate", 16000))


def extract(repo, out_dir: Path, text_keys, filters=None, max_clips=None):
    """Download repo parquet(s), decode selected rows to wav + manifest."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    man = out_dir / "manifest.jsonl"
    done_marker = out_dir / ".done"
    if done_marker.exists():
        print(f"== {out_dir.name}: already built ({sum(1 for _ in open(man))} clips)")
        return

    files = [f for f in list_repo_files(repo, repo_type="dataset")
             if f.endswith(".parquet")]
    print(f"== {repo}: {len(files)} parquet file(s)")
    rows_out, skipped, fail_decode = [], 0, 0
    local_files = []
    try:
        for fname in sorted(files):
            local = Path(hf_hub_download(repo, fname, repo_type="dataset"))
            local_files.append(local)
            pf = pq.ParquetFile(local)
            # NOTE: read ALL columns — vote/quality filters need them
            for batch in pf.iter_batches(batch_size=64):
                d = batch.to_pylist()
                for row in d:
                    text_key = next((k for k in text_keys if row.get(k)), None)
                    text = (row.get(text_key) or "").strip() if text_key else ""
                    if not text or (filters and not filters(row)):
                        skipped += 1
                        continue
                    arr, sr = decode_audio(row["audio"])
                    if arr is None or len(arr) < 0.4 * sr or len(arr) > 30 * sr:
                        skipped += 1
                        continue
                    idx = len(rows_out)
                    wav_path = out_dir / f"clip_{idx:05d}.wav"
                    sf.write(wav_path, arr, sr, subtype="PCM_16")
                    meta = {k: row[k] for k in ("up_votes", "down_votes", "gender",
                                                "age", "speaker_name", "snr")
                            if k in row}
                    rows_out.append({"audio_path": str(wav_path), "text": text,
                                     "source": out_dir.name, **meta})
    finally:
        pass  # keep parquet cache until success confirmed below

    random.shuffle(rows_out)
    if max_clips:
        rows_out = rows_out[:max_clips]
    with open(man, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    done_marker.touch()
    print(f"   -> {len(rows_out)} clips written to {out_dir} "
          f"({skipped} filtered, {fail_decode} decode failures)")


def extract_santhosh(out_dir: Path):
    """Santhosh-kumar/Indian-Accent-Dataset: plain wav+txt pairs under audio/."""
    from huggingface_hub import snapshot_download
    out_dir.mkdir(parents=True, exist_ok=True)
    man = out_dir / "manifest.jsonl"
    done_marker = out_dir / ".done"
    if done_marker.exists():
        print(f"== {out_dir.name}: already built")
        return
    snap = Path(snapshot_download("Santhosh-kumar/Indian-Accent-Dataset",
                                  repo_type="dataset"))
    rows_out = []
    for wav in sorted((snap / "audio").glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue
        text = txt.read_text(encoding="utf-8", errors="ignore").strip()
        if not text or len(text) < 2:
            continue
        import os, shutil
        dst = out_dir / f"clip_{len(rows_out):05d}.wav"
        # HF cache files are symlinks — copy the real bytes
        shutil.copyfile(os.path.realpath(wav), dst)
        rows_out.append({"audio_path": str(dst), "text": text,
                         "source": out_dir.name})
    with open(man, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    done_marker.touch()
    print(f"== santhosh_indian: -> {len(rows_out)} clips")


def main():
    extract(
        CV_REPO, DATA_DIR / "cv_indian",
        text_keys=["text"],
        filters=lambda r: (r.get("down_votes") in (0, None)
                           and (r.get("up_votes") or 0) >= 1),
        max_clips=1500,
    )
    try:
        extract_santhosh(DATA_DIR / "santhosh_indian")
    except Exception as e:
        print(f"santhosh dataset unavailable ({e}); continuing without it.")
    print("\nIndian-English corpus ready.")


if __name__ == "__main__":
    main()
