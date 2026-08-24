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
    rows_out, skipped = [], 0
    for fname in sorted(files):
        local = Path(hf_hub_download(repo, fname, repo_type="dataset"))
        pf = pq.ParquetFile(local)
        cols = [c for c in pf.schema_arrow.names]
        need = set(["text"] + text_keys + ["audio"])
        cols = [c for c in cols if c in need] or None
        for batch in pf.iter_batches(batch_size=64, columns=cols):
            d = batch.to_pylist()
            for row in d:
                text_key = next((k for k in text_keys if row.get(k)), None)
                text = (row.get(text_key) or "").strip() if text_key else ""
                votes_ok = True
                if filters:
                    votes_ok = filters(row)
                if not text or not votes_ok:
                    skipped += 1
                    continue
                arr, sr = decode_audio(row["audio"])
                if arr is None or len(arr) < 0.5 * sr or len(arr) > 30 * sr:
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
        local.unlink(missing_ok=True)  # delete parquet after extraction

    random.shuffle(rows_out)
    if max_clips:
        rows_out = rows_out[:max_clips]
    with open(man, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    done_marker.touch()
    print(f"   -> {len(rows_out)} clips written to {out_dir} ({skipped} skipped)")


def main():
    extract(
        CV_REPO, DATA_DIR / "cv_indian",
        text_keys=["text"],
        filters=lambda r: (r.get("down_votes") in (0, None)
                           and (r.get("up_votes") or 0) >= 1),
        max_clips=1500,
    )
    extract(
        NPTEL_REPO, DATA_DIR / "nptel_indian",
        text_keys=["transcription_normalised", "text"],
        filters=lambda r: (r.get("snr") is None or (r.get("snr") or 0) >= 12),
    )
    print("\nIndian-English corpus ready.")


if __name__ == "__main__":
    main()
