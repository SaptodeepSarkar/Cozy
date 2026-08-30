"""Cozy logging - rotating JSONL with append-only write.

Lifted from Prime Agent logging pattern. The 4 rules:
  1. Append-only. Never truncate. Whole session recoverable from log.
  2. Rotate at size cap (default 20 MB). Old file becomes .old.
  3. Every try/except has a comment explaining why we keep going.
  4. Single-process writer. No concurrency.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

LOG_DIR = Path.home() / ".cozy" / "logs"
LOG_FILE = LOG_DIR / "cozy.jsonl"
MAX_BYTES = 20 * 1024 * 1024  # 20 MB


def append_rotating_log(entry, log_file=None, max_bytes=MAX_BYTES):
    """Append a JSON-encoded entry to the log. Rotate at max_bytes.

    Errors writing the log are swallowed (a log failure must never
    break the caller) but each swallow has a comment explaining why.
    """
    if log_file is None:
        log_file = LOG_FILE
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            if log_file.exists() and log_file.stat().st_size > max_bytes:
                old = log_file.with_suffix(log_file.suffix + ".old")
                try:
                    old.unlink(missing_ok=True)
                except OSError:
                    pass
                os.replace(log_file, old)
        except OSError:
            pass
        if "ts" not in entry:
            entry["ts"] = time.time()
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with open(log_file, "a", buffering=1) as f:
            f.write(line)
    except OSError:
        pass
    except Exception:
        pass


def log_event(kind, **fields):
    """Helper for the common case of one structured event."""
    append_rotating_log({"kind": kind, **fields})


if __name__ == "__main__":
    log_event("startup", version="2.0", component="cozy-fast-harness")
    log_event("test", msg="hello")
    print(f"log written to {LOG_FILE}")
