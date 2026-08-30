"""Base class for lazy-loaded Cozy plugins (LLM, STT, TTS, Vision)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from ..harness_fast import Plugin  # re-export for convenience


def json_emit_safe(kind, **fields):
    """Emit a JSON event if --json-events is set. Falls back silently
    otherwise. Safe to call at module load time (e.g. inside a plugin's
    _do_load) because it only writes when COZY_TUI_MODE is set.
    """
    if os.environ.get("COZY_TUI_MODE") != "node":
        return
    ev = {"kind": kind, "ts": time.time(), **fields}
    try:
        # Write to the captured real stdout, in case a contextlib
        # redirect_stdout is in effect (used by transformers etc).
        sys.__stdout__.write(json.dumps(ev, ensure_ascii=False) + "\n")
        sys.__stdout__.flush()
    except Exception:
        pass
