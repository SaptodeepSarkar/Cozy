"""TTS plugin: Kokoro-82M. Lazy loaded on first speak().

Memory budget:
  - Kokoro model: ~400 MB RAM (CPU)
"""
from __future__ import annotations

import os
import sys
import json
import time

from ..harness_fast import Plugin, ASSISTANT


def json_emit_safe(kind, **fields):
    """Emit a JSON event if --json-events is set. Falls back silently otherwise."""
    if os.environ.get("COZY_TUI_MODE") != "node":
        return
    ev = {"kind": kind, "ts": time.time(), **fields}
    try:
        sys.stdout.write(json.dumps(ev, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


class TTSPlugin(Plugin):
    name = "tts"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._tts = None

    def _do_load(self):
        import contextlib, io, sys
        sys.path.insert(0, str(ASSISTANT))
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            import tts
        self._tts = tts
        # Warm Kokoro while the startup screen is visible. This avoids the
        # first reply silently waiting on model download/initialization.
        if os.environ.get("COZY_SKIP_TTS_WARMUP", "0") != "1":
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                pipeline = self._tts._get_pipeline()
            if pipeline is None:
                raise RuntimeError("Kokoro TTS could not be initialized")
        from ._base import json_emit_safe as _emit
        _emit("warmup", model="tts", state="done")

    def _do_free(self):
        self._tts = None

    def speak(self, text):
        if not self._loaded:
            self.load()
        if self._tts is not None:
            self._tts.speak(text)
            self.touch()
