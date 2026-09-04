"""STT plugin: faster-whisper. Lazy loaded.

Memory budget:
  - CT2 model: ~500 MB VRAM
  - HF Whisper (fallback): ~1 GB VRAM
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

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


class STTPlugin(Plugin):
    name = "stt"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._stt = None
        self._ct2 = ASSISTANT.parent / "stt-finetune" / "output" / "cozy_stt_v1_ct2_int8"
        self._hf = ASSISTANT.parent / "stt-finetune" / "output" / "hf_finetuned"

    def _do_load(self):
        import contextlib, io
        from ._base import json_emit_safe as _emit
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            from stt import CozySTT
            self._stt = CozySTT()
            # Load the speech engine during the startup gate, not on the
            # first command. If CT2 cannot load its CUDA runtime, CozySTT
            # will transparently select its configured fallback on demand.
            try:
                self._stt._get_ct2()
            except Exception:
                pass
        _emit("warmup", model="stt", state="done")

    def _do_free(self):
        self._stt = None

    def transcribe(self, audio_or_path):
        assert self._loaded, "call .load() first"
        if isinstance(audio_or_path, (str, Path)):
            text = self._stt.transcribe_file(str(audio_or_path))
        else:
            text = self._stt.transcribe_array(audio_or_path)
        self.touch()
        return text
