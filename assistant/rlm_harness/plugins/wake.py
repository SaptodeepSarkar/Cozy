"""Wake plugin: livekit-wakeword. Lazy loaded.

Memory budget:
  - livekit-wakeword ONNX: ~10 MB RAM (CPU)
"""
from __future__ import annotations

import os
import sys
import json
import time
import numpy as np

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


class WakePlugin(Plugin):
    name = "wake"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._model = None
        self._name = None
        self._ww = ASSISTANT.parent / "wakeword"
        self._ww_dir = ASSISTANT.parent / "wakeword"
        self._use_dpo = False

    def _do_load(self):
        # livekit-wakeword lives in wakeword/.venv. Add it to sys.path.
        if str(self._ww) not in sys.path:
            sys.path.insert(0, str(self._ww))
        import contextlib, io
        from rlm_harness.plugins._base import json_emit_safe as _emit
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            from livekit.wakeword import WakeWordModel
            model_path = self._ww_dir / "output" / "hey_cozy" / "hey_cozy.onnx"
            self._model = WakeWordModel(models=[model_path])
        self._name = next(iter(self._model._classifiers.keys()))
        _emit("warmup", model="wake", state="done")

    def _do_free(self):
        self._model = None

    def score_window(self, audio_2s):
        assert self._loaded, "call .load() first"
        scores = self._model.predict(audio_2s)
        score = float(scores[self._name])
        self.touch()
        return score
