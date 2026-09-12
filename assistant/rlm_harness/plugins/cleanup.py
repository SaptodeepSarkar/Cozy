"""Persistent ArchFlow-compatible input cleanup plugin."""
from __future__ import annotations

from ..harness_fast import Plugin


class CleanupPlugin(Plugin):
    name = "cleanup"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._cleaner = None

    def _do_load(self):
        from transcript_cleanup import TranscriptCleaner
        self._cleaner = TranscriptCleaner()
        self._cleaner.load()

    def clean(self, text: str) -> str:
        assert self._loaded and self._cleaner is not None
        result = self._cleaner.clean(text)
        self.touch()
        return result

    def _do_free(self):
        if self._cleaner is not None:
            self._cleaner.close()
        self._cleaner = None
