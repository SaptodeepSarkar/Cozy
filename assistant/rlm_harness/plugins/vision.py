"""Vision plugin: cozy-vision harness (Qwen2.5-VL + UI-TARS).

Memory budget:
  - Qwen2.5-VL-3B-AWQ: 3.4 GB VRAM
  - UI-TARS-2B (offloaded to CPU): 9.5 GB RAM
  - Both auto-loaded by ``VisionRunner.warmup()`` on first use.
"""
from __future__ import annotations
from ..harness_fast import Plugin, ASSISTANT


class VisionPlugin(Plugin):
    name = "vision"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._runner = None

    def _do_load(self) -> None:
        import sys
        cv = ASSISTANT.parent / "cozy-vision"
        if str(cv) not in sys.path:
            sys.path.insert(0, str(cv))
        from harness.runner import VisionRunner
        self._runner = VisionRunner()
        self._runner.warmup()
        print(f"[vision] planner + grounder loaded")

    def _do_free(self) -> None:
        if self._runner is not None:
            self._runner.shutdown()
        self._runner = None

    def run(self, goal: str) -> dict:
        assert self._loaded, "call .load() first"
        r = self._runner.run(goal)
        self.touch()
        return {
            "success": r.success,
            "score": r.score,
            "sub_goals": r.sub_goals,
            "actions": r.actions,
            "duration_s": r.duration_s,
            "error": r.error,
        }
