from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from rlm_harness import rlm


class DelegationTests(unittest.TestCase):
    def parent(self, root: str):
        parent = Mock()
        parent.cfg = Mock(
            state_dir=Path(root) / "harness", use_wake=False, use_stt=False,
            use_tts=False, use_llm=True, use_vision=False, idle_unload_s=60.0)
        parent.plugins = {}
        parent.trace = Mock()
        return parent

    def test_child_executes_selected_tool_and_does_not_duplicate_task(self):
        with TemporaryDirectory() as root:
            parent = self.parent(root)
            child = Mock()
            child.trace.recent = []
            child.decide.return_value = ("time.now", {})
            with patch.object(rlm, "FastHarness", return_value=child), \
                 patch("executor.execute", return_value={"ok": True, "output": "12:00"}) as execute, \
                 patch("cozy_log.log_event"):
                result = rlm.rlm_delegate("check the time", parent, allow=["time.now"])
            self.assertEqual(result, "12:00")
            child.decide.assert_called_once_with("check the time")
            execute.assert_called_once_with("time.now", {})

    def test_scope_is_enforced_after_model_selection(self):
        with TemporaryDirectory() as root:
            parent = self.parent(root)
            child = Mock()
            child.trace.recent = []
            child.decide.return_value = ("app.open", {"name": "firefox"})
            with patch.object(rlm, "FastHarness", return_value=child), \
                 patch("executor.execute") as execute, patch("cozy_log.log_event"):
                result = rlm.rlm_delegate("open firefox", parent, allow=["time.now"])
            self.assertIn("blocked", result)
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
