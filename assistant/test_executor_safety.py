from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import executor


class PowerActionSafetyTests(unittest.TestCase):
    @patch.object(executor, "_run")
    def test_shutdown_is_disabled_by_default(self, run):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COZY_ALLOW_POWER_ACTIONS", None)
            ok, message = executor.system_shutdown({"confirm": True})
        self.assertFalse(ok)
        self.assertIn("disabled", message)
        run.assert_not_called()

    @patch.object(executor, "_run")
    def test_string_confirmation_is_rejected(self, run):
        with patch.dict(os.environ, {"COZY_ALLOW_POWER_ACTIONS": "1"}):
            ok, message = executor.system_shutdown({"confirm": "true"})
        self.assertFalse(ok)
        self.assertIn("boolean", message)
        run.assert_not_called()

    @patch.object(executor, "_run")
    def test_app_close_will_not_kill_runtime_dependencies(self, run):
        ok, message = executor.app_close({"name": "python"})
        self.assertFalse(ok)
        self.assertIn("no running app", message)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
