import unittest
from unittest.mock import Mock, patch
from rlm_harness.harness_fast import FastHarness


class DecisionRecoveryTests(unittest.TestCase):
    def harness(self, outputs):
        harness = FastHarness.__new__(FastHarness)
        harness.system = 'You are Cozy.'
        harness.trace = Mock()
        harness.trace.build_prompt.return_value = ([{'role': 'user', 'content': 'time?'}], 10)
        harness.tools = Mock()
        llm = Mock()
        llm.generate.side_effect = outputs
        harness.plugins = {'llm': llm}
        return harness

    def test_none_retries_once_and_accepts_exact_parameterless_tool(self):
        h = self.harness(['none', 'time.now'])
        with patch('cozy_log.log_event'):
            self.assertEqual(h.decide('what time is it'), ('time.now', {}))
        self.assertEqual(h.plugins['llm'].generate.call_count, 2)

    def test_repeated_none_produces_readable_failure(self):
        h = self.harness(['none', 'none'])
        with patch('cozy_log.log_event'):
            self.assertEqual(h.decide('hello'), ('', {}))
        self.assertIn("couldn't understand", h.trace.append.call_args.args[0].content)

    def test_bare_name_never_invents_required_parameters(self):
        h = self.harness(['system.volume.set'])
        with patch('cozy_log.log_event'):
            self.assertEqual(h.decide('change volume'), ('', {}))
