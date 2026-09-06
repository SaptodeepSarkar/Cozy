import unittest
from unittest.mock import patch
import numpy as np
import stt
import executor
from rlm_harness.dataset_mode import parse_tool_call
from rlm_harness.harness import RuleBackend

class PipelineTests(unittest.TestCase):
    def test_empty_transcription_is_success(self):
        model = stt.CozySTT()
        with patch.object(stt.CT2_DIR.__class__, 'exists', return_value=True), patch.object(model, '_run_ct2', return_value=''), patch.object(model, '_run_hf') as hf:
            self.assertEqual(model.transcribe_array(np.ones(1600)), '')
            hf.assert_not_called()

    def test_invalid_audio(self):
        model = stt.CozySTT()
        for audio in [np.zeros((2, 10)), np.array([float('nan')])]:
            with self.assertRaises(ValueError):
                model.transcribe_array(audio)

    def test_quoted_tool_arguments(self):
        self.assertEqual(parse_tool_call('tool browser.search query="hello world"')['parameters'], {'query': 'hello world'})

    def test_decisions_do_not_execute(self):
        backend = RuleBackend()
        with patch.object(backend, '_exec') as execute:
            result = backend.decide([{'role': 'user', 'content': 'set volume to 30'}], [])
            execute.assert_not_called()
            self.assertEqual(result['tool']['name'], 'system.volume.set')

    def test_diagnostic_tools(self):
        for name in ['system.info', 'system.disk.usage', 'system.memory.status', 'system.uptime']:
            self.assertTrue(executor.execute(name)['ok'], name)

if __name__ == '__main__':
    unittest.main()
