import unittest
from unittest.mock import Mock, patch


class WakeEngineTests(unittest.TestCase):
    def test_runtime_uses_bounded_non_spinning_workers(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'wakeword' / 'src'))
        from wake_engine import create_wake_model
        with patch('livekit.wakeword.WakeWordModel') as model:
            create_wake_model('model.onnx')
        options = model.call_args.kwargs['sess_options']
        self.assertEqual(options.intra_op_num_threads, 1)
        self.assertEqual(options.inter_op_num_threads, 1)
        self.assertEqual(options.get_session_config_entry('session.intra_op.allow_spinning'), '0')
