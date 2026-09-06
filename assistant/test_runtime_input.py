import os
import threading
import unittest
from runtime_input import read_commands


class CommandInputTests(unittest.TestCase):
    def test_batched_fragmented_and_invalid_commands(self):
        reader, writer = os.pipe()
        stop = threading.Event()
        accepted, errors = [], []
        with os.fdopen(reader, 'rb', buffering=0) as stream:
            thread = threading.Thread(target=read_commands, args=(stream, stop, accepted.append, errors.append))
            thread.start()
            os.write(writer, b'{"cmd":"decide","te')
            os.write(writer, b'xt":"hello"}\n{"cmd":"decide","text":"time"}\ninvalid\n[]\n')
            os.close(writer)
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(accepted, ['hello', 'time'])
        self.assertEqual(len(errors), 2)
        self.assertTrue(stop.is_set())

    def test_reader_can_stop_without_input(self):
        reader, writer = os.pipe()
        stop = threading.Event()
        try:
            with os.fdopen(reader, 'rb', buffering=0) as stream:
                thread = threading.Thread(target=read_commands, args=(stream, stop, lambda _: None, lambda _: None))
                thread.start()
                stop.set()
                thread.join(timeout=1)
                self.assertFalse(thread.is_alive())
        finally:
            os.close(writer)
