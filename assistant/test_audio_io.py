from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import audio_io


class AudioRouteTests(unittest.TestCase):
    @patch.object(audio_io.subprocess, "run")
    @patch.object(audio_io, "playback_sink", return_value="test-sink")
    def test_playback_reaches_paplay_with_duration_timeout(self, _sink, run):
        import numpy as np
        run.return_value.returncode = 0
        audio_io.play(np.zeros(2400, dtype=np.float32), 24000)
        self.assertEqual(run.call_args.args[0][0], "paplay")
        self.assertGreaterEqual(run.call_args.kwargs["timeout"], 10.0)

    @patch.object(audio_io, "pipewire_defaults",
                  return_value=("alsa_input.internal", "bluez_output.headset"))
    def test_non_bluetooth_default_is_respected(self, _defaults):
        self.assertEqual(audio_io.capture_source(), "alsa_input.internal")

    @patch.object(audio_io, "_sources", return_value=[
        "bluez_input.headset", "effect_output.rnnoise", "alsa_input.internal"])
    @patch.object(audio_io, "pipewire_defaults",
                  return_value=("bluez_input.headset", "bluez_output.headset"))
    def test_bluetooth_duplex_prefers_noise_filtered_source(self, _defaults, _sources):
        self.assertEqual(audio_io.capture_source(), "effect_output.rnnoise")

    @patch.object(audio_io, "_sources",
                  return_value=["bluez_input.headset", "alsa_input.internal"])
    @patch.object(audio_io, "pipewire_defaults",
                  return_value=("bluez_input.headset", "bluez_output.headset"))
    def test_bluetooth_duplex_falls_back_to_internal(self, _defaults, _sources):
        self.assertEqual(audio_io.capture_source(), "alsa_input.internal")

    @patch.dict(os.environ, {"COZY_ALLOW_BLUETOOTH_MIC": "1"})
    @patch.object(audio_io, "pipewire_defaults",
                  return_value=("bluez_input.headset", "bluez_output.headset"))
    def test_bluetooth_microphone_can_be_enabled(self, _defaults):
        self.assertEqual(audio_io.capture_source(), "bluez_input.headset")


if __name__ == "__main__":
    unittest.main()

class CaptureStreamTests(unittest.TestCase):
    def test_partial_reads_are_assembled_into_complete_pcm_blocks(self):
        from unittest.mock import Mock
        import numpy as np
        blocks = []
        with patch.object(audio_io, 'capture_source', return_value='test'):
            stream = audio_io.PipeWireInputStream(samplerate=16000, channels=1,
                dtype='int16', blocksize=2, callback=lambda pcm, *_: blocks.append(pcm))
        process = Mock()
        process.stdout.read.side_effect = [b'\x01', b'\x00\x02', b'\x00', b'']
        stream._process = process
        stream._read_loop()
        np.testing.assert_array_equal(blocks[0][:, 0], [1, 2])
        with self.assertRaisesRegex(RuntimeError, 'stream stopped'):
            stream.check()

    @patch.object(audio_io, '_run_text', return_value='Mute: yes')
    def test_muted_source_is_reported(self, _run):
        self.assertTrue(audio_io.source_muted('test'))
