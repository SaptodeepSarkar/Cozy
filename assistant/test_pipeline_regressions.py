import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import numpy as np
import stt
import runtime
import tts
import executor
from transcript_cleanup import TranscriptCleaner, _safe_rewrite, polish_transcript
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

    def test_app_listing_never_falls_back_to_service_processes(self):
        with patch.object(executor, '_which_any', return_value=None):
            ok, output = executor.app_list_running()
        self.assertFalse(ok)
        self.assertIn('desktop windows', output)

    def test_terminal_tool_runs_in_requested_directory(self):
        ok, output = executor.terminal_run({'command': 'printf cozy', 'cwd': '/tmp'})
        self.assertTrue(ok)
        self.assertEqual(output, 'cozy')

    def test_terminal_tool_can_read_system_but_cannot_write_outside_home_or_tmp(self):
        ok, output = executor.terminal_run({'command': 'head -n 1 /etc/hosts', 'cwd': '/tmp'})
        self.assertTrue(ok)
        self.assertTrue(output)
        blocked = Path('/etc/cozy_terminal_test_forbidden')
        self.assertFalse(blocked.exists())
        ok, output = executor.terminal_run({
            'command': f'touch {blocked}', 'cwd': '/tmp'})
        self.assertFalse(ok)
        self.assertIn('Read-only file system', output)
        self.assertFalse(blocked.exists())

    def test_terminal_tool_refuses_elevation_in_the_agent_sandbox(self):
        ok, output = executor.terminal_run({'command': 'sudo id', 'cwd': '/tmp'})
        self.assertFalse(ok)
        self.assertIn('terminal.elevate', output)

    def test_runtime_uses_room_safe_wake_threshold(self):
        self.assertAlmostEqual(runtime._default_wake_threshold(), 0.60)
        with patch.dict("os.environ", {"COZY_WAKE_THRESHOLD": "0.72"}):
            self.assertAlmostEqual(runtime._default_wake_threshold(), 0.72)

    def test_command_preroll_and_wake_phrase_cleanup(self):
        audio = np.arange(32000, dtype=np.int16)
        np.testing.assert_array_equal(
            runtime._capture_preroll(audio, 32000, 0.5), audio[-8000:])
        self.assertEqual(runtime._strip_wake_phrase("Hey Cozy, open Firefox"), "open Firefox")
        self.assertEqual(runtime._strip_wake_phrase("okay cosy: what time is it"), "what time is it")

    def test_repeated_wake_phrases_are_all_stripped(self):
        self.assertEqual(
            runtime._strip_wake_phrase("hey cozy hey cozy, open Firefox"), "open Firefox")
        self.assertEqual(runtime._strip_wake_phrase("Hey Cozy"), "")
        self.assertEqual(runtime._strip_wake_phrase("open Firefox"), "open Firefox")

    def test_reply_echo_never_reaches_tts_empty(self):
        self.assertEqual(
            runtime._strip_reply_echo("Hey Cozy, volume set to 40."), "volume set to 40.")
        self.assertEqual(
            runtime._strip_reply_echo("Cozy here, all done."), "Cozy here, all done.")
        self.assertEqual(runtime._strip_reply_echo("Done."), "Done.")

    def test_current_stt_prefers_complete_v12_artifacts(self):
        self.assertEqual(stt.CT2_DIR.name, "cozy_stt_v1.2_ct2_int8")
        self.assertEqual(stt.HF_DIR.name, "hf_finetuned_v1.2")

    def test_ct2_is_greedy_without_second_vad_cut(self):
        model = stt.CozySTT()
        engine = Mock()
        engine.transcribe.return_value = ([Mock(text=" hello ", no_speech_prob=0.0)], None)
        with patch.object(model, "_get_ct2", return_value=engine):
            self.assertEqual(model._run_ct2(np.ones(1600)), "hello")
        self.assertEqual(engine.transcribe.call_args.kwargs["beam_size"], 1)
        self.assertFalse(engine.transcribe.call_args.kwargs["vad_filter"])
        self.assertTrue(engine.transcribe.call_args.kwargs["initial_prompt"])

    def test_ct2_drops_decoder_marked_no_speech(self):
        model = stt.CozySTT()
        engine = Mock()
        engine.transcribe.return_value = ([
            Mock(text=" invented sentence ", no_speech_prob=0.91),
            Mock(text=" real command ", no_speech_prob=0.08),
        ], None)
        with patch.object(model, "_get_ct2", return_value=engine):
            self.assertEqual(model._run_ct2(np.ones(1600)), "real command")

    def test_archflow_transcript_polish_is_conservative(self):
        self.assertEqual(
            polish_transcript("uh i can't i can't genuine genuinely move"),
            "i can't genuinely move",
        )
        self.assertEqual(polish_transcript("very very good"), "very very good")
        self.assertTrue(_safe_rewrite("set volume to 35 not 50", "Set volume to 35, not 50."))
        self.assertFalse(_safe_rewrite("set volume to 35 not 50", "Set volume to 80."))

    def test_cleanup_falls_back_to_rules_without_external_model(self):
        cleaner = TranscriptCleaner()
        cleaner.model_dir = Path("/missing/base")
        cleaner.adapter_dir = Path("/missing/adapter")
        cleaner.load()
        self.assertEqual(cleaner.clean("um open open firefox"), "open firefox")

    def test_tts_keeps_all_generated_segments_and_hashes_cache_keys(self):
        pipeline = Mock(return_value=iter([
            (None, None, np.array([1.0, 2.0])),
            (None, None, np.array([3.0])),
        ]))
        with patch.object(tts, "_get_pipeline", return_value=pipeline), \
             patch.object(tts, "_load_cfg", return_value={"voice": "test", "speed": 1.0}):
            samples, rate = tts._synthesize("two sentences")
        np.testing.assert_array_equal(samples, [1.0, 2.0, 3.0])
        self.assertEqual(rate, 24000)
        prefix = "same first forty characters exactly here"
        self.assertNotEqual(tts._cache_path(prefix + " A"), tts._cache_path(prefix + " B"))

    def test_common_commands_have_a_zero_llm_fast_path(self):
        from rlm_harness.harness_fast import FastHarness
        cases = {
            "what time is it": ("time.now", {}),
            "set volume to 31": ("system.volume.set", {"level": 31}),
            "Could you put the speakers at thirty seven percent?":
                ("system.volume.set", {"level": 37}),
            "I need complete silence from the laptop.": ("system.volume.mute", {}),
            "Set my display brightness to sixty two.":
                ("system.brightness.set", {"level": 62}),
            "Bring up Firefox for me.": ("app.open", {"name": "firefox"}),
            "Please quit the calculator app.":
                ("app.close", {"name": "gnome-calculator"}),
            "Capture what is on my screen right now.": ("screenshot.take", {}),
            "Resume my music.": ("media.play", {}),
            "Hold the song where it is.": ("media.pause", {}),
        }
        for command, expected in cases.items():
            with self.subTest(command=command):
                self.assertEqual(FastHarness._rule_tool_call(command), expected)
        self.assertEqual(FastHarness._rule_tool_call("tell me a story"), ("", {}))

    def test_incomplete_thought_is_never_returned_or_replayed(self):
        from evaluation import strip_thinking
        from rlm_harness.harness_fast import HarnessConfig, Trace, Turn
        from tempfile import TemporaryDirectory
        self.assertEqual(strip_thinking("<think>a very long unfinished thought"), "")
        with TemporaryDirectory() as root:
            cfg = HarnessConfig(
                state_dir=Path(root), trace_file=Path(root) / "trace.jsonl",
                max_context_tokens=200)
            trace = Trace(cfg)
            trace.append(Turn(role="assistant", content="<think>polluted history"))
            trace.append(Turn(role="user", content="current request"))
            messages, _ = trace.build_prompt("system", "tools")
        rendered = str(messages)
        self.assertNotIn("polluted history", rendered)
        self.assertIn("current request", rendered)

if __name__ == '__main__':
    unittest.main()
