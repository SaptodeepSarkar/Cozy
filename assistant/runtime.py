#!/usr/bin/env python3
"""Cozy runtime - ties everything together:

    wake word -> capture -> STT (faster-whisper) -> LLM (Qwen3 tool call)
    -> executor

Mode:
    python runtime.py                 # full voice loop (OpenTUI uses --json-events)
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import select
import signal
import sys
import threading
import time
import tempfile
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", message=r".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=r".*unauthenticated requests.*HF Hub.*")

HERE = Path(__file__).resolve().parent
WW = HERE.parent / "wakeword"

# JSON event emission (used by the OpenTUI frontend)
_json_mode = [False]
_json_out = [None]

# Capture the real stdout at import time, so contextlib.redirect_stdout
# (used by plugin loaders) doesn't swallow our events.
_REAL_STDOUT = sys.stdout
_REAL_STDERR = sys.stderr


def json_emit(kind, **fields):
    """Emit a JSON event to stdout if --json-events is set."""
    if not _json_mode[0]:
        return
    import json as _json, time as _t
    ev = {"kind": kind, "ts": _t.time(), **fields}
    try:
        # Write to the captured real stdout, not whatever the current
        # sys.stdout is (a plugin loader might have redirected it).
        _REAL_STDOUT.write(_json.dumps(ev, ensure_ascii=False) + "\n")
        _REAL_STDOUT.flush()
    except Exception:
        pass


def _flush_emit():
    """Force a flush. The Node TUI needs real-time updates."""
    if _json_mode[0]:
        try:
            sys.stdout.flush()
        except Exception:
            pass

def json_cmd():
    """Read commands from stdin in --json-events mode (one JSON per line)."""
    if not _json_mode[0]:
        return None
    import json as _json, select
    if not select.select([sys.stdin], [], [], 0)[0]:
        return None
    line = sys.stdin.readline().strip()
    if not line:
        return None
    try:
        return _json.loads(line)
    except Exception:
        return None


def _default_wake_threshold() -> float:
    """Return the room-safe threshold.

    The evaluation set's mathematically optimal threshold is intentionally not
    used here: it contains balanced clips, while a live microphone sees hours
    of negatives.  0.31 caused frequent phantom wakes, and 0.50 still fired
    on room noise/TV transients.  Override per room with COZY_WAKE_THRESHOLD
    (or `cozy --threshold`) after a `cozy --calibrate` run.
    """
    return float(os.environ.get("COZY_WAKE_THRESHOLD", "0.60"))


# A lone window must be exceptionally confident to skip the two-hit rule.
# Anything lower needs two adjacent 2 s rolling-window scores, which is
# what rejects single-frame transients (coughs, clicks, TV pops).
WAKE_SINGLE_HIT_BYPASS = 0.95


def _strip_wake_phrase(text: str) -> str:
    """Remove wake phrases included by command pre-roll.

    Loops so repeated phrases ("hey cozy hey cozy, ...") are all removed
    and none leak into the LLM prompt or the spoken reply.
    """
    pattern = r"^\s*(?:(?:hey|hi|okay|ok)\s+)?co[sz]y\b[\s,.:;!\-]*"
    prev = None
    while prev != text:
        prev = text
        text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()
    return text


def _strip_reply_echo(reply: str) -> str:
    """Drop a leading wake-word echo from an LLM reply before display/TTS.

    The model sometimes opens with "Hey Cozy, ..." — speaking that back is
    pure echo. Only the hey/hi/okay-prefixed form is stripped; a bare
    "Cozy here ..." is the assistant referring to itself and is kept.
    Never returns empty: falls back to the original reply.
    """
    cleaned = re.sub(
        r"^\s*(?:hey|hi|okay|ok)\s+co[sz]y\b[\s,.:;!\-]*",
        "", reply, count=1, flags=re.IGNORECASE,
    ).strip()
    return cleaned if cleaned else reply


def _capture_preroll(audio_buf, audio_buf_fill: int, seconds: float,
                     sample_rate: int = 16000):
    """Return a copy of valid audio immediately preceding wake detection."""
    available = max(0, min(len(audio_buf), int(audio_buf_fill)))
    wanted = max(0, min(available, int(seconds * sample_rate)))
    if wanted == 0:
        return np.empty(0, dtype=np.int16)
    return np.asarray(audio_buf[-wanted:], dtype=np.int16).copy()


def run_json_mode(harness, executor, threshold=0.5, *, voice=True, no_wake=False, tts_enabled=True):
    """Run the voice loop and emit NDJSON events to stdout.

    Used by the OpenTUI frontend. Skips the textual app entirely.
    Loads enabled plugins in order, listens for the wake word, transcribes
    the user's command via STT, runs the LLM, executes any tool call,
    speaks the result via TTS, and emits NDJSON events the whole time.
    """
    from tts import is_available, is_speaking, speak as tts_speak
    import contextlib, io, threading, queue as _q, time as _time

    WW_PATH = WW / "output" / "hey_cozy" / "hey_cozy.onnx"
    if voice and not no_wake and not WW_PATH.exists():
        json_emit("error", msg=f"wake model missing: {WW_PATH}")
        return

    # Load in a deterministic order. Loading Torch, CTranslate2, ONNX and
    # Kokoro concurrently caused CPU/GPU/RAM contention and intermittent TTS
    # initialization failures on the 6 GB target machine.
    load_failures = []
    critical_failures = []
    enabled = [name for name in ("wake", "stt", "llm", "foxmcp", "cleanup", "tts")
               if harness.plugins.get(name) is not None
               and not (name == "tts" and not tts_enabled)
               and not (name == "wake" and (not voice or no_wake))
               and not (name == "stt" and not voice)]
    for index, name in enumerate(enabled, start=1):
        p_obj = harness.plugins.get(name)
        started = _time.monotonic()
        json_emit("warmup", model=name, state="loading", index=index,
                  total=len(enabled), progress=(index - 1) / max(1, len(enabled)))
        _flush_emit()
        try:
            p_obj.load()
            if name == "foxmcp":
                tools = getattr(p_obj, "tools", [])
                harness.mcp_tools = tools
                names = ", ".join(str(t.get("name")) for t in tools if t.get("name"))
                if names:
                    harness.system += (
                        "\nFoxMCP browser tools discovered at startup: " + names +
                        ". Use browser.mcp with the exact tool name and arguments."
                    )
            json_emit("warmup", model=name, state="done", index=index,
                      total=len(enabled), elapsed_s=round(_time.monotonic() - started, 2),
                      progress=index / max(1, len(enabled)))
            _flush_emit()
        except Exception as exc:
            load_failures.append(name)
            if name in {"llm", "stt", "tts", "foxmcp"} or (name == "wake" and voice and not no_wake):
                critical_failures.append(name)
            json_emit("warmup", model=name, state="failed", index=index,
                      total=len(enabled), elapsed_s=round(_time.monotonic() - started, 2))
            json_emit("error", msg=f"{name} load: {exc}")
            _flush_emit()
            if critical_failures:
                break
    if critical_failures:
        json_emit("startup_failed", models=critical_failures,
                  msg="Model startup failed: " + ", ".join(critical_failures))
        for plugin in reversed(list(harness.plugins.values())):
            try:
                plugin.free()
            except Exception:
                pass
        return

    # Reuse the warmed plugin objects. The old path created a second wake
    # model, leaving three extra ONNX sessions and their native worker threads
    # alive during interpreter shutdown.
    wake_plugin = harness.plugins.get("wake") if harness else None
    wake = getattr(wake_plugin, "_model", None)
    stt_plugin = harness.plugins.get("stt") if harness else None
    stt = getattr(stt_plugin, "_stt", None)
    cleanup_plugin = harness.plugins.get("cleanup") if harness else None
    if voice and (stt is None or (not no_wake and wake is None)):
        json_emit("error", msg="wake/STT plugin did not initialize")
        return
    vad_model = None
    try:
        if not voice:
            raise ImportError("text mode")
        from silero_vad import load_silero_vad
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            vad_model = load_silero_vad(onnx=True)
    except Exception:
        pass
    wake_name = next(iter(wake._classifiers.keys())) if wake is not None else None

    # Signal: all loaded
    json_emit("ready", voice=voice)

    SR = 16000
    CHUNK = 1280
    WIN = SR * 2
    audio_q = _q.Queue(maxsize=64)
    audio_buf = None  # numpy array
    audio_buf_fill = 0
    stop_flag = threading.Event()
    command_q = _q.Queue(maxsize=8)

    # stdin reader thread - check for "decide" commands
    def stdin_reader():
        from runtime_input import read_commands
        def submit(text):
            try:
                command_q.put_nowait(text)
            except _q.Full:
                json_emit("rejected", reason="command queue is full")
        read_commands(sys.stdin, stop_flag, submit,
                      lambda reason: json_emit("rejected", reason=reason))

    command_lock = threading.Lock()

    def handle_user_text(text):
        if harness is None:
            return
        if not command_lock.acquire(blocking=False):
            json_emit("rejected", reason="another command is still running")
            return
        json_emit("heard", text=text)
        try:
            t0 = _time.time()
            name, args = harness.decide(text)
            dt = _time.time() - t0
            if name == "none" or name == "":
                reply = ""
                for tr in reversed(harness.trace.recent):
                    if tr.role == "assistant" and tr.content:
                        reply = re.sub(r"<\|im_end\|>", "", tr.content)
                        reply = re.sub(r"<\|im_start\|>", "", reply)
                        reply = re.sub(r"<\|endoftext\|>", "", reply)
                        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
                        break
                reply = reply or "I couldn't produce a response."
                reply = _strip_reply_echo(reply)
                json_emit("llm_text", text=reply, dt=dt)
                if tts_enabled and is_available():
                    json_emit("tts", text=reply)
                    tts_speak(reply)
                json_emit("done", text=reply, dt=_time.time() - t0)
            elif name:
                json_emit("llm", tool=name, args=str(args)[:60], dt=dt)
                if name == "rlm.delegate":
                    from rlm_harness.rlm import execute_delegate
                    delegated = execute_delegate(args or {}, harness)
                    result = {"ok": True, "output": delegated}
                else:
                    result = executor(name, args or {})
                output = str(result.get("output", ""))
                if result.get("ok"):
                    reply = output or "Done."
                    json_emit("tool_result", name=name, out=output)
                else:
                    reply = f"Failed: {output or 'the action did not complete.'}"
                    json_emit("tool_fail", name=name, out=output)
                reply = _strip_reply_echo(reply)
                from rlm_harness.harness_fast import Turn
                harness.trace.append(Turn(role="tool", name=name, content=output, producer="tool"))
                harness.trace.append(Turn(role="assistant", content=reply, producer="model"))
                if tts_enabled and is_available():
                    json_emit("tts", text=reply)
                    tts_speak(reply)
                json_emit("done", text=reply, dt=_time.time() - t0)
        except Exception as exc:
            json_emit("error", msg=f"command failed: {exc}")
            reply = "I hit an internal error. Please try that once more."
            if tts_enabled and is_available():
                json_emit("tts", text=reply)
                tts_speak(reply)
            json_emit("done", text=reply)
        finally:
            command_lock.release()

    # Start reading only after handle_user_text exists. Starting the reader
    # earlier created a small startup race for commands typed immediately.
    stdin_thread = threading.Thread(target=stdin_reader, daemon=True)
    stdin_thread.start()

    from audio_io import PipeWireInputStream, capture_source, source_muted
    import numpy as np
    if voice:
        json_emit("audio_route", source=capture_source())
    def audio_cb(indata, *_):
        try:
            audio_q.put_nowait(indata.copy())
        except _q.Full:
            # Preserve live audio instead of allowing an unbounded backlog.
            try:
                audio_q.get_nowait()
            except _q.Empty:
                pass
            try:
                audio_q.put_nowait(indata.copy())
            except _q.Full:
                pass

    def voice_loop():
        nonlocal audio_buf, audio_buf_fill
        audio_buf = np.zeros(WIN, dtype=np.int16)
        audio_buf_fill = 0
        samples_since_score = 0
        cooldown_until = 0.0
        wake_hits = 0
        next_health_check = 0.0
        muted = None
        with PipeWireInputStream(samplerate=SR, channels=1, dtype="int16",
                                 blocksize=CHUNK, callback=audio_cb) as stream:
            while not stop_flag.is_set():
                stream.check()
                if _time.monotonic() >= next_health_check:
                    muted = source_muted(stream.source)
                    json_emit("audio_status", source=stream.source, muted=muted)
                    next_health_check = _time.monotonic() + 2.0
                try:
                    chunk = audio_q.get(timeout=0.5)
                except _q.Empty:
                    continue
                if muted is True:
                    audio_buf[:] = 0
                    audio_buf_fill = 0
                    continue
                if _time.time() < cooldown_until:
                    continue
                if is_speaking():
                    # Never feed Cozy's own spoken reply back into wake/STT.
                    audio_buf[:] = 0
                    audio_buf_fill = 0
                    continue
                chunk = chunk[:, 0].astype(np.int16)
                n = len(chunk)
                audio_buf = np.roll(audio_buf, -n)
                audio_buf[-n:] = chunk
                audio_buf_fill = min(WIN, audio_buf_fill + n)
                samples_since_score += n
                if audio_buf_fill < WIN or samples_since_score < SR * 0.16:
                    continue
                samples_since_score = 0
                if no_wake:
                    score = 1.0
                else:
                    scores = wake.predict(audio_buf.copy())
                    score = float(scores[wake_name])
                json_emit("wake_score", score=score)
                if score < threshold:
                    wake_hits = 0
                    continue
                wake_hits += 1
                # A single marginal score is not enough to interrupt the
                # user. Two adjacent 2 s rolling-window scores are required;
                # only an exceptionally strong score may trigger at once.
                if wake_hits < 2 and score < WAKE_SINGLE_HIT_BYPASS:
                    continue
                wake_hits = 0
                cooldown_until = _time.time() + 4.0
                json_emit("wake", score=score)
                # Capture 7s + VAD
                text = _capture(stt, audio_q, audio_buf, audio_buf_fill)
                if not text:
                    audio_buf[:] = 0
                    audio_buf_fill = 0
                    continue
                # Self-feedback guard
                if _is_self_feedback(harness, text):
                    json_emit("rejected", reason="self-feedback (TTS echo)")
                    cooldown_until = _time.time() + 6.0
                    audio_buf[:] = 0
                    audio_buf_fill = 0
                    continue
                try:
                    command_q.put_nowait(text)
                except _q.Full:
                    json_emit("rejected", reason="command queue is full")
                audio_buf[:] = 0
                audio_buf_fill = 0

    def _capture(stt, audio_q, audio_buf, audio_buf_fill):
        import soundfile as sf
        from pathlib import Path as _P
        json_emit("stt_start")
        # Keep a short pre-roll because the 2-second wake classifier may fire
        # after the user has already started the command. The recognized wake
        # phrase is stripped from the transcript below.
        pre_roll = max(0.0, min(1.5, float(os.environ.get("COZY_COMMAND_PREROLL", "0.8"))))
        initial = _capture_preroll(audio_buf, audio_buf_fill, pre_roll)
        frames = [initial] if initial.size else []
        levels = []
        silent_for = 0.0
        spoken = False
        voiced_seconds = 0.0
        min_capture = max(0.3, float(os.environ.get("COZY_CAPTURE_MIN", "0.55")))
        max_capture = max(min_capture, float(os.environ.get("COZY_CAPTURE_TIMEOUT", "10")))
        silence_after = max(0.55, float(os.environ.get("COZY_SILENCE_AFTER", "0.80")))
        speech_start_timeout = max(0.7, float(os.environ.get("COZY_SPEECH_START_TIMEOUT", "1.20")))
        speech_floor = max(80.0, float(os.environ.get("COZY_SPEECH_RMS", "160")))
        if vad_model is not None and hasattr(vad_model, "reset_states"):
            vad_model.reset_states()
        t0 = _time.time()
        while _time.time() - t0 < max_capture:
            try:
                chunk = audio_q.get(timeout=0.05)
            except _q.Empty:
                chunk = None
            if chunk is not None:
                pcm = chunk[:, 0]
                frames.append(pcm.copy())
                chunk_vad_speech = 0
                chunk_vad_quiet = 0
                level = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
                levels.append(level)
                json_emit("capture_level", level=min(1.0, level / 2000.0))
                if vad_model is not None:
                    import torch
                    block = pcm.astype(np.float32) / 32768.0
                    for start in range(0, len(block), 512):
                        part = block[start:start + 512]
                        if len(part) < 512: part = np.pad(part, (0, 512 - len(part)))
                        try:
                            probability = float(vad_model(torch.from_numpy(part), 16000).item())
                        except Exception:
                            probability = 1.0 if level > speech_floor else 0.0
                        if probability >= 0.35: chunk_vad_speech += 1
                        else: chunk_vad_quiet += 1
                # Energy remains a fallback even when Silero is loaded. The
                # old code trusted VAD exclusively and could record until the
                # timeout without ever marking speech as started.
                if (chunk_vad_speech >= 1) or level >= speech_floor:
                    spoken = True
                    voiced_seconds += len(pcm) / 16000
                    silent_for = 0.0
                elif spoken and (vad_model is None or chunk_vad_quiet >= 1):
                    silent_for += len(pcm) / 16000
            if spoken and (_time.time() - t0) >= min_capture and silent_for >= silence_after:
                break
            if not spoken and (_time.time() - t0) >= speech_start_timeout:
                break
        pcm = np.concatenate(frames) if frames else np.zeros(16000, np.int16)
        energy = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))) if pcm.size else 0.0
        peak = int(np.max(np.abs(pcm))) if pcm.size else 0
        clipped = float(np.mean(np.abs(pcm) >= 32760)) if pcm.size else 0.0
        json_emit("audio_profile", rms=round(energy, 1), peak=peak, clipped=round(clipped, 4), noise=round(float(np.median(levels[:5])) if levels else 0.0, 1))
        # Never send silence/noise to Whisper: generative STT models can turn
        # it into fluent-looking sentences. Require verified post-wake speech,
        # not merely energy in the pre-roll containing the wake phrase.
        if not spoken or voiced_seconds < 0.16:
            json_emit("rejected", reason="No speech detected after wake word")
            return ""
        if energy < 80:
            json_emit("rejected", reason=f"low energy ({energy:.0f})")
            return ""
        fd, tmp_name = tempfile.mkstemp(prefix="cozy_cmd_", suffix=".wav")
        os.close(fd)
        tmp = _P(tmp_name)
        sf.write(str(tmp), pcm, 16000, subtype="PCM_16")
        if os.environ.get("COZY_AUDIO_DEBUG_WAV", "0") == "1":
            sf.write("/tmp/cozy_last_capture.wav", pcm, 16000, subtype="PCM_16")
        try:
            json_emit("stt_infer")
            text = stt.transcribe_file(str(tmp))
        except Exception as exc:
            json_emit("error", msg=f"transcription failed: {exc}")
            return ""
        finally:
            try: tmp.unlink()
            except OSError: pass
        text = _strip_wake_phrase((text or "").strip())
        if len(text) < 3:
            json_emit("rejected", reason="STT returned no speech")
            return ""
        if not any(c.isalpha() for c in text):
            json_emit("rejected", reason="No recognizable command; please try again")
            return ""
        if cleanup_plugin is not None:
            try:
                cleaned = cleanup_plugin.clean(text)
                if cleaned:
                    text = cleaned
            except Exception as exc:
                # Input cleanup is quality enhancement, never a reason to lose
                # a successfully recognized command.
                json_emit("error", msg=f"transcript cleanup failed: {exc}")
        json_emit("transcribed", text=text)
        return text

    def _is_self_feedback(harness, text):
        if harness is None:
            return False
        last = ""
        for tr in reversed(harness.trace.recent):
            if tr.role == "assistant" and tr.content:
                last = re.sub(r"<\|im_end\|>", "", tr.content).strip()
                break
        if not last:
            return False
        from difflib import SequenceMatcher
        a = " ".join(re.findall(r"[a-z0-9]+", last.lower()))
        b = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
        if len(a.split()) < 4 or len(b.split()) < 4:
            return False
        return SequenceMatcher(None, a, b).ratio() >= 0.88

    def voice_runner():
        try:
            voice_loop()
        except Exception as exc:
            json_emit("error", msg=f"audio loop stopped: {exc}")

    voice_thread = threading.Thread(target=voice_runner, daemon=True)
    if voice:
        voice_thread.start()

    def command_worker():
        while not stop_flag.is_set():
            try:
                text = command_q.get(timeout=0.2)
            except _q.Empty:
                continue
            if text is None:
                break
            try:
                handle_user_text(text)
            finally:
                command_q.task_done()

    command_thread = threading.Thread(
        target=command_worker, daemon=True, name="cozy-command")
    command_thread.start()

    # Main thread: idle until stdin / signal
    import signal
    signal.signal(signal.SIGTERM, lambda *_a: stop_flag.set())
    signal.signal(signal.SIGINT, lambda *_a: stop_flag.set())
    try:
        while not stop_flag.is_set():
            _time.sleep(0.2)
    finally:
        stop_flag.set()
        try:
            command_q.put_nowait(None)
        except _q.Full:
            pass
        if voice:
            voice_thread.join(timeout=5)
        command_thread.join(timeout=30)
        stdin_thread.join(timeout=1)
        # Explicitly destroy native inference objects while Python's GIL and
        # module state are still intact. The captured crashes occurred when
        # these destructors ran later from Py_Finalize.
        for plugin in reversed(list(harness.plugins.values())):
            try:
                plugin.free()
            except Exception:
                pass
        vad_model = None
        wake = None
        stt = None
        import gc
        gc.collect()
        json_emit("shutdown")
        _flush_emit()
        # Never enter Py_Finalize with an unresponsive native model thread.
        # The OS safely reclaims process resources without generating a core.
        if voice_thread.is_alive() or command_thread.is_alive():
            os._exit(0)

SR = 16000
CHUNK = 1280  # 80 ms - audio capture granularity
WIN_SAMPLES = SR * 2  # 2 seconds - livekit-wakeword inference window

# The assistant venv installs livekit-wakeword directly. Do not inject the
# wakeword venv's site-packages: it carries a different Transformers/
# huggingface-hub stack and would shadow the versions used by Qwen3 and
# Kokoro. Only the in-tree package path is needed for local model assets.
if str(WW / "src") not in sys.path:
    sys.path.insert(0, str(WW / "src"))


# ------------------------------------------------------------------ loading
def load_wake(threshold):
    """Load the livekit-wakeword model for hey_cozy.

    The model is a small ONNX (122 KB) that takes a 2-second 16kHz int16
    audio window and returns a wake-word score in [0, 1]. Trained on 138
    user-voice positives + 500 synth positives + 2568 negatives.
    """
    from wake_engine import create_wake_model
    model_path = WW / "output" / "hey_cozy" / "hey_cozy.onnx"
    if not model_path.exists():
        raise SystemExit(
            f"wake model not found: {model_path}\n"
            f"train it first:\n"
            f"  cd {WW}\n"
            f"  uv run livekit-wakeword setup --config configs/hey_cozy_test.yaml --skip-acav\n"
            f"  uv run livekit-wakeword run configs/hey_cozy_test.yaml"
        )
    m = create_wake_model(model_path)
    name = next(iter(m._classifiers.keys()))
    print(f"[wake] loaded livekit-wakeword model '{name}' threshold={threshold}")
    return m, name, threshold


def load_stt():
    # stt-agent's dual-engine wrapper: fast CT2 (v1, English) with automatic
    # transformers-HF (v4, Hinglish-aware) fallback when CT2 returns empty.
    from stt import CozySTT

    wrapper = CozySTT()
    ct2 = HERE.parent / "stt-finetune" / "output" / "cozy_stt_v1_ct2_int8"
    hf = HERE.parent / "stt-finetune" / "output" / "hf_finetuned"
    if not (ct2.exists() or hf.exists()):
        raise SystemExit("no STT model found - ask the stt-agent")

    class _Segment:
        def __init__(self, text):
            self.text = text

    class _Adapter:
        """Mimics faster-whisper transcribe() -> (segments, info) API."""
        def transcribe(self, path, language="en", beam_size=1):
            text = wrapper.transcribe_file(path)
            return [_Segment(text)], None

    print("[stt] CozySTT dual-engine (fast-ct2 / hinglish-hf fallback)")
    return _Adapter()


def load_llm(use_dpo=True):
    import json as _json
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = HERE / "model" / "cozy-llm-v1"
    if not path.exists():
        raise SystemExit("LLM not fine-tuned yet - run sft_qwen.py")
    # Qwen3 checkpoints saved by newer Transformers store this field as a
    # list, while the 4.x loader expects a token->token mapping. Supplying it
    # explicitly keeps the pinned runtime compatible without editing model
    # artifacts.
    _tok_cfg = path / "tokenizer_config.json"
    _extra = {}
    if _tok_cfg.exists():
        try:
            _raw = _json.loads(_tok_cfg.read_text()).get("extra_special_tokens", [])
            _extra = {str(token): str(token) for token in _raw} if isinstance(_raw, list) else {}
        except (OSError, ValueError, TypeError):
            pass
    tok = AutoTokenizer.from_pretrained(str(path), extra_special_tokens=_extra)
    model = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16)
    # DPO adapter improves tool-call precision (78% vs 25% on the verifier
    # probe set) but tends to over-fire on chitchat ("hello" -> time.now).
    # Default: SFT only. Use --dpo to opt into the RLVR'd adapter.
    dpo_adapter = HERE / "model" / "cozy-llm-v1-dpo"
    if use_dpo and (dpo_adapter / "adapter_model.safetensors").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(dpo_adapter))
        model = model.merge_and_unload()
        print("[llm] loaded", path.name, "+ DPO adapter (post-RLVR)")
    else:
        print("[llm] loaded", path.name, "(DPO post-RLVR; pass --sft-only for the SFT model alone)")
    model.to("cuda")
    model.eval()
    return tok, model


import torch  # noqa: E402


# ------------------------------------------------------------------- llm io
def _strip_special(text):
    """Strip Qwen3 special tokens and artifacts that shouldn't be spoken."""
    if not text:
        return ""
    import re
    # Qwen3 end-of-turn token and similar
    text = re.sub(r"<\|im_end\|>", "", text)
    text = re.sub(r"<\|im_start\|>", "", text)
    text = re.sub(r"<\|endoftext\|>", "", text)
    # Strip any residual think blocks
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.S)
    return text.strip()


def llm_decide_fast(user_text, fast_harness):
    """Use the optimized harness_fast for RAM + context efficiency."""
    name, args = fast_harness.decide(user_text)
    if name:
        from executor import execute
        result = execute(name, args)
        from rlm_harness.harness_fast import Turn
        from rlm_harness.truncate import truncate_tail
        output = truncate_tail(str(result.get("output", "")).strip())
        if result.get("ok"):
            text = f"Done. {output}"
        else:
            text = f"Failed: {output}"
        fast_harness.trace.append(Turn(role="tool", name=name,
                                       content=output, producer="tool"))
        fast_harness.trace.append(Turn(role="assistant", content=text,
                                       producer="model"))
        return _strip_special(text), None
    # Plain text reply - last turn is the assistant's reply
    if fast_harness.trace.recent:
        return _strip_special(fast_harness.trace.recent[-1].content), None
    return "...", None


def llm_decide(tok, model, user_text):
    schema = json.loads(
        (HERE.parent / "team" / "tool_schema.json").read_text())["tools"]
    system = (
        "You are Cozy, a voice assistant running fully offline on the "
        "user's laptop. Respond fast and short. When the user wants an "
        "action, call exactly one tool with compact JSON. For plain chat, "
        "answer briefly and warmly without tools."
    )
    prompt = tok.apply_chat_template(
        # Match the non-thinking generation used by the training smoke test.
        [{"role": "system", "content": system},
         {"role": "user", "content": user_text}],
        tools=schema,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=512, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=False).strip()
    # defensive: drop any residual Qwen3 thinking block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    # Qwen3 emits tool calls in <tool_call>...</tool_call> tags
    m_tag = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.S)
    candidates = []
    if m_tag:
        candidates.append(m_tag.group(1))
    # Balanced-brace fallback for the cases where the model drops the tags
    # but still emits a valid JSON object with a "name" field.
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                chunk = text[start:i + 1]
                try:
                    call = json.loads(chunk)
                    if isinstance(call, dict) and isinstance(call.get("name"), str):
                        candidates.append(chunk)
                except json.JSONDecodeError:
                    pass
                start = None
    # Also try simple non-nested braces
    candidates += re.findall(r"\{[^{}]*\}", text)

    # Load valid tool names for normalization
    valid_tools = {t["name"] for t in json.loads(
        (HERE.parent / "team" / "tool_schema.json").read_text())["tools"]}

    def normalize_tool_name(name):
        """The model sometimes hallucinates prefix variants (system.mute,
        app.list-running, system.calc.compute, ...). Snap the closest
        valid tool name to whatever the model emitted, so the executor
        still gets called.
        """
        if not isinstance(name, str):
            return name
        if name in valid_tools:
            return name
        # Dash-to-underscore
        candidate = name.replace("-", "_")
        if candidate in valid_tools:
            return candidate
        # Try stripping a sequence of hallucinated prefixes. The model
        # often emits "system.app.X" when the real tool is "X" (or "app.X").
        for prefix in ("system.", "app.", "browser.", "media."):
            if name.startswith(prefix):
                candidate = name[len(prefix):]
                if candidate in valid_tools:
                    return candidate
                # Recurse: maybe two prefixes (system.app.list_running)
                for prefix2 in ("system.", "app.", "browser.", "media."):
                    if candidate.startswith(prefix2):
                        candidate2 = candidate[len(prefix2):]
                        if candidate2 in valid_tools:
                            return candidate2
        # Try matching the last segment (the leaf after the last dot)
        if "." in name:
            leaf = name.rsplit(".", 1)[-1]
            for vt in valid_tools:
                if vt.endswith("." + leaf):
                    return vt
        # Substring fallback: find any valid tool whose name contains the
        # last segment of `name`. Catches "system.mute" -> system.volume.mute,
        # "system.read_notes" -> note.read, etc.
        if "." in name:
            leaf = name.rsplit(".", 1)[-1].lower()
            if len(leaf) >= 4:  # avoid matching "set", "now", "all" etc
                candidates = [vt for vt in valid_tools if leaf in vt.lower()]
                if len(candidates) == 1:
                    return candidates[0]
                # Try the second-to-last segment too (handles "read_notes" -> "note.read")
                parts = name.lower().split(".")
                for p in reversed(parts):
                    if len(p) >= 4:
                        cands = [vt for vt in valid_tools if p in vt.lower()]
                        if len(cands) == 1:
                            return cands[0]
        # Last resort: case-insensitive match
        lname = name.lower()
        for v in valid_tools:
            if v.lower() == lname:
                return v
        return name  # let the executor return "unknown tool"

    for c in candidates:
        try:
            call = json.loads(c)
            if isinstance(call.get("name"), str):
                params = call.get("parameters") or call.get("arguments") or call.get("params") or {}
                fixed_name = normalize_tool_name(call["name"])
                return None, {"name": fixed_name, "parameters": params}
        except json.JSONDecodeError:
            continue
    return text or "...", None


# ------------------------------------------------------------------- audio
def record_command(stt_model, max_seconds=7.0, silence_after=1.1):
    import soundfile as sf
    from audio_io import PipeWireInputStream

    frames = []
    silent_for = 0.0
    spoken = False
    started = time.time()

    def cb(indata, _f, _t, _s):
        nonlocal silent_for, spoken
        pcm = indata[:, 0]
        frames.append(pcm.copy())
        level = float(np.abs(pcm).mean())
        if level > 300:
            spoken = True
            silent_for = 0.0
        elif spoken:
            silent_for += len(pcm) / 16000.0

    with PipeWireInputStream(samplerate=16000, channels=1, dtype="int16",
                             blocksize=1280, callback=cb):
        while time.time() - started < max_seconds:
            time.sleep(0.05)
            if spoken and silent_for >= silence_after:
                break
    pcm = np.concatenate(frames) if frames else np.zeros(16000, np.int16)

    tmp = Path("/tmp/cozy_cmd.wav")
    sf.write(str(tmp), pcm, 16000, subtype="PCM_16")
    segments, _info = stt_model.transcribe(str(tmp), language="en",
                                           beam_size=1)
    text = " ".join(s.text for s in segments).strip()
    return text, str(tmp)


# --------------------------------------------------------------------- main
def handle_text(text, tok, llm, speak, fast_harness=None):
    if not text:
        return
    print("[you]", text)
    if fast_harness is not None:
        # Fast path: the harness already recorded the tool result + affirmation
        # in the trace, so we just speak the latest assistant turn.
        reply, call = llm_decide_fast(text, fast_harness)
        print("[cozy]", reply)
        speak(reply)
        return
    reply, call = llm_decide(tok, llm, text)
    if call is not None:
        from executor import execute
        # Route rlm.delegate() to the RLM child-agent spawner, not the
        # regular executor. The child shares the parent's plugins.
        if call["name"] == "rlm.delegate":
            from rlm_harness.rlm import execute_delegate
            result_text = execute_delegate(
                call.get("parameters") or {}, fast_harness or _get_parent_harness())
            print("[cozy] (delegate) " + result_text[:200])
            speak(result_text)
            return
        result = execute(call["name"], call.get("parameters") or {})
        out = result["output"]
        # Truncate long outputs (e.g. 8KB web search results, 4KB STT
        # transcripts) before they reach the TTS or the LLM context.
        from rlm_harness.truncate import truncate_tail
        out = truncate_tail(out)
        prefix = "Done. " if result["ok"] else "Failed: "
        print("[cozy]", prefix, out)
        speak(prefix + out)
    else:
        print("[cozy]", reply)
        speak(reply)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-wake", action="store_true",
                        help="skip wake word gate (voice loop still on)")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--calibrate", action="store_true",
                        help="print live wake scores for 30 s")
    parser.add_argument("--no-tts", action="store_true",
                        help="suppress spoken replies (still log them)")
    parser.add_argument("--sft-only", action="store_true",
                        help="use the SFT-only merged model (default is SFT+DPO)")
    parser.add_argument("--json-events", action="store_true",
                        help="emit NDJSON events on stdout for the Node TUI")
    args = parser.parse_args()
    _json_mode[0] = args.json_events
    if _json_mode[0]:
        # Disable stdout buffering so events are visible immediately
        import sys as _sys
        _sys.stdout.reconfigure(line_buffering=True)

    # OpenTUI consumes the JSON event stream. Direct runtime invocation keeps
    # the same voice engine without providing a second user interface.
    if args.json_events:
        from rlm_harness.harness_fast import FastHarness, HarnessConfig
        from executor import execute as executor_execute
        cfg = HarnessConfig()
        cfg.use_wake = not args.no_wake
        cfg.use_stt = True
        cfg.use_llm = True
        cfg.use_cleanup = True
        cfg.use_tts = not args.no_tts
        # These plugins are owned by the continuous runtime until shutdown.
        cfg.idle_unload_s = float("inf")
        h = FastHarness(cfg)
        threshold = args.threshold if args.threshold is not None else _default_wake_threshold()
        if not 0.0 <= threshold <= 1.0:
            parser.error("--threshold must be between 0 and 1")
        run_json_mode(h, executor_execute, threshold,
                      voice=True, no_wake=args.no_wake, tts_enabled=not args.no_tts)
        return


    # Default threshold comes from the trained model's eval JSON
    # (AUT/FPPH/recall-optimal threshold computed during livekit training)
    threshold = args.threshold
    if threshold is None:
        threshold = _default_wake_threshold()
    if not 0.0 <= threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    print("[config] threshold =", threshold)

    # --calibrate: print live wake scores for 30s
    if args.calibrate:
        from audio_io import PipeWireInputStream
        m, name, thr = load_wake(threshold)
        print("Calibrating 30s - say 'hey cozy' and other stuff...")
        q2 = queue.Queue()

        def cb2(indata, _f, _t, _s):
            q2.put(indata.copy())

        audio_buf2 = np.zeros(WIN_SAMPLES, dtype=np.int16)
        with PipeWireInputStream(samplerate=16000, channels=1, dtype="int16",
                                 blocksize=CHUNK, callback=cb2):
            end = time.time() + 30
            while time.time() < end:
                try:
                    chunk = q2.get(timeout=0.25)
                except queue.Empty:
                    continue
                chunk = chunk[:, 0].astype(np.int16)
                n = len(chunk)
                audio_buf2 = np.roll(audio_buf2, -n)
                audio_buf2[-n:] = chunk
                if audio_buf2.shape[0] < WIN_SAMPLES:
                    continue
                scores = m.predict(audio_buf2.copy())
                score = float(scores[name])
                bar = "#" * int(min(score, 1.0) * 40)
                fired = " <-- WAKE" if score >= thr else ""
                print(f"\r{bar.ljust(40)} {score:.3f} (thr {thr}){fired}    ",
                      end="", flush=True)
        print()
        return

    from rlm_harness.harness_fast import FastHarness, HarnessConfig
    from executor import execute as executor_execute
    cfg = HarnessConfig()
    cfg.use_wake = not args.no_wake
    cfg.use_stt = True
    cfg.use_llm = True
    cfg.use_cleanup = True
    cfg.use_tts = not args.no_tts
    cfg.idle_unload_s = float("inf")
    h = FastHarness(cfg)
    run_json_mode(h, executor_execute, threshold,
                  voice=True, no_wake=args.no_wake, tts_enabled=not args.no_tts)


if __name__ == "__main__":
    main()
