#!/usr/bin/env python3
"""Cozy runtime - ties everything together:

    wake word -> capture -> STT (faster-whisper) -> LLM (Qwen3 tool call)
    -> executor

Modes:
    python runtime.py                 # full voice loop
    python runtime.py --text          # type commands instead (test LLM+exec)
    python runtime.py --no-wake       # skip wake gate, always transcribe
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

# JSON event emission (used by the Node Ink TUI)
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


def run_json_mode(harness, executor, threshold=0.5):
    """Run the voice loop and emit NDJSON events to stdout.

    Used by the Node Ink TUI. Skips the textual app entirely.
    Loads all 4 plugins in parallel, listens for the wake word, transcribes
    the user's command via STT, runs the LLM, executes any tool call,
    speaks the result via TTS, and emits NDJSON events the whole time.
    """
    from livekit.wakeword import WakeWordModel
    from stt import CozySTT
    from tts import is_available, speak as tts_speak
    import contextlib, io, threading, queue as _q, time as _time

    WW_PATH = WW / "output" / "hey_cozy" / "hey_cozy.onnx"
    if not WW_PATH.exists():
        json_emit("error", msg=f"wake model missing: {WW_PATH}")
        return

    # Parallel warmup
    load_failures = []
    def _loader(name):
        p_obj = harness.plugins.get(name)
        if p_obj is None:
            return
        # Emit loading state first so the TUI shows ◐
        json_emit("warmup", model=name, state="loading")
        _flush_emit()
        try:
            p_obj.load()
            json_emit("warmup", model=name, state="done")
            _flush_emit()
        except Exception as exc:
            load_failures.append(name)
            json_emit("warmup", model=name, state="failed")
            json_emit("error", msg=f"{name} load: {exc}")
            _flush_emit()

    threads = []
    for name in ("wake", "stt", "llm", "tts"):
        t = threading.Thread(target=_loader, args=(name,), daemon=True)
        t.start()
        threads.append(t)
    # Wait for wake + STT before starting audio capture
    for t in threads:
        t.join(timeout=60)
    if any(t.is_alive() for t in threads) or load_failures:
        json_emit("error", msg="Model startup did not complete: " + ", ".join(load_failures or ["timeout"]))
        return

    # Now load the wake + STT objects
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        wake = WakeWordModel(models=[str(WW_PATH)])
        stt_plugin = harness.plugins.get("stt") if harness else None
        stt = getattr(stt_plugin, "_stt", None) or CozySTT()
    wake_name = next(iter(wake._classifiers.keys()))

    # Signal: all loaded
    json_emit("ready")

    SR = 16000
    CHUNK = 1280
    WIN = SR * 2
    audio_q = _q.Queue()
    audio_buf = None  # numpy array
    audio_buf_fill = 0
    stop_flag = [False]

    # stdin reader thread - check for "decide" commands
    def stdin_reader():
        import sys as _sys
        for line in _sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = _json.loads(line)
            except Exception:
                continue
            if cmd.get("cmd") == "decide":
                text = cmd.get("text", "")
                if text:
                    # Run the same flow as a voice command
                    threading.Thread(target=handle_user_text, args=(text,), daemon=True).start()

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
                json_emit("llm_text", text=reply, dt=dt)
                if is_available():
                    json_emit("tts", text=reply)
                    tts_speak(reply)
                json_emit("done", text=reply, dt=_time.time() - t0)
            elif name:
                json_emit("llm", tool=name, args=str(args)[:60], dt=dt)
                result = executor(name, args or {})
                output = str(result.get("output", ""))
                if result.get("ok"):
                    reply = output or "Done."
                    json_emit("tool_result", name=name, out=output)
                else:
                    reply = f"Failed: {output or 'the action did not complete.'}"
                    json_emit("tool_fail", name=name, out=output)
                if is_available():
                    json_emit("tts", text=reply)
                    tts_speak(reply)
                json_emit("done", text=reply, dt=_time.time() - t0)
        except Exception as exc:
            json_emit("error", msg=f"command failed: {exc}")
        finally:
            command_lock.release()

    # Start reading only after handle_user_text exists. Starting the reader
    # earlier created a small startup race for commands typed immediately.
    stdin_thread = threading.Thread(target=stdin_reader, daemon=True)
    stdin_thread.start()

    import sounddevice as sd
    import numpy as np
    def audio_cb(indata, *_):
        audio_q.put(indata.copy())

    def voice_loop():
        nonlocal audio_buf, audio_buf_fill
        audio_buf = np.zeros(WIN, dtype=np.int16)
        audio_buf_fill = 0
        cooldown_until = 0.0
        with sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                            blocksize=CHUNK, callback=audio_cb):
            while not stop_flag[0]:
                try:
                    chunk = audio_q.get(timeout=0.5)
                except _q.Empty:
                    continue
                if _time.time() < cooldown_until:
                    continue
                chunk = chunk[:, 0].astype(np.int16)
                n = len(chunk)
                audio_buf = np.roll(audio_buf, -n)
                audio_buf[-n:] = chunk
                audio_buf_fill = min(WIN, audio_buf_fill + n)
                if audio_buf_fill < WIN:
                    continue
                scores = wake.predict(audio_buf.copy())
                score = float(scores[wake_name])
                json_emit("wake_score", score=score)
                if score < threshold:
                    continue
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
                threading.Thread(target=handle_user_text, args=(text,), daemon=True).start()
                audio_buf[:] = 0
                audio_buf_fill = 0

    def _capture(stt, audio_q, audio_buf, audio_buf_fill):
        import soundfile as sf
        from pathlib import Path as _P
        json_emit("stt_start")
        frames = [audio_buf.copy()]
        silent_for = 0.0
        spoken = False
        t0 = _time.time()
        while _time.time() - t0 < float(os.environ.get("COZY_CAPTURE_TIMEOUT", "10")):
            try:
                chunk = audio_q.get(timeout=0.05)
            except _q.Empty:
                chunk = None
            if chunk is not None:
                pcm = chunk[:, 0]
                frames.append(pcm.copy())
                level = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
                json_emit("capture_level", level=min(1.0, level / 2000.0))
                if level > 180:
                    spoken = True
                    silent_for = 0.0
                elif spoken:
                    silent_for += len(pcm) / 16000
            if spoken and silent_for >= 1.0:
                break
        pcm = np.concatenate(frames) if len(frames) > 1 else np.zeros(16000, np.int16)
        energy = float(np.abs(pcm).mean())
        if energy < 100:
            json_emit("rejected", reason=f"low energy ({energy:.0f})")
            return ""
        fd, tmp_name = tempfile.mkstemp(prefix="cozy_cmd_", suffix=".wav")
        os.close(fd)
        tmp = _P(tmp_name)
        sf.write(str(tmp), pcm, 16000, subtype="PCM_16")
        try:
            text = stt.transcribe_file(str(tmp))
        except Exception as exc:
            json_emit("error", msg=f"transcription failed: {exc}")
            return ""
        finally:
            try: tmp.unlink()
            except OSError: pass
        text = (text or "").strip()
        if len(text) < 3:
            return ""
        if not any(c.isalpha() for c in text):
            return ""
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
        a = set(last.lower().split())
        b = set(text.lower().split())
        return len(a & b) / min(len(a), len(b)) > 0.5 if (a and b) else False

    def voice_runner():
        try:
            voice_loop()
        except Exception as exc:
            json_emit("error", msg=f"audio loop stopped: {exc}")

    voice_thread = threading.Thread(target=voice_runner, daemon=True)
    voice_thread.start()

    # Main thread: idle until stdin / signal
    import signal
    signal.signal(signal.SIGTERM, lambda *a: stop_flag.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *a: (stop_flag.__setitem__(0, True),
                                             sys.exit(0)))
    while not stop_flag[0]:
        _time.sleep(0.2)

SR = 16000
CHUNK = 1280  # 80 ms - audio capture granularity
WIN_SAMPLES = SR * 2  # 2 seconds - livekit-wakeword inference window

# Auto-detect the wakeword venv site-packages and add to sys.path.
# This lets the assistant runtime find livekit-wakeword regardless of
# which Python is used to launch it. We check for the .venv at:
#   <Cozy>/.venv/   (project-level venv)
#   <wakeword>/.venv/  (wakeword venv)
# The wakeword venv has livekit-wakeword installed (the assistant venv
# is currently empty).
for venv_candidate in [WW.parent / ".venv", WW / ".venv"]:
    sp = venv_candidate / "lib"
    if sp.exists():
        for sub in sp.iterdir():
            if sub.is_dir() and sub.name.startswith("python"):
                candidate = sub / "site-packages"
                if candidate.exists() and str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
                    break
# Also add wakeword itself for in-tree imports
if str(WW) not in sys.path:
    sys.path.insert(0, str(WW))


# ------------------------------------------------------------------ loading
def load_wake(threshold):
    """Load the livekit-wakeword model for hey_cozy.

    The model is a small ONNX (122 KB) that takes a 2-second 16kHz int16
    audio window and returns a wake-word score in [0, 1]. Trained on 138
    user-voice positives + 500 synth positives + 2568 negatives.
    """
    from livekit.wakeword import WakeWordModel
    model_path = WW / "output" / "hey_cozy" / "hey_cozy.onnx"
    if not model_path.exists():
        raise SystemExit(
            f"wake model not found: {model_path}\n"
            f"train it first:\n"
            f"  cd {WW}\n"
            f"  uv run livekit-wakeword setup --config configs/hey_cozy_test.yaml --skip-acav\n"
            f"  uv run livekit-wakeword run configs/hey_cozy_test.yaml"
        )
    m = WakeWordModel(models=[model_path])
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = HERE / "model" / "cozy-llm-v1"
    if not path.exists():
        raise SystemExit("LLM not fine-tuned yet - run sft_qwen.py")
    tok = AutoTokenizer.from_pretrained(str(path))
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
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
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
        # note: enable_thinking=False passed below to keep voice replies fast
        # enable_thinking=False keeps Qwen3 fast for voice commands
        [{"role": "system", "content": system},
         {"role": "user", "content": user_text}],
        tools=schema,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=96, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()
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
                params = call.get("parameters") or call.get("arguments") or {}
                fixed_name = normalize_tool_name(call["name"])
                return None, {"name": fixed_name, "parameters": params}
        except json.JSONDecodeError:
            continue
    return text or "...", None


# ------------------------------------------------------------------- audio
def record_command(stt_model, max_seconds=7.0, silence_after=1.1):
    import sounddevice as sd
    import soundfile as sf

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

    with sd.InputStream(samplerate=16000, channels=1, dtype="int16",
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
    parser.add_argument("--text", action="store_true",
                        help="type commands instead of speaking")
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
    parser.add_argument("--fast-harness", action="store_true",
                        help="(internal) the default is already the fast harness")
    parser.add_argument("--harness-only", action="store_true",
                        help="test the harness without loading LLM/STT/wake (rules only)")
    args = parser.parse_args()
    _json_mode[0] = args.json_events
    if _json_mode[0]:
        # Disable stdout buffering so events are visible immediately
        import sys as _sys
        _sys.stdout.reconfigure(line_buffering=True)

    # Default (no flags) AND we have a TTY: launch the textual TUI with
    # voice listening. This is the one-liner `cozy` that the user wants.
    # --json-events: run the voice loop and emit NDJSON to stdout.
    # The Node Ink TUI consumes this. No textual app overhead.
    if args.json_events:
        from rlm_harness.harness_fast import FastHarness, HarnessConfig
        from executor import execute as executor_execute
        cfg = HarnessConfig()
        cfg.use_wake = cfg.use_stt = cfg.use_llm = cfg.use_tts = True
        h = FastHarness(cfg)
        # Run the voice loop inline, no textual TUI
        run_json_mode(h, executor_execute, args.threshold or 0.5)
        return

    is_default_mode = (
        not args.text and
        not args.calibrate and
        not args.harness_only and
        not args.no_wake and
        not args.json_events and
        sys.stdout.isatty()
    )
    if is_default_mode:
        from rlm_harness.harness_fast import FastHarness, HarnessConfig
        from tui_textual import run_textual
        from executor import execute as executor_execute
        cfg = HarnessConfig()
        cfg.use_wake = True
        cfg.use_stt = True
        cfg.use_tts = not args.no_tts
        cfg.use_llm = True
        h = FastHarness(cfg)
        # Threshold: use the eval-optimal 0.5, not the over-sensitive 0.30
        # that caused the false-positive wake fires in the earlier build.
        threshold = 0.5
        if not args.no_tts and hasattr(args, "no_tts"):
            pass  # threshold tunable in voice.cfg in a future revision
        run_textual(h, executor_execute, voice_mode=True, threshold=threshold)
        return

    # If --no-tts, swap the speak callback for a no-op so we never load Kokoro
    # or touch the audio device.
    if args.no_tts:
        global _noop_speak
        def _noop_speak(text):
            print(f"(no-tts) {text}")
        # patch the helper to swap speak globally below


    # Default threshold comes from the trained model's eval JSON
    # (AUT/FPPH/recall-optimal threshold computed during livekit training)
    metrics_file = WW / "output" / "hey_cozy" / "hey_cozy_eval.json"
    threshold = args.threshold
    if threshold is None:
        threshold = 0.5  # the eval-optimal threshold. Lower values cause
        # many false-positive wake fires from background noise, which
        # makes the assistant speak when the user didn't say "hey cozy".
        if metrics_file.exists():
            try:
                metrics = json.loads(metrics_file.read_text())
                # prefer the eval-optimal threshold (best FPPH/recall tradeoff)
                if metrics.get("optimal_threshold"):
                    threshold = float(metrics["optimal_threshold"])
                elif metrics.get("threshold"):
                    threshold = float(metrics["threshold"])
            except Exception:
                pass
    print("[config] threshold =", threshold)

    def _get_parent_harness():
        return fast_harness

    if args.harness_only:
        # Smoke-test the harness without loading the LLM, STT, or wake.
        # The TUI handles /commands and falls back to the rule router.
        from rlm_harness.harness_fast import FastHarness, HarnessConfig
        from executor import execute as executor_execute
        cfg = HarnessConfig()
        cfg.use_llm = cfg.use_stt = cfg.use_tts = cfg.use_wake = cfg.use_vision = False
        h = FastHarness(cfg)
        if sys.stdin.isatty() and sys.stdout.isatty():
            from tui_textual import run_textual
            run_textual(h, executor_execute, voice_mode=False)
        else:
            from rlm_harness.tui import TUI
            TUI(h, executor_execute, None).run_forever()
        return

    # --text: textual TUI with input prompt (no voice).
    # The TUI needs a TTY; if stdin is not a TTY, fall through to the
    # legacy input() loop (which exits cleanly on EOF).
    if args.json_events:
        # The Node Ink TUI consumes NDJSON on stdout. It provides its
        # own raw-mode terminal handling, so the runtime doesn't need
        # a TTY to operate.
        from rlm_harness.harness_fast import FastHarness, HarnessConfig
        from executor import execute as executor_execute
        cfg = HarnessConfig()
        cfg.use_llm = True
        h = FastHarness(cfg)
        run_json_mode(h, executor_execute, args.threshold or 0.5)
        return
    if sys.stdin.isatty() and sys.stdout.isatty():
        from rlm_harness.harness_fast import FastHarness, HarnessConfig
        from tui_textual import run_textual
        from executor import execute as executor_execute
        if args.fast_harness:
            h = FastHarness()
        else:
            cfg = HarnessConfig()
            cfg.use_llm = True
            h = FastHarness(cfg)
        # Don't pre-load the LLM - the TUI runs warmup() in a background
        # thread and shows a "loading" state. This way the TUI mounts
        # instantly (no 20s block) and the user can see progress.
        run_textual(h, executor_execute, voice_mode=False)
        return
    # Non-TTY fallback: simple input() loop (good for piped tests)
    if not sys.stdin.isatty():
        print("cozy --text needs a TTY; use `cozy --harness-only` for non-interactive tests",
              file=sys.stderr)
        return
    tok, llm = (None, None)
    fast_harness = None
    if args.fast_harness:
        from rlm_harness.harness_fast import FastHarness
        fast_harness = FastHarness()
    else:
        tok, llm = load_llm(use_dpo=not args.sft_only)
    from rlm_harness.tui import TUI
    from executor import execute as executor_execute
    tui = TUI(fast_harness if fast_harness else None, executor_execute, None)
    tui.run_forever()
    return

    # voice modes need audio deps
    import sounddevice as sd  # noqa: F401
    stt = load_stt() if not args.calibrate else None
    wake_tuple = load_wake(threshold) if not args.no_wake else None
    tok, llm = (None, None)
    fast_harness = None
    if args.fast_harness:
        # The fast harness loads the LLM lazily on first decide().
        from rlm_harness.harness_fast import FastHarness
        fast_harness = FastHarness()
    else:
        tok, llm = load_llm(use_dpo=not args.sft_only)

    if args.calibrate:
        m, name, thr = wake_tuple
        print("Calibrating 30s - say 'hey cozy' and other stuff...")
        q2 = queue.Queue()

        def cb2(indata, _f, _t, _s):
            q2.put(indata.copy())

        audio_buf2 = np.zeros(WIN_SAMPLES, dtype=np.int16)
        with sd.InputStream(samplerate=16000, channels=1, dtype="int16",
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

    cooldown_until = 0.0
    print("[runtime] READY - say 'hey cozy' then your command.")
    q = queue.Queue()

    def audio_cb(indata, _f, _t, _s):
        q.put(indata.copy())

    # Rolling 2s buffer for livekit-wakeword (model needs >=2s per inference)
    audio_buf = np.zeros(WIN_SAMPLES, dtype=np.int16)
    audio_buf_fill = 0

    with sd.InputStream(samplerate=16000, channels=1, dtype="int16",
                        blocksize=CHUNK, callback=audio_cb):
        while True:
            try:
                chunk = q.get(timeout=1.0)
            except queue.Empty:
                continue
            chunk = chunk[:, 0].astype(np.int16)

            # Update the rolling 2s buffer
            n = len(chunk)
            audio_buf = np.roll(audio_buf, -n)
            audio_buf[-n:] = chunk
            audio_buf_fill = min(WIN_SAMPLES, audio_buf_fill + n)
            if audio_buf_fill < WIN_SAMPLES:
                continue  # not enough audio yet

            if wake_tuple is not None:
                m, name, thr = wake_tuple
                # livekit-wakeword's predict takes a 2s window; it returns
                # {model_name: score} for that window
                scores = m.predict(audio_buf.copy())
                score = float(scores[name])
                if score < thr or time.time() < cooldown_until:
                    continue
                cooldown_until = time.time() + 4.0
                print(f"\a[wake] {name}! (score {score:.3f}, thr {thr})")

            text, wav = record_command(stt)
            print("[heard]", text or "(silence)")
            if text:
                speak = _noop_speak if args.no_tts else (lambda t: __import__("tts").speak(t))
            handle_text(text, tok, llm, lambda o: speak(o),
                          fast_harness=fast_harness)
            # Reset the rolling buffer so we don't re-process the wake audio
            audio_buf[:] = 0
            audio_buf_fill = 0


if __name__ == "__main__":
    main()
