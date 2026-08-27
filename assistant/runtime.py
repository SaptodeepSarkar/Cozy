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
import queue
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WW = HERE.parent / "wakeword"
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


def load_llm():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = HERE / "model" / "cozy-llm-v1"
    if not path.exists():
        raise SystemExit("LLM not fine-tuned yet - run sft_qwen.py")
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_pretrained(
        str(path), dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()
    print("[llm] loaded", path.name)
    return tok, model


import torch  # noqa: E402


# ------------------------------------------------------------------- llm io
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
    candidates += re.findall(r"\{[^{}]*\}", text)
    for c in candidates:
        try:
            call = json.loads(c)
            if isinstance(call.get("name"), str):
                params = call.get("parameters") or call.get("arguments") or {}
                return None, {"name": call["name"], "parameters": params}
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
def handle_text(text, tok, llm, speak):
    if not text:
        return
    print("[you]", text)
    reply, call = llm_decide(tok, llm, text)
    if call is not None:
        from executor import execute

        result = execute(call["name"], call.get("parameters") or {})
        print("[cozy]", ("Done." if result["ok"] else "Failed:")
              , result["output"])
        speak(result["output"])
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
    args = parser.parse_args()

    # Default threshold comes from the trained model's eval JSON
    # (AUT/FPPH/recall-optimal threshold computed during livekit training)
    metrics_file = WW / "output" / "hey_cozy" / "hey_cozy_eval.json"
    threshold = args.threshold
    if threshold is None:
        threshold = 0.30  # tuned default for the user-voice-trained model
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

    if args.text:
        tok, llm = load_llm()
        print("Type commands ('exit' to quit):")
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in {"exit", "quit"}:
                break
            handle_text(text, tok, llm, lambda o: print("(speak)", o))
        return

    # voice modes need audio deps
    import sounddevice as sd  # noqa: F401
    stt = load_stt() if not args.calibrate else None
    wake_tuple = load_wake(threshold) if not args.no_wake else None
    tok, llm = load_llm()

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
                handle_text(text, tok, llm,
                            lambda o: None)
            # Reset the rolling buffer so we don't re-process the wake audio
            audio_buf[:] = 0
            audio_buf_fill = 0


if __name__ == "__main__":
    main()
