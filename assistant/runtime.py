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
CHUNK = 1280


# ------------------------------------------------------------------ loading
def load_wake(threshold):
    sys.path.insert(0, str(WW))
    from openwakeword.model import Model

    model_path = WW / "models" / "cozy_v1.onnx"
    m = Model(wakeword_models=[str(model_path)],
              inference_framework="onnx")
    name = next(iter(m.models.keys()))
    print("[wake] loaded", name, "threshold", threshold)
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
        [{"role": "system", "content": system},
         {"role": "user", "content": user_text}],
        tools=schema,
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=96, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()

    match = re.search(r"{[^{}]*}", text)
    if match:
        try:
            call = json.loads(match.group(0))
            if isinstance(call.get("name"), str):
                return None, call
        except json.JSONDecodeError:
            pass
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

    metrics_file = WW / "models" / "metrics.json"
    threshold = args.threshold
    if threshold is None:
        threshold = 0.5
        if metrics_file.exists():
            try:
                metrics = json.loads(metrics_file.read_text())
                threshold = float(metrics.get("safe_threshold_zero_fpr",
                                              0.5))
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
        with sd.InputStream(samplerate=16000, channels=1, dtype="int16",
                            blocksize=CHUNK):
            end = time.time() + 30
            while time.time() < end:
                time.sleep(0.25)
                buf = list(m.prediction_buffer[name])
                recent = buf[-5:]
                bar = "#" * int(max(recent or [0]) * 40)
                print("\r" + bar.ljust(42)
                      + format(max(recent or [0]), ".3f"),
                      end="", flush=True)
        print()
        return

    cooldown_until = 0.0
    print("[runtime] READY - say 'hey cozy' then your command.")
    q = queue.Queue()

    def audio_cb(indata, _f, _t, _s):
        q.put(indata.copy())

    with sd.InputStream(samplerate=16000, channels=1, dtype="int16",
                        blocksize=CHUNK, callback=audio_cb):
        while True:
            try:
                chunk = q.get(timeout=1.0)
            except queue.Empty:
                continue
            chunk = chunk[:, 0]

            if wake_tuple is not None:
                m, name, thr = wake_tuple
                score = float(m.predict(chunk)[name])
                if score < thr or time.time() < cooldown_until:
                    continue
                cooldown_until = time.time() + 4.0
                print("\a[wake] cozy! (score", format(score, ".2f") + ")")

            text, wav = record_command(stt)
            print("[heard]", text or "(silence)")
            if text:
                handle_text(text, tok, llm,
                            lambda o: None)


if __name__ == "__main__":
    main()
