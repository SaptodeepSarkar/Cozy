#!/usr/bin/env python
"""Cozy STT CLI — finetuned whisper-small, fully local.

Modes:
    .venv/bin/python scripts/infer.py                # LIVE STREAM (default):
                                                     # Silero VAD segments your
                                                     # speech; each finished
                                                     # sentence -> final text.
    .venv/bin/python scripts/infer.py --mic 8        # one-shot take
    .venv/bin/python scripts/infer.py --wav f.wav    # transcribe a file

Engines: auto (default) = fast CTranslate2 first, HF transformers fallback
(Hinglish-aware v4). Force with --engine ct2 | hf.
Model loads ONCE and stays resident in stream mode.
"""
import argparse
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUTPUT_DIR  # noqa: E402

CT2 = OUTPUT_DIR / "cozy_stt_v1_ct2_int8"
HF = OUTPUT_DIR / "hf_finetuned"


# ------------------------------------------------------------------ engines
class Engines:
    """Lazy, load-once engine holders (array-based cores)."""

    def __init__(self, beam=1):
        self.beam = beam
        self._ct2 = None
        self._hf = None

    def ct2(self):
        if self._ct2 is None:
            from faster_whisper import WhisperModel
            import torch
            assert torch.cuda.is_available(), "dGPU required: no CUDA device"
            t0 = time.time()
            self._ct2 = WhisperModel(str(CT2), device="cuda", device_index=0,
                                     compute_type="int8_float16")
            print(f"[engine] CTranslate2 loaded in {time.time()-t0:.1f}s")
        return self._ct2

    def hf(self):
        if self._hf is None:
            import numpy as np
            import torch
            from transformers import (WhisperForConditionalGeneration,
                                      WhisperProcessor)
            t0 = time.time()
            self._hf_proc = WhisperProcessor.from_pretrained(str(HF))
            self._hf = WhisperForConditionalGeneration.from_pretrained(
                str(HF), torch_dtype=torch.float16).to("cuda").eval()
            self._np = np
            self._torch = torch
            print(f"[engine] HF transformers loaded in {time.time()-t0:.1f}s")
        return self._hf

    def warmup(self, engine="auto"):
        """Preload every relevant engine BEFORE the user starts speaking.
        Tolerates per-engine failures (e.g. CUDA OOM while another process
        trains) - that engine simply stays unloaded until first use."""
        import numpy as np
        z = np.zeros(8000, dtype=np.float32)   # 0.5 s of silence
        if engine in ("auto", "ct2") and CT2.exists():
            try:
                m = self.ct2()
                next(iter(m.transcribe(z, language="en", beam_size=1)), None)
            except Exception as e:
                print(f"[engine] ct2 unavailable ({e}); will rely on hf",
                      file=sys.stderr)
                self._ct2 = None
        if engine in ("auto", "hf") and HF.exists():
            model = self.hf()
            feats = self._hf_proc(z, sampling_rate=16000,
                                  return_tensors="pt").input_features
            feats = feats.to("cuda", self._torch.float16)
            with self._torch.inference_mode():
                model.generate(feats, language="english", task="transcribe",
                               max_new_tokens=1)

    def transcribe_array(self, audio_f32, engine="auto"):
        """audio: float32 16 kHz mono. Returns (text, engine_used)."""
        import torch
        audio_f32 = audio_f32.astype("float32", copy=False)
        if engine in ("auto", "ct2") and CT2.exists():
            try:
                segs, _ = self.ct2().transcribe(
                    audio_f32, language="en", beam_size=self.beam,
                    initial_prompt="Cozy assistant. Romanized Hindi words: "
                                   "aaj kaisa karo yaar accha theek thoda "
                                   "nahi bas arre.")
                text = " ".join(s.text.strip() for s in segs).strip()
                if text:
                    return text, "ct2"
            except Exception as e:
                print(f"[ct2 error: {e}]", file=sys.stderr)
        if HF.exists():
            model = self.hf()
            feats = self._hf_proc(audio_f32, sampling_rate=16000,
                                  return_tensors="pt").input_features
            feats = feats.to("cuda", torch.float16)
            with torch.inference_mode():
                ids = model.generate(feats, language="english",
                                     task="transcribe", max_new_tokens=224)
            text = self._hf_proc.batch_decode(
                ids, skip_special_tokens=True)[0].strip()
            return text, "hf"
        raise RuntimeError("no usable STT model found")


def transcribe_file(path, engines, engine="auto"):
    import librosa
    audio, _ = librosa.load(str(path), sr=16000, mono=True)
    return engines.transcribe_array(audio, engine)


# ------------------------------------------------------------- live stream
def stream_mode(engines, engine="auto", max_len=60.0):
    import queue

    import numpy as np
    import sounddevice as sd
    from silero_vad import VADIterator, load_silero_vad

    print("[stream] loading Silero VAD ...")
    vad = VADIterator(load_silero_vad(), sampling_rate=16000,
                      threshold=0.45, min_silence_duration_ms=650,
                      speech_pad_ms=120)

    q = queue.Queue()

    def cb(indata, frames, _t, _status):
        q.put(bytes(indata))

    BLOCK = 512  # 32 ms — what Silero expects @16 kHz
    preroll_max = 12  # blocks (~0.38 s) kept before speech starts
    preroll = []
    speech_buf = []
    speaking = False
    t_start = 0.0

    print("[stream] warming up engines (loads once) ...")
    engines.warmup(engine)
    print(f"[stream] ready ({engine}) — speak naturally; final text prints "
          "after each sentence. Ctrl+C to quit.")

    with sd.RawInputStream(samplerate=16000, blocksize=BLOCK, dtype="int16",
                           channels=1, callback=cb):
        try:
            while True:
                pcm = np.frombuffer(q.get(), dtype=np.int16)
                chunk = pcm.astype(np.float32) / 32768.0
                event = vad(chunk)

                if not speaking:
                    preroll.append(chunk)
                    if len(preroll) > preroll_max:
                        preroll.pop(0)
                    if event and "start" in event:
                        speaking = True
                        t_start = time.time()
                        speech_buf = list(preroll)
                        preroll = []
                        print("\r● ", end="", flush=True)
                else:
                    speech_buf.append(chunk)
                    dur = time.time() - t_start
                    print(f"\r● {dur:0.1f}s ", end="", flush=True)
                    if (event and "end" in event) or dur > max_len:
                        audio = np.concatenate(speech_buf) if speech_buf \
                            else np.zeros(1, dtype=np.float32)
                        speech_buf = []
                        speaking = False
                        preroll = []
                        print("\r" + " " * 20 + "\r", end="", flush=True)
                        if len(audio) < 4800:      # < 0.3 s — ignore blips
                            continue
                        t0 = time.time()
                        text, used = engines.transcribe_array(audio, engine)
                        dt = time.time() - t0
                        print(f"[{used} {dt*1000:.0f}ms] {text}")
        except KeyboardInterrupt:
            print("\n[stream] bye")


# ------------------------------------------------------------------ one-shot
def record_take(seconds: int) -> Path:
    import array
    import math
    import select

    out = Path(tempfile.gettempdir()) / "cozy_mic.wav"
    sr = 16000
    print(f"Speak after countdown — stops ~1s after you finish (max {seconds}s)")
    for c in range(3, 0, -1):
        print(f"  {c}...", flush=True)
        time.sleep(1)
    proc = subprocess.Popen(["arecord", "-q", "-f", "S16_LE", "-r", str(sr),
                             "-c", "1", "-t", "raw"], stdout=subprocess.PIPE)
    frames, silent_tail, elapsed = [], 0.0, 0.0
    try:
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r and sys.stdin.read(1) == "\n":
                break
            raw = proc.stdout.read(sr // 10 * 2)
            if not raw:
                break
            samples = array.array("h")
            samples.frombytes(raw)
            if len(samples):
                frames.append(samples.tobytes())
                rms = math.sqrt(sum(s * s for s in samples[::4]) /
                                max(1, len(samples) // 4))
                silent_tail = silent_tail + 0.1 if rms < 500 else 0.0
                elapsed += 0.1
                print(f"\r  [rec {elapsed:0.1f}s] ENTER=stop ",
                      end="", flush=True)
                if elapsed >= 1.0 and silent_tail >= 0.9:
                    break
                if elapsed >= seconds:
                    break
    finally:
        print()
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    buf = b"".join(frames)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(buf)
    all_s = array.array("h")
    all_s.frombytes(buf)
    peak = max((abs(s) for s in all_s), default=0)
    if peak >= 32000:
        print("  !! input CLIPPED — lower mic gain (alsamixer)")
    print(f"  captured {len(all_s)/sr:.1f}s -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", help="transcribe an audio file")
    ap.add_argument("--mic", type=int, default=0,
                    help="one-shot take of N seconds")
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--max-len", type=float, default=60.0,
                    help="force-flush an utterance after this many seconds")
    ap.add_argument("--engine", choices=["auto", "ct2", "hf"], default="auto",
                    help="'hf' recommended for Hinglish")
    args = ap.parse_args()
    engines = Engines(beam=args.beam)

    if not args.wav and not args.mic:
        stream_mode(engines, args.engine, args.max_len)  # default: live stream
        return

    wav = args.wav or str(record_take(args.mic))
    text, used = transcribe_file(wav, engines, args.engine)
    print(f"[model: {used}]")
    print(text)


if __name__ == "__main__":
    main()
