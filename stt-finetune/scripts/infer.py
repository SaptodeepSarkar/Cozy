#!/usr/bin/env python
"""Quick transcription CLI for your finetuned Cozy STT model.

    .venv/bin/python scripts/infer.py --wav somefile.wav
    .venv/bin/python scripts/infer.py --mic 5        # record 5 s and transcribe
Prefers the CTranslate2 export; falls back to the merged HF checkpoint.
"""
import argparse
import subprocess
import wave
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUTPUT_DIR  # noqa: E402

CT2 = OUTPUT_DIR / "cozy_stt_v1_ct2_int8"
HF = OUTPUT_DIR / "hf_finetuned"


def transcribe_mic(seconds: int) -> Path:
    import array
    import math
    import select
    import time

    out = Path(tempfile.gettempdir()) / "cozy_mic.wav"
    sr = 16000
    print("Speak after the countdown — recording stops ~1s after you finish "
          f"(max {seconds}s)")
    for c in range(3, 0, -1):
        print(f"  {c}...", flush=True)
        time.sleep(1)

    proc = subprocess.Popen(
        ["arecord", "-q", "-f", "S16_LE", "-r", str(sr), "-c", "1", "-t", "raw"],
        stdout=subprocess.PIPE)
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
                print(f"\r  [rec {elapsed:0.1f}s] ENTER=stop ", end="", flush=True)
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
    dur = len(all_s) / sr
    peak = max((abs(s) for s in all_s), default=0)
    rms = math.sqrt(sum(s * s for s in all_s[::4]) / max(1, len(all_s) // 4))
    if peak >= 32000:
        print(f"  !! input CLIPPED (peak {peak}) — lower mic gain (alsamixer)")
    elif rms < 300 and dur > 0.5:
        print(f"  !! very quiet (rms {rms:.0f}) — speak up / move closer")
    print(f"  captured {dur:.1f}s -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--mic", type=int, default=0, help="record N seconds from mic")
    ap.add_argument("--beam", type=int, default=1)
    args = ap.parse_args()

    wav = args.wav or (transcribe_mic(args.mic) if args.mic else None)
    if not wav:
        sys.exit("Give --wav PATH or --mic SECONDS")
    wav = str(wav)

def transcribe_ct2(wav, beam):
    from faster_whisper import WhisperModel
    import torch
    assert torch.cuda.is_available(), "dGPU required: CUDA device not found"
    model = WhisperModel(str(CT2), device="cuda", device_index=0,
                         compute_type="int8_float16")
    segments, _ = model.transcribe(wav, language="en", beam_size=beam,
                                   vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def transcribe_hf(wav):
    import librosa
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(str(HF))
    model = WhisperForConditionalGeneration.from_pretrained(
        str(HF), torch_dtype=torch.float16).to("cuda")
    audio, _ = librosa.load(wav, sr=16000, mono=True)
    feats = proc(audio, sampling_rate=16000, return_tensors="pt").to("cuda", torch.float16)
    ids = model.generate(feats.input_features, language="english",
                         task="transcribe", max_new_tokens=224)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--mic", type=int, default=0, help="record N seconds from mic")
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--engine", choices=["auto", "ct2", "hf"], default="auto")
    args = ap.parse_args()

    wav = args.wav or (transcribe_mic(args.mic) if args.mic else None)
    if not wav:
        sys.exit("Give --wav PATH or --mic SECONDS")
    wav = str(wav)

    text, engine = "", None
    if args.engine in ("auto", "ct2") and CT2.exists():
        text, engine = transcribe_ct2(wav, args.beam), f"ct2:{CT2.name}"
        if not text:   # known CT2 edge case with some merged weights -> fallback
            print("[ct2 empty, falling back to hf]", file=sys.stderr)
            text, engine = None, None
    if not text and HF.exists():
        text, engine = transcribe_hf(wav), "hf_finetuned"
    if not text:
        sys.exit(f"No working finetuned model. Looked in {CT2} and {HF}.")

    print(f"[model: {engine}]")
    print(text)


if __name__ == "__main__":
    main()
