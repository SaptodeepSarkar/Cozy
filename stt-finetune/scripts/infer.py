#!/usr/bin/env python
"""Quick transcription CLI for your finetuned Cozy STT model.

    .venv/bin/python scripts/infer.py --wav somefile.wav
    .venv/bin/python scripts/infer.py --mic 5        # record 5 s and transcribe
Prefers the CTranslate2 export; falls back to the merged HF checkpoint.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUTPUT_DIR  # noqa: E402

CT2 = OUTPUT_DIR / "cozy_stt_v1_ct2_int8"
HF = OUTPUT_DIR / "hf_finetuned"


def transcribe_mic(seconds: int) -> Path:
    out = Path(tempfile.gettempdir()) / "cozy_mic.wav"
    subprocess.run(["arecord", "-q", "-d", str(seconds), "-f", "S16_LE",
                    "-r", "16000", "-c", "1", str(out)], check=True)
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
