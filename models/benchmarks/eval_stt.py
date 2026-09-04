#!/usr/bin/env python3
"""Evaluate v1.0 (openai/whisper-small base) and v1.1 (Cozy LoRA) on the
held-out eval set and write JSON results for the README benchmark plot.

Run from repo root:
    python models/benchmarks/eval_stt.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL_MANIFEST = REPO / "stt-finetune" / "data" / "manifests" / "eval.jsonl"
OUT_DIR = REPO / "models" / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The eval manifest audio_path is relative to stt-finetune/. chdir there
import os
os.chdir(REPO / "stt-finetune")


def evaluate(name: str, model_path: str | Path, use_ct2: bool = True, limit: int = 0) -> dict:
    import soundfile as sf
    import librosa

    if use_ct2:
        from faster_whisper import WhisperModel

        def transcribe(audio):
            segs, _ = model.transcribe(audio, language="en", beam_size=1,
                                        vad_filter=False)
            return " ".join(s.text.strip() for s in segs).strip()

        print(f"  loading (faster-whisper CT2) {model_path} ...")
        model = WhisperModel(str(model_path), device="cuda",
                             device_index=0, compute_type="int8_float16")
        engine = "faster-whisper CT2"
    else:
        import torch
        from transformers import (WhisperForConditionalGeneration,
                                  WhisperProcessor)

        def transcribe(audio):
            inputs = processor(audio, sampling_rate=16000,
                                return_tensors="pt").input_features
            inputs = inputs.to("cuda", dtype=torch.float16)
            with torch.no_grad():
                ids = m.generate(inputs, language="english", task="transcribe",
                                 max_new_tokens=224)
            return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

        print(f"  loading (transformers fp16) {model_path} ...")
        processor = WhisperProcessor.from_pretrained(str(model_path),
                                                     language="english",
                                                     task="transcribe")
        m = WhisperForConditionalGeneration.from_pretrained(
            str(model_path), torch_dtype=torch.float16).to("cuda").eval()
        m.config.forced_decoder_ids = None
        engine = "transformers fp16"
    from jiwer import wer as compute_wer
    rows = [json.loads(l) for l in open(EVAL_MANIFEST, encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]
    refs, hyps, secs, audio_secs = [], [], [], 0.0
    for r in rows:
        audio, sr = sf.read(r["audio_path"], dtype="float32")
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        audio_secs += len(audio) / sr
        t0 = time.time()
        hyp = transcribe(audio).lower()
        secs.append(time.time() - t0)
        hyps.append(hyp)
        raw = r["text"]
        refs.append(raw.split("\t")[-1].strip().lower())

    out = {
        "model": name,
        "wer": compute_wer(refs, hyps),
        "rtf": sum(secs) / max(audio_secs, 1e-6),
        "audio_secs": audio_secs,
        "n_clips": len(rows),
        "engine": engine,
    }
    print(f"  WER = {out['wer'] * 100:.2f}%   RTF = {out['rtf']:.3f}")
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="openai/whisper-small")
    parser.add_argument("--model", default=str(REPO / "stt-finetune" / "output" / "hf_finetuned"))
    parser.add_argument("--ct2", action="store_true", help="evaluate CT2 model instead of HF transformers")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    v1 = evaluate("v1.0 (openai/whisper-small base)",
                  args.base, use_ct2=False, limit=args.limit)
    v2 = evaluate("v1.1 (Cozy LoRA, Hinglish-aware)",
                  args.model, use_ct2=args.ct2, limit=args.limit)
    summary = {"v1.0": v1, "v1.1": v2}
    (OUT_DIR / "stt_eval.json").write_text(json.dumps(summary, indent=2))
    print("wrote", OUT_DIR / "stt_eval.json")


if __name__ == "__main__":
    main()
