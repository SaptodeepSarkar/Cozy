#!/usr/bin/env python3
"""Evaluate a Whisper HF checkpoint on data/manifests/eval.jsonl with
per-source WER breakdown (cv_indian / santhosh_indian / flux_tts).

Writes models/benchmarks/stt_eval_v1.2.json:
  {model, wer, rtf, audio_secs, n_clips, engine, by_source: {...}}

Usage:
  .venv/bin/python scripts/eval_model.py --hf output/hf_finetuned_v1.2 --tag v1.2
  .venv/bin/python scripts/eval_model.py --base openai/whisper-small --tag v1.0   # stock baseline
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_MODEL, MANIFEST_DIR, english_normalizer, read_manifest, wer  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "models" / "benchmarks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default=None, help="HF checkpoint dir (default: stock base)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    name = args.hf or f"base:{BASE_MODEL}"
    print(f"== Evaluating [{args.tag}] {name} ==")
    processor = WhisperProcessor.from_pretrained(
        args.hf or BASE_MODEL, language="english", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.hf or BASE_MODEL, torch_dtype=torch.float16).to("cuda").eval()
    model.config.forced_decoder_ids = None

    import librosa
    rows = list(read_manifest(MANIFEST_DIR / "eval.jsonl"))
    if args.limit:
        rows = rows[:args.limit]
    print(f"eval clips: {len(rows)}")

    norm = english_normalizer()
    refs, hyps, srcs, gen_total, audio_secs = [], [], [], 0.0, 0.0
    with torch.inference_mode():
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            feats = []
            for r in chunk:
                audio, _ = librosa.load(r["audio_path"], sr=16000, mono=True)
                audio_secs += len(audio) / 16000
                feats.append(processor(audio, sampling_rate=16000,
                                       return_tensors="pt").input_features[0])
            batch = torch.stack(feats).half().to("cuda")
            t0 = time.time()
            ids = model.generate(batch, language="english", task="transcribe",
                                 max_new_tokens=224, do_sample=False, use_cache=True)
            gen_total += time.time() - t0
            texts = processor.batch_decode(ids, skip_special_tokens=True)
            for r, h in zip(chunk, texts):
                refs.append(norm(r["text"]))
                hyps.append(norm(h))
                srcs.append(r["source"])
            if (i + len(chunk)) % 40 < args.batch:
                print(f"  {i+len(chunk)}/{len(rows)}", flush=True)

    total_s = gen_total
    out = {
        "model": args.tag,
        "wer": wer(hyps, refs),
        "rtf": gen_total / max(audio_secs, 1e-6),
        "audio_secs": round(audio_secs, 1),
        "n_clips": len(rows),
        "engine": "transformers fp16",
        "by_source": {},
    }
    for s in sorted(set(srcs)):
        sr_ = [r for r, ss in zip(refs, srcs) if ss == s]
        sh = [h for h, ss in zip(hyps, srcs) if ss == s]
        out["by_source"][s] = {"wer": wer(sh, sr_), "n": len(sr_)}
    print(json.dumps(out, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"stt_eval_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT / f'stt_eval_{args.tag}.json'}")


if __name__ == "__main__":
    main()
