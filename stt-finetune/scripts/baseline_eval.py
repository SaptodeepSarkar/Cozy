#!/usr/bin/env python
"""Baseline / post-train WER evaluation on data/manifests/eval.jsonl.

Usage:
    .venv/bin/python scripts/baseline_eval.py                    # vanilla base model
    .venv/bin/python scripts/baseline_eval.py --adapter checkpoints/lora_final
    .venv/bin/python scripts/baseline_eval.py --hf output/hf_finetuned
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MANIFEST_DIR, english_normalizer, read_manifest, wer  # noqa: E402


def load_rows():
    rows = list(read_manifest(MANIFEST_DIR / "eval.jsonl"))
    assert rows, "No eval manifest. Run prepare_data.py first."
    return rows


def load_audio(row):
    import librosa
    wav, _ = librosa.load(row["audio_path"], sr=16000, mono=True)
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter")
    ap.add_argument("--hf")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    name = args.tag or (args.hf or args.adapter or f"base:{BASE_MODEL}")
    print(f"== Evaluating [{name}] ==")
    processor = WhisperProcessor.from_pretrained(
        args.hf or BASE_MODEL, language="english", task="transcribe")

    if args.hf:
        model = WhisperForConditionalGeneration.from_pretrained(
            args.hf, torch_dtype=torch.float16).to("cuda")
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16).to("cuda")
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    norm = english_normalizer()
    rows = load_rows()
    if args.limit:
        rows = rows[: args.limit]

    refs, hyps, srcs = [], [], []
    B = 4
    with torch.inference_mode():
        for i in range(0, len(rows), B):
            chunk = rows[i:i + B]
            feats = []
            for r in chunk:
                audio = load_audio(r)
                feats.append(processor(audio, sampling_rate=16000,
                                       return_tensors="pt").input_features[0])
            batch = torch.stack(feats).half().to("cuda")
            ids = model.generate(batch, language="english", task="transcribe",
                                 max_new_tokens=200, do_sample=False,
                                 use_cache=True)
            texts = processor.batch_decode(ids, skip_special_tokens=True)
            for r, h in zip(chunk, texts):
                refs.append(norm(r["text"]))
                hyps.append(norm(h))
                srcs.append(r["source"].split("_r")[0])
            done = min(i + B, len(rows))
            if done % 40 < B:
                print(f"   {done}/{len(rows)}")

    overall = wer(hyps, refs)
    print(f"\n=== WER [{name}] overall: {overall:.2f}%  ({len(rows)} clips)")
    for s in sorted(set(srcs)):
        sub_r = [r for r, ss in zip(refs, srcs) if ss == s]
        sub_h = [h for h, ss in zip(hyps, srcs) if ss == s]
        print(f"    {s:16s}: {wer(sub_h, sub_r):6.2f}%  ({len(sub_r)} clips)")
    out = Path("eval_results.txt")
    with open(out, "a") as f:
        f.write(f"{name}\toverall={overall:.2f}\n")
    print(f"(appended to {out})")


if __name__ == "__main__":
    main()
