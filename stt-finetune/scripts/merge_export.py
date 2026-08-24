#!/usr/bin/env python
"""Merge LoRA adapters into the base model, save a plain-HF checkpoint, and
convert it to CTranslate2 int8_float16 for faster-whisper (your roadmap's STT).

Run:  .venv/bin/python scripts/merge_export.py [--adapter checkpoints/lora_cozy_v1/adapter]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_MODEL, CHECKPOINT_DIR, OUTPUT_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(CHECKPOINT_DIR / "lora_cozy_v1" / "adapter"))
    ap.add_argument("--hf-out", default=str(OUTPUT_DIR / "hf_finetuned"))
    ap.add_argument("--ct2-out", default=str(OUTPUT_DIR / "cozy_stt_v1_ct2_int8"))
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print(f"== Merging {args.adapter} into {BASE_MODEL} ...")
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32)  # merge in fp32 for accuracy
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    Path(args.hf_out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.hf_out, safe_serialization=True)
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    processor.save_pretrained(args.hf_out)
    print(f"   merged HF model -> {args.hf_out}")

    print("== Converting to CTranslate2 (int8_float16) ...")
    from ctranslate2.converters import TransformersConverter
    converter = TransformersConverter(args.hf_out, load_as_float16=True)
    converter.convert(args.ct2_out, quantization="int8_float16")
    print(f"   CT2 model -> {args.ct2_out}")

    # sanity check with faster-whisper on any available clip
    import torch
    assert torch.cuda.is_available(), "dGPU required: CUDA device not found"
    from faster_whisper import WhisperModel
    m = WhisperModel(args.ct2_out, device="cuda", device_index=0,
                     compute_type="int8_float16")
    probe = next(Path("recordings").rglob("*.wav"), None)
    if probe:
        segs, _ = m.transcribe(str(probe), language="en", beam_size=1)
        text = " ".join(s.text for s in segs).strip()
        print(f"== Sanity transcription of {probe.name}:")
        print(f"   \"{text}\"")
    else:
        print("== No recordings found yet; skipping audio sanity check.")
    print("\nDone. Use output/cozy_stt_v1_ct2_int8 with faster-whisper.")


if __name__ == "__main__":
    main()
