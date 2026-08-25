# Cozy Team Status Board

## wakeword-agent
- v13 retrain running (streaming-matched features, normalized recordings,
  energy-gated labels, balanced loss). Watcher auto-finalizes:
  export -> acceptance gate -> commit. ETA ~1 h.
- Deliverable contract: models/cozy_v1.onnx (openWakeWord-compatible),
  thresholds in models/metrics.json (use safe_threshold_zero_fpr).

## stt-agent
- FINETUNING DONE: whisper-small + LoRA (v1-v3); eval WER 11.73% -> 10.45%
  overall, 9.92% on personal holdout, corpus 10.50%. Data: 1,425 Indian-English
  corpus clips + 105 user clips (incl Hinglish session).
- Merged HF model verified working (stt-finetune/output/hf_finetuned);
  CTranslate2 export producing empty outputs - root-causing now.
- Next: assistant glue layer (wake -> STT -> intent router) + CT2 fix.

## llm-agent
- Phase starting: small-model fine-tune (Qwen3-0.6B primary,
  SmolLM2-360M fallback) for function calling + conversation +
  system control. Dataset generator being written.
