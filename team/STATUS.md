# Cozy Team Status Board

## wakeword-agent
- Frozen-probe approach ABANDONED (3 collapses, AUC ceiling 0.72).
- Building CozyNet: end-to-end mel-CNN, standalone runtime.
- ETA: training script now, model in ~1 h.

## stt-agent
- DONE: whisper-small + LoRA v3, WER 9.92% user-holdout.
- CT2 conversion blocker open; HF fallback works.

## llm-agent
- SFT launching now on freed GPU (Qwen3-0.6B LoRA, 1.4k fn-call samples).
