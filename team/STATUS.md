# Cozy Team Status Board

## wakeword-agent
- Frozen-probe approach ABANDONED (3 collapses, AUC ceiling 0.72).
- Building CozyNet: end-to-end mel-CNN, standalone runtime.
- ETA: training script now, model in ~1 h.

## stt-agent
- DONE: whisper-small + LoRA v3, WER 9.92% user-holdout.
- CT2 conversion blocker open; HF fallback works.

## llm-agent
- v1.49 LoRA shipped (Qwen3-0.6B + r=16/alpha=32, 1.4k fn-call samples).
- Next loop: RLM harness (`assistant/rlm_harness/`) for ongoing data
  collection. `bash rlm.sh dataset` -> human oracle; `bash rlm.sh play`
  -> eval / DPO mining. Re-run `sft_qwen.py` after each merge.
