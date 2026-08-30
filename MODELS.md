# Cozy Models Catalog

All models used by the Cozy voice assistant, with versions, paths, and
metrics. This is the source of truth for "what model version are we on?".

## v1.49 (current)

| Component | Model | Version | Path | Size | Metrics | Date |
|---|---|---|---|---|---|---|
| Wake word | `hey_cozy` (livekit-wakeword) | v2 | `wakeword/output/hey_cozy/hey_cozy.onnx` | 122 KB | AUT 0.020, FPPH 1.66, Recall 69% | 2026-08-28 |
| STT | `whisper-small` + LoRA v3 (CT2) | v1 | `stt-finetune/output/cozy_stt_v1_ct2_int8/` | ~80 MB | WER 9.92% user-holdout | 2026-08-26 |
| STT | `whisper-small` + LoRA v3 (HF) | v1 | `stt-finetune/output/hf_finetuned/` | ~310 MB | (HF fallback) | 2026-08-26 |
| LLM | `Qwen3-0.6B` + LoRA r=16 | v1 | `assistant/model/cozy-llm-v1/` + `cozy-llm-v1-adapter/` | 1.2 GB + 40 MB | 15-tool function calling | 2026-08-26 |
| RLM harness | tool-call SFT data collector | v1 | `assistant/rlm_harness/` | (code only) | `dataset` / `play` / `serve` modes | 2026-08-28 |

## Versioning

Each model has an independent version. The repo-level `vX.YZ` tag tracks the
full snapshot of all models. To upgrade a single model, retrain it and
update its eval JSON; the repo version only changes when there's a
significant integration change.

## Wake word model lineage

| Version | Trained on | Key change | AUT | FPPH | Recall |
|---|---|---|---|---|---|
| cozy_v1 (openwakeword) | 500 synth Piper | Frozen-probe + openwakeword | 0.725 | 8.5 | 68.6% (claimed) |
| cozynet_v2 (CozyNet) | 156 real-voice | End-to-end mel-CNN, custom | 0.997 AUC | n/a | 86% (claimed) |
| **hey_cozy v1** (livekit) | 500 synth Piper | First livekit-wakeword | 0.0510 | 12.18 | 68.0% |
| **hey_cozy v2** (livekit + user voice) | 138 user + 500 synth | User voice augmentation | **0.0195** | **1.66** | 69.0% |

## STT model lineage

| Version | Trained on | WER (user-holdout) | Notes |
|---|---|---|---|
| whisper-small (stock) | n/a | 11.93% | Indian English baseline |
| whisper-small + LoRA v3 | user recordings + Indian corpora | **9.92%** | production |

## LLM model lineage

| Version | Base | Adapter | Notes |
|---|---|---|---|
| cozy-llm-v1 | Qwen3-0.6B | LoRA r=16, alpha=32 | 1.4 k function-call samples |
