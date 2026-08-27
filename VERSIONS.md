# Cozy Release History

## v1.49 (current)
**Wire assistant to livekit-wakeword + user-voice retrain**
- Assistant runtime switched from openwakeword (cozy_v1.onnx) to
  livekit-wakeword (hey_cozy.onnx)
- Wake word model retrained with 138 user-voice positives + 2568 negatives
  (AUT 0.020, FPPH 1.66, Recall 69%)
- Default threshold updated to 0.30
- Three-venv setup (`wakeword/`, `stt-finetune/`, `assistant/`)
- Comprehensive READMEs in every folder
- `setup.sh` and `run.sh` at repo root for one-shot install + launch

## v1.48
**Wakeword → livekit-wakeword + trained hey_cozy v1**
- Vendored livekit-wakeword v0.2.0 (replaces custom CozyNet v1/v2)
- Trained hey_cozy model (122 KB ONNX, AUT 0.051, FPPH 12.18, Recall 68%)
- 5 YAML configs, full pytest suite, Swift package
- Removed 1.1 GB LLM safetensors that blocked pushes

## v1.47 (pre-wakeword rewrite)
- Full repo snapshot before wakeword pipeline swap
- Custom CozyNet v1/v2 with openwakeword embeddings
- 32 user "cozy" recordings, 60+ hard-negative similar-word recordings
- ASHA-based training with energy gate
- AUC ~0.997 on synthetic test set, 0.76 stream-level AUC

## Earlier
- LLM SFT: Qwen3-0.6B + LoRA, 1.4 k function-call samples
- STT: whisper-small + LoRA v3, WER 9.92% on user holdout
- Cozy assistant runtime: STT → LLM → executor (15 tools)
- 657+ user voice recordings across 44 sessions
- Indian English accent adaptation (kaushalgawri, santhosh corpora)
