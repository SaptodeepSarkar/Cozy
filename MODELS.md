# Local speech models

The general-purpose agent model and tools are maintained by Hermes Agent; Cozy
no longer trains a local LLM or bespoke action harness.

| Component | Model/artifact | Location | Notes |
|---|---|---|---|
| Wake word | LiveKit `hey_cozy` ONNX | `wakeword/output/hey_cozy/hey_cozy.onnx` | Copied into `$HERMES_HOME/wakewords/` by setup |
| STT experiments | Whisper-small LoRA / CTranslate2 | `stt-finetune/output/` | User-local generated model files are ignored by Git |

Wake thresholds need new evaluation. A short labelled negative recording
triggered high scores in the current model, so old aggregate metrics are not
evidence of acceptable day-to-day false-wake behavior.
