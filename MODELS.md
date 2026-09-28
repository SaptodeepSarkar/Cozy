# Local speech models

The general-purpose agent model and tools are maintained by Hermes Agent; Cozy
no longer trains a local LLM or bespoke action harness.

| Component | Model/artifact | Location | Notes |
|---|---|---|---|
| Wake word | LiveKit `hey_cozy` ONNX | `wakeword/output/hey_cozy/hey_cozy.onnx` | Copied into `$HERMES_HOME/wakewords/` by setup |
| STT | ArchFlow V6 Whisper-small CTranslate2 | `$XDG_DATA_HOME/vaani/models/v6-stt-whisper-ami-clean-eosfix-1000-20260928-export/ct2-int8-float16/` | Selected for Cozy by explicit request; user model files stay outside Cozy |
| Transcript cleanup | ArchFlow V6 seq2seq adapter | `$XDG_DATA_HOME/vaani/cleanup/v6-seq2seq-20260927/` | Runs as a local sidecar; source-preservation guard falls back to raw STT |
| Cleanup base | SmolLM2 360M base | `$XDG_DATA_HOME/vaani/cleanup/v5-formatter-smollm2-360m/` | V6 adapter's declared base model; not a V5 cleanup adapter |

Wake thresholds need new evaluation. A short labelled negative recording
triggered high scores in the current model, so old aggregate metrics are not
evidence of acceptable day-to-day false-wake behavior. The V6 STT/cleanup
model choice is user-directed; run ArchFlow's evaluation tools for quality
before treating the candidate as production-validated.
