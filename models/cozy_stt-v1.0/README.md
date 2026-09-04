# cozy_stt-v1.0 (baseline)

This is the **unmodified** `openai/whisper-small` base model. No
fine-tuning has been applied. It is the reference for WER comparisons
against `cozy_stt-v1.1`.

- Base: `openai/whisper-small` (244 M params, fp16, en)
- Training data: none
- Test WER on `santhosh_indian` held-out: see `models/benchmarks/stt_wer.csv`
- Source: https://huggingface.co/openai/whisper-small
