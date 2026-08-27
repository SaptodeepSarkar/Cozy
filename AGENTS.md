# AGENTS.md — Cozy project conventions for AI agents

This file gives an AI agent enough context to navigate the repo and follow
project conventions without re-reading everything from scratch.

## Read this first

```
Cozy/
├── README.md              high-level overview, models, quick start
├── setup.sh               one-shot environment installer
├── run.sh                 launch the assistant
├── assistant/             voice assistant runtime + LLM SFT
├── wakeword/              "hey cozy" detection (livekit-wakeword)
├── stt-finetune/          Whisper-small LoRA finetune
├── team/                  multi-agent notes + tool schema
```

## Critical paths (do not break)

- `wakeword/output/hey_cozy/hey_cozy.onnx` is the trained wake word model.
  If you change it, the assistant's wake detection will break.
- `stt-finetune/output/cozy_stt_v1_ct2_int8/` is the trained STT model.
- `assistant/model/cozy-llm-v1/` is the LLM base.
- `assistant/runtime.py` is the live voice loop. Edit it to change behavior.

## Build & verify

```bash
# from repo root
bash setup.sh                              # one-time environment setup
bash run.sh --text                         # smoke test (no mic needed)
bash run.sh --calibrate                    # test wake word via mic
cd wakeword && source .venv/bin/activate
uv run livekit-wakeword eval configs/hey_cozy_test.yaml -m output/hey_cozy/hey_cozy.onnx
```

## Conventions

- Git commit format: `feat(...):` or `fix(...):` for incremental; `vX.YZ:` for
  full-repo snapshots.
- Python ≥ 3.11. `uv` is the package manager. Each subfolder has its own
  venv — don't try to share.
- Heavy data (recordings, training data, model weights) is gitignored.
- Model files under `output/`, `model/`, and `user_voice/` are not committed
  by default (regenerate from sources or fetch from a release).

## Common gotchas

1. **Assistant venv is empty after a fresh clone.** Run `bash setup.sh` to
   install deps in all three venvs.
2. **Wakeword venv is the source of truth for `livekit-wakeword`.** The
   assistant venv also has it (via pip install), but the wakeword venv is
   the editable install.
3. **GPU contention**: the LLM + STT both want the dGPU. The assistant uses
   `cuda:0` for both. The wakeword training runs on CPU.
4. **The wakeword model needs 2-second windows** at inference (not the 80 ms
   streaming chunks that openwakeword used). The runtime buffers 2 s of
   audio before scoring.
5. **`livekit-wakeword` augment stage requires `clip_NNNNNN.wav` filename
   pattern.** Other names are silently skipped.

## Don't break these

- The `wakeword/output/hey_cozy/hey_cozy.onnx` is on the critical path for
  every voice command. Any change requires retraining (`cd wakeword &&
  uv run livekit-wakeword run configs/hey_cozy_test.yaml`).
- The `assistant/runtime.py` wake loop uses a rolling 2 s buffer; do not
  refactor it to use 80 ms chunks (that would silently drop detection rate).
- The `stt-finetune/env.sh` sets `CUDA_VISIBLE_DEVICES=0` to pin to dGPU.
  Don't override in a child process.
