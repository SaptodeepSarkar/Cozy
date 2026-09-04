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
- `assistant/model/cozy-llm-v1-adapter/` is the LoRA adapter.
- `assistant/runtime.py` is the live voice loop. Edit it to change behavior.
- `assistant/rlm_harness/` is the data-collection side of the LLM. Use
  `bash rlm.sh dataset` to grow the SFT set, `bash rlm.sh play` to
  evaluate. The output JSONL drops straight into `sft_qwen.py`.

## Build & verify

```bash
# from repo root
bash setup.sh                              # one-time environment setup
bash run.sh --text                         # smoke test (no mic needed)
bash run.sh --calibrate                    # test wake word via mic
cd wakeword && source .venv/bin/activate
uv run livekit-wakeword eval configs/hey_cozy_test.yaml -m output/hey_cozy/hey_cozy.onnx

# RLM harness
bash rlm.sh info                                 # task + tool stats
bash rlm.sh dataset --limit 5                    # collect 5 SFT traces
bash rlm.sh play    --backend rule --limit 20    # rule-based smoke eval
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
## Cozy shell alias (added in v2.0)

After `bash setup.sh`, source the alias block from your shell rc:

```bash
echo 'source "$(dirname $(realpath $(which cozy 2>/dev/null || echo ./cozy)))/cozy.shell"' >> ~/.bashrc
echo 'source "$(dirname $(realpath $(which cozy 2>/dev/null || echo ./cozy)))/cozy.shell"' >> ~/.zshrc
```

Or just re-run `bash setup.sh` — it writes the correct line for you.

Then:

```bash
cozy                  # full voice loop (wake + STT + LLM + executor + TTS)
cozy --text           # type commands
cozy --calibrate      # 30s wake score log
cozy --no-wake        # skip wake gate
cozy --no-tts         # log replies, no audio
cozy --threshold 0.5  # custom threshold
cozy --status         # which models are on disk
cozy --stop           # kill any running cozy
cozystop              # alias for --stop
cozystatus            # alias for --status
```

## Cozy-Vision (desktop GUI agent)

`cozy-vision/` is the local desktop GUI agent that sits next to the
voice stack. It uses a 3B VLM (Qwen2.5-VL-3B-Instruct) as the
**planner** (vision + reasoning, outputs a precise, checkable todo
list grounded in the live OS context) and a 2B VLA (UI-TARS-2B-SFT)
as the **executor** (native GUI grounding, outputs click/type/hotkey
actions). Both run on the dGPU in NF4 4-bit via bitsandbytes, with
the cozy voice stack sharing the same 6 GB.

Quick start:
```bash
cd cozy-vision
bash setup.sh                # one-shot venv + model fetch
bash run.sh smoke            # load both models + synthetic inference
bash run.sh plan "open firefox"   # planner only
bash run.sh ask "what window is focused?"  # VLM Q&A
bash run.sh run "close the current window"  # full agent
bash run.sh collect --tasks 5   # collect SFT traces
bash run.sh train                # QLoRA SFT on the VLM
bash run.sh train-vla            # QLoRA SFT on the VLA
```

Model paths (downloaded by setup.sh):
- VLM: `cozy-vision/models/qwen2.5-vl-3b/` (Qwen2.5-VL-3B-Instruct, fp16, ~7.5 GB)
- VLA: `cozy-vision/models/ui-tars-2b-sft/` (ByteDance UI-TARS-2B-SFT, fp16, ~9.8 GB)

Hardware budget on RTX 3050 6 GB:
- VLM NF4: 2.4 GB GPU, VLA NF4: 1.5 GB GPU, cosy-llm-v1: 1.2 GB, STT: 0.5 GB, wake: 0.1 GB
