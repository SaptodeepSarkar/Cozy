# Cozy 🎙️

A local, private, voice-controlled assistant. Say **"hey cozy"** and your PC
(and your AI agents) come alive — fully offline, no cloud calls.

## Components

| Folder | Purpose | Venv |
|---|---|---|
| `wakeword/` | "hey cozy" detection (livekit-wakeword v0.2.0) | `wakeword/.venv` |
| `stt-finetune/` | Speech-to-text (Whisper-small LoRA, Hinglish-aware) | `stt-finetune/.venv` |
| `assistant/` | Voice assistant runtime: wake → STT → LLM → executor | `assistant/.venv` |
| `assistant/rlm_harness/` | RLM tool-call SFT data collector + evaluator | shares assistant venv |
| `team/` | Multi-agent notes, training scripts, status board | (no venv) |

## Quick start

```bash
# One-shot setup (creates all three venvs, ~5-10 min, ~3 GB disk)
bash setup.sh

# Talk to Cozy
bash run.sh                 # full voice loop
bash run.sh --text          # type commands instead
bash run.sh --no-wake       # skip wake gate
bash run.sh --calibrate     # print live wake scores for 30s
bash run.sh --threshold 0.50

# Collect tool-call SFT data with the RLM harness
bash rlm.sh info                                   # show task stats
bash rlm.sh dataset --limit 10                     # 10 traces by hand
bash rlm.sh play    --limit 50                     # log 50 model decisions
bash rlm.sh merge --source assistant/data/sft_extra.jsonl
```

## Architecture

```
                     ┌──────────────┐
                     │  microphone   │
                     └──────┬───────┘
                            │ 16 kHz int16
                            ▼
                  ┌──────────────────────┐
                  │  wakeword (livekit)  │  hey_cozy.onnx (122 KB)
                  │  2s window -> score  │  AUT 0.020, FPPH 1.66, Recall 69%
                  └──────────┬───────────┘
                             │  score >= 0.30
                             ▼
                  ┌──────────────────────┐
                  │   STT (faster-whisper)│  whisper-small LoRA v3
                  │  CT2 int8 (CUDA)      │  9.92% WER on user-holdout
                  └──────────┬───────────┘
                             │  text
                             ▼
                  ┌──────────────────────┐
                  │   LLM (Qwen3-0.6B)    │  LoRA r=16, alpha=32
                  │  + tool-call schema  │  15 tools
                  └──────────┬───────────┘
                             │  tool call
                             ▼
                  ┌──────────────────────┐
                  │   executor (Python)  │  system.volume.set, app.open,
                  │                       │  browser.search, screenshot.take,
                  │                       │  time.now, agent skills, ...
                  └──────────────────────┘
```

## Models shipped in this repo

| Model | Path | Size | Trained on |
|---|---|---|---|
| **hey_cozy** wake word | `wakeword/output/hey_cozy/hey_cozy.onnx` | 122 KB | 138 user-voice + 500 synth Piper positives, 2568 negatives |
| **hey_cozy** PyTorch | `wakeword/output/hey_cozy/hey_cozy.pt` | 104 KB | (same) |
| Whisper-small LoRA v3 (CT2) | `stt-finetune/output/cozy_stt_v1_ct2_int8/` | ~80 MB | user recordings + Indian English corpora |
| Whisper-small LoRA v3 (HF) | `stt-finetune/output/hf_finetuned/` | ~310 MB | (same) |
| Qwen3-0.6B base | `assistant/model/cozy-llm-v1/` | 1.2 GB | base from HuggingFace |
| Qwen3-0.6B LoRA adapter | `assistant/model/cozy-llm-v1-adapter/` | 40 MB | 1.4 k function-call samples (synthetic + STT seeds) |

### RLM harness

`assistant/rlm_harness/` is the data-collection side of the assistant.
Two modes:

* `bash rlm.sh dataset` — you (or another AI) play the oracle for every
  task; the trace is dumped to JSONL in the exact same schema as
  `assistant/data/sft_train.jsonl`.
* `bash rlm.sh play`    — run the current `cozy-llm-v1` on every task and
  log its decisions (for review and DPO pair mining).

`bash rlm.sh merge --source <file>` folds collected rows into the
training set; rerun `sft_qwen.py` to train the next iteration.

## Current state (v1.49)

- ✅ **Wake word** — `hey_cozy` model, AUT 0.020, FPPH 1.66, Recall 69% (user-voice trained)
- ✅ **STT** — whisper-small + LoRA v3, 9.92% WER on user-holdout
- ✅ **LLM** — Qwen3-0.6B + LoRA function-calling SFT
- ✅ **Assistant** — runtime wires wake → STT → LLM → executor end-to-end
- ✅ **All venvs set up** — one-command install via `bash setup.sh`
- 🚧 **Executor** — basic system tools; agent skills stubbed

## Folder layout

```
Cozy/
├── setup.sh              one-shot environment installer
├── run.sh                launch the assistant
├── README.md             this file
├── LICENSE
│
├── wakeword/             "hey cozy" detection
│   ├── README.md
│   ├── pyproject.toml    livekit-wakeword v0.2.0 (vendored)
│   ├── output/hey_cozy/  trained ONNX + eval metrics
│   ├── user_voice/       your real-voice data (regenerable)
│   ├── configs/          training YAMLs
│   ├── docs/             livekit-wakeword architecture docs
│   ├── test_model.py     live-mic / wav test CLI
│   ├── extract_user_voice.py  regenerate user_voice/ from sources
│   └── src/livekit/wakeword/  library source
│
├── stt-finetune/         Whisper-small finetune
│   ├── README.md
│   ├── env.sh            dGPU pinning, offline mode
│   ├── recordings/       your voice recordings (gitignored)
│   ├── scripts/          train_lora, prepare_data, infer, ...
│   ├── data/             Indian English corpora
│   ├── output/           CT2 + HF exports
│   └── third_party/      whisper.cpp / openai-whisper
│
├── assistant/            Voice assistant runtime
│   ├── README.md
│   ├── pyproject.toml    cozy-assistant v1.49
│   ├── runtime.py        main voice loop
│   ├── stt.py            STT dual-engine wrapper (CT2 + HF fallback)
│   ├── bridge.py         rule-based intent router
│   ├── intents.py        intent definitions
│   ├── executor.py       tool implementations
│   ├── sft_qwen.py       LLM SFT trainer
│   ├── make_dataset.py   tool-call dataset generator
│   ├── data/             SFT training data
│   └── model/
│       ├── cozy-llm-v1/         Qwen3-0.6B base
│       └── cozy-llm-v1-adapter/ LoRA adapter
│
└── team/                 Multi-agent team notes
    ├── STATUS.md
    ├── tool_schema.json  LLM tool definitions
    ├── scripts/          training scripts (legacy)
    ├── channel.jsonl     team communication log
    └── data/
```

## Hardware requirements

- **GPU**: NVIDIA RTX 3050 6 GB (or any ≥6 GB CUDA)
- **Disk**: ~5 GB for venvs, ~3 GB for downloaded models, ~2 GB for training data
- **RAM**: 16 GB recommended
- **OS**: Linux (Ubuntu 24+ tested)
- **Mic**: any USB / built-in

## Documentation per folder

- `wakeword/README.md` — wake word training & inference guide
- `stt-finetune/README.md` — STT training & inference guide
- `assistant/README.md` — assistant runtime modes
- `team/STATUS.md` — multi-agent status board

## Versioning

We use `vX.YZ` tags at the repo level for "full repo snapshot" releases, with
incremental `feat(...)` / `fix(...)` commits in between.

| Tag | Date | What |
|---|---|---|
| v1.47 | — | full repo snapshot (pre-wakeword rewrite) |
| v1.48 | — | swap in livekit-wakeword + train hey_cozy v1 |
| v1.49 | — | wire assistant runtime to livekit-wakeword + user-voice retrain |
