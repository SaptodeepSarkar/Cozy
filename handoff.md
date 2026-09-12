# Cozy Model and Agent Handoff

Date: 2026-09-13
Repo: `/home/saptodeep/Projects/Cozy`
Branch: `main`

This document is for the next agent working on Cozy's agentic behavior. Keep
the runtime, browser bridge, voice models, and model-training artifacts
separate so a new LLM can be tested by changing configuration only.

## Non-negotiable project rules

- Python 3.11+ and `uv` are the supported toolchain.
- Do not modify `wakeword/output/hey_cozy/hey_cozy.onnx`.
- Do not modify or commit private `.env` files or user recordings.
- Do not modify the trained STT model under
  `stt-finetune/output/cozy_stt_v1_ct2_int8/`.
- Keep all LLM experiments under `assistant/model/` and datasets under
  `assistant/data/` or `assistant/rlm_harness/`.
- Use `apply_patch` for source edits. Run tests before committing.
- Commit with `Saptodeep Sarkar <saptodeepsarkar28@gmail.com>`.

## Current model stack

### Voice models

- Wake word: `wakeword/output/hey_cozy/hey_cozy.onnx`
- STT: `stt-finetune/output/cozy_stt_v1_ct2_int8/`
- Transcript cleanup: optional Qwen3-0.6B cleanup model and adapter through
  `assistant/transcript_cleanup.py`
- TTS: Kokoro/local TTS through `assistant/rlm_harness/plugins/tts.py`

Do not retrain these while changing tool use. The action adapter must learn
actions, not speech recognition or speech synthesis.

### Current action LLM

The checked-in runtime default is:

- Base: Qwen3-0.6B, stored at `assistant/model/cozy-llm-v1/`
- Existing adapter: `assistant/model/cozy-llm-v1-adapter/`
- Optional DPO adapter: `assistant/model/cozy-llm-v1-dpo/` when valid adapter
  files are present
- Thinking is disabled at inference with `enable_thinking=False`; the model
  emits a concise answer or tool call rather than hidden chain-of-thought.

The current merged 0.6B model is the safest always-on laptop profile. The
repository contains the QLoRA path for a stronger model, but no trained 7B
agent adapter should be assumed to exist.

## Recommended model choices

Use one of these Hugging Face IDs as `COZY_AGENT_BASE`:

1. `Qwen/Qwen2.5-3B-Instruct` for the most reliable 6 GB laptop profile.
2. `Qwen/Qwen2.5-7B-Instruct` for the strongest practical NF4 QLoRA
   experiment. It may require unloading STT/TTS or CPU offload.
3. `Qwen/Qwen3-4B-Instruct-2507` as a newer alternative, after verifying its
   chat template and tool-call format.

Do not download a 9B fp16 model for this machine. A LoRA adapter is small, but
the frozen base still has to fit at inference time. NF4 3B is conservative;
NF4 7B is an experiment, not a guaranteed simultaneous voice-stack profile.

Hugging Face downloads happen automatically on first use. To make a local
copy outside git:

```bash
assistant/.venv/bin/huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
  --local-dir assistant/model/bases/qwen2.5-3b-instruct
```

Then point Cozy at it with `COZY_LLM_BASE`. Never commit base weights.

## Model switching

The runtime reads these variables:

```bash
export COZY_LLM_BASE="Qwen/Qwen2.5-3B-Instruct"
export COZY_LLM_ADAPTER="$PWD/assistant/model/adapters/qwen2.5-3b-cozy-tools-v1"
export COZY_LLM_4BIT=1
```

For a local base, use an absolute or repository-relative path. The adapter
must contain `adapter_config.json`, `adapter_model.safetensors`, and a
compatible tokenizer. Its `base_model_name_or_path` must match the base.

Never use a Qwen2.5 adapter with Qwen3, or vice versa. Keep one directory per
base:

```text
assistant/model/adapters/
  qwen2.5-3b-cozy-tools-v1/
  qwen2.5-7b-cozy-tools-v1/
```

Do not overwrite the known-good 0.6B files while experimenting.

## Dataset layout and preparation

The trainer expects JSONL rows like this:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"set the volume to 30"},{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"system.volume_set","arguments":"{\"level\":30}"}}]},{"role":"tool","content":"Volume set to 30"},{"role":"assistant","content":"Done. I set the volume to 30."}],"tools":[{"type":"function","function":{"name":"system.volume_set","description":"...","parameters":{"type":"object","properties":{"level":{"type":"integer"}},"required":["level"]}}}]}
```

Dataset rules:

- Include the complete current tool schema from `team/tool_schema.json`.
- Use exact tool names and valid JSON arguments.
- Include successes, failures, recovery, and multi-step tasks as separate
  assistant/tool/result turns.
- Include the final summary the user should hear.
- Do not include hidden chain-of-thought. Store observable status only.
- Remove API keys, cookies, email contents, private URLs, and personal data.
- Keep a deterministic validation split and never train on it.

### Existing data and trace collection

- Training: `assistant/data/sft_train.jsonl`
- Validation: `assistant/data/sft_val.jsonl`
- Runtime traces: `assistant/data/harness/trace.jsonl`

Collect candidate traces with:

```bash
bash rlm.sh info
bash rlm.sh dataset --limit 50
```

Review each generated trace before adding it to SFT. Keep only traces with
correct tool names, arguments, result handling, and final summaries.

### External datasets

External ToolBench/ToolAlpaca-style, API-Bank-style, or browser-agent data may
be used as seed material only after checking its license. Convert it to the
JSONL schema above, replace tools with Cozy's schema, and remove provider-only
APIs, secrets, and unverifiable actions. Local RLM traces are authoritative.

## Train the LoRA adapter

Install the environment first:

```bash
bash setup.sh
```

For the 6 GB QLoRA path:

```bash
COZY_AGENT_BASE=Qwen/Qwen2.5-3B-Instruct \
COZY_AGENT_ADAPTER_OUT="$PWD/assistant/model/adapters/qwen2.5-3b-cozy-tools-v1" \
bash train_agent_qlora.sh
```

For the larger experiment:

```bash
COZY_AGENT_BASE=Qwen/Qwen2.5-7B-Instruct \
COZY_AGENT_ADAPTER_OUT="$PWD/assistant/model/adapters/qwen2.5-7b-cozy-tools-v1" \
COZY_AGENT_EPOCHS=2 bash train_agent_qlora.sh
```

The script uses a frozen 4-bit NF4 base, LoRA rank 16, alpha 32, dropout
0.05, projection modules q/k/v/o and gate/up/down, batch size 1, gradient
accumulation 16, and max length 1024. It saves adapter-only output and does
not merge or rewrite the base model.

If CUDA is unavailable, stop and fix it instead of silently using CPU:

```bash
nvidia-smi
assistant/.venv/bin/python -c 'import torch; print(torch.cuda.is_available())'
```

Custom training is available through `assistant/sft_qwen.py`:

```bash
cd assistant
.venv/bin/python sft_qwen.py \
  --base Qwen/Qwen2.5-3B-Instruct \
  --adapter-out model/adapters/qwen2.5-3b-cozy-tools-v1 \
  --run-dir model/runs/qwen2.5-3b-cozy-tools-v1 \
  --qlora --adapter-only --epochs 2 --batch-size 1 --grad-accum 16 \
  --max-length 1024
```

Checkpoints belong under `assistant/model/runs/` and remain ignored. Final
adapters belong under `assistant/model/adapters/` and may be published apart
from the repository if large or private.

## Evaluation before activation

```bash
bash rlm.sh play --backend rule --limit 20
PYTHONPATH=assistant:. assistant/.venv/bin/python -m unittest discover \
  -s assistant -p 'test_*.py'
```

The evaluation set must cover local tools, FoxMCP navigation and extraction,
multi-step research, uploads/downloads, email drafting/sending, malformed
arguments, tool failures, and detailed summaries based on observed results.

Reject an adapter if it emits invalid JSON, loses numbers or URLs, repeats a
tool forever, hallucinates completion, or claims browser success while the
Firefox extension is disconnected.

## Agent behavior contract

`assistant/rlm_harness/harness_fast.py` builds prompts and parses the next
action. `assistant/executor.py` executes it. FoxMCP loads at startup and its
discovered tools are added to the system prompt. `browser.mcp` routes to the
exact FoxMCP tool name and argument object.

LoRA training happens offline; it is not retrained after each live tool call.
At runtime the frozen base plus adapter is loaded once. Multi-step behavior
must be represented by repeated decision/tool/result turns. Keep each
decision concise and produce the final detailed summary from actual results.

## Startup and hardware contract

OpenTUI launches one JSON-event runtime. Startup loads, in stable order, wake
word, STT, LLM, FoxMCP, cleanup, and TTS, then emits one `ready` event. Do
not reintroduce a text UI or another model-loading path.

The 6 GB target requires measurements for every new base: peak GPU memory,
cold-load time, tool-call accuracy, and whether STT/TTS remain usable while
the LLM is resident. If the larger base competes with voice models, use the
3B profile or add explicit idle unload/CPU offload before making it default.

## Handoff checklist

1. Prepare and validate new Cozy-format JSONL rows.
2. Train an adapter in a new directory under `assistant/model/adapters/`.
3. Run unit tests and the action evaluation set.
4. Smoke-test with `COZY_LLM_BASE` and `COZY_LLM_ADAPTER` overrides.
5. Test FoxMCP with Firefox open and its extension enabled.
6. Record model ID, adapter path, metrics, VRAM peak, and known failures.
7. Only then update defaults or commit source/documentation changes.
