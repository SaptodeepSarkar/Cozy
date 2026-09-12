# Cozy Assistant Runtime

The voice assistant that sits behind the wake word. Wires together:

**wakeword → STT → LLM → executor**

The default stack runs locally on an NVIDIA RTX 3050 6GB (any ≥6 GB CUDA GPU
works). For temporary capability testing, Cozy can route its action model
through OpenRouter while wake word, STT, cleanup, TTS, and tool execution stay
on this machine.

## Quick start

The repo-root `setup.sh` already created `assistant/.venv` with all
dependencies. To run:

```bash
# from the repo root
bash run.sh                            # full voice loop
bash run.sh --no-wake                  # skip wake gate (always transcribe)
bash run.sh --calibrate                # 30s live wake-score log
bash run.sh --threshold 0.50          # custom wake threshold

# or run directly
./.venv/bin/python runtime.py
```

If the assistant venv is missing, recreate it with:

```bash
uv venv .venv --python 3.11
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install \
    livekit-wakeword pyaudio sounddevice soundfile \
    faster-whisper librosa transformers torch torchaudio \
    huggingface-hub safetensors tokenizers pyyaml numpy
```

## Files

| File | Purpose |
|---|---|
| `runtime.py` | Main voice loop (wake → STT → LLM → executor) |
| `stt.py` | Dual-engine STT wrapper: fast CT2 with HF fallback |
| `intents.py` | Intent definitions (set_volume, open_app, ...) |
| `executor.py` | Tool implementations (system.volume.set, etc.) |
| `sft_qwen.py` | LLM SFT trainer (LoRA on Qwen3-0.6B) |
| `../train_agent_qlora.sh` | 6 GB-safe QLoRA trainer for the larger action model |
| `make_dataset.py` | Function-call dataset generator |
| `model/cozy-llm-v1/` | Qwen3-0.6B base model |
| `model/cozy-llm-v1-adapter/` | LoRA adapter (40 MB) |
| `data/` | SFT training data (jsonl) |
| `rlm_harness/` | RLM tool-call SFT data collector + evaluator (see its README) |

## Runtime modes

| Mode | Description |
|---|---|
| (default) | Live mic: wake → record command → STT → LLM → executor |
| `--no-wake` | Skip wake word; always record + transcribe |
| `--calibrate` | 30 s live wake-score log; prints peak score per inference |
| `--threshold N` | Custom wake threshold (default reads from `wakeword/output/hey_cozy/hey_cozy_eval.json`) |

## Models used

- **Wake word**: `../wakeword/output/hey_cozy/hey_cozy.onnx` (livekit-wakeword, 122 KB)
- **STT**: `../stt-finetune/output/cozy_stt_v1_ct2_int8` (CTranslate2 int8)
  with HF fallback at `../stt-finetune/output/hf_finetuned`
- **LLM**: `model/cozy-llm-v1/` (Qwen3-0.6B base, bf16) + `model/cozy-llm-v1-adapter/` (LoRA r=16, alpha=32)

For stronger tool selection, use the frozen 7B-class Qwen2.5-Instruct base:

```bash
COZY_AGENT_BASE=Qwen/Qwen2.5-7B-Instruct bash train_agent_qlora.sh
COZY_LLM_BASE=Qwen/Qwen2.5-7B-Instruct \
COZY_LLM_ADAPTER=assistant/model/cozy-agent-adapter COZY_LLM_4BIT=1 \
bash run.sh
```

Only the adapter is retrained when tools or examples change; the NF4 loader
keeps the model within the 6 GB GPU budget alongside the audio stack.

### OpenRouter capability testing

When the repo-root `.env` contains `OPENROUTER_API`, Cozy automatically uses
`inclusionai/ling-3.0-flash-vl:free` for planning and tool selection. The key
is never printed or committed. The normal tool schema is sent with each
request, and returned tool calls go through the same Cozy executor and FoxMCP
browser bridge as the local model.

That model currently advertises a 262,144-token context window. Cozy uses
that as the OpenRouter context budget by default (`COZY_MAX_CONTEXT_TOKENS`,
`COZY_COMPACT_THRESHOLD`, and `COZY_RECENT_TURNS` can override it). The full
conversation and tool results remain in the parent RLM harness trace; only a
model-selected `rlm.delegate` creates a child RLM with its own scoped trace.

Set `COZY_OPENROUTER_MODEL` to change the temporary remote model. Remove the
key or unset it to return to the local LoRA-backed model.

### Terminal safety boundary

`terminal.run` runs through Bubblewrap. It can read system files, but its only
writable mounts are the owner's home directory and `/tmp`; `/etc`, `/usr`,
and the rest of the system are read-only. `sudo`, `doas`, `pkexec`, and `su`
are refused inside that sandbox. For an action that genuinely needs elevation,
`terminal.elevate` opens a separate interactive terminal and the owner enters
the password there. Cozy never receives, stores, or sends a password.

### Firefox browser tools

Run `bash install_foxmcp.sh` once, install and enable the FoxMCP Firefox
extension, then start Cozy normally. FoxMCP starts during the OpenTUI loading
screen, its tools are discovered before the assistant becomes ready, and every
browser operation is included in the final task summary.

### Files, Python kernel, and external MCP

Attach a text file directly to a voice/task request with `@path/to/file`.
Cozy bounds attachment text before it enters the LLM context. The agent can
also use `python.kernel` for bounded calculations and parsing in a persistent
IPython kernel; install dependencies with `uv sync` after changing the
assistant environment.

Copy `cozy.jsonc.example` to `cozy.jsonc` to add OpenCode-compatible MCP
servers. Cozy discovers stdio and HTTP servers at startup and exposes their
tools as `mcp.<server>.<tool>` through `mcp.call`:

```jsonc
{
  "mcp": {
    "filesystem": {
      "type": "stdio",
      "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/home/me"]
    }
  }
}
```

Keep credentials in the server's environment, never in the config committed
to git. See `../handoff.md` for the model/context and dataset contract.

## venv auto-detection

`runtime.py` automatically scans for the wakeword venv and adds its
site-packages to `sys.path` so `livekit.wakeword` can be imported
regardless of which `python` is used to launch the assistant.

If you have multiple venvs, it tries in this order:
1. `<repo>/.venv/lib/python3.X/site-packages/`
2. `<repo>/wakeword/.venv/lib/python3.X/site-packages/`
3. `assistant/.venv/lib/python3.X/site-packages/` (if already there)

## Conversation example

```
[runtime] READY - say 'hey cozy' then your command.
[wake] hey_cozy! (score 0.683, thr 0.3)
[heard] set a timer for fifteen minutes
[you] set a timer for fifteen minutes
[cozy] Done. 15-minute timer set.
```

## Tool schema

The LLM has access to 15 tools (see `../team/tool_schema.json`):
- `system.volume.set`
- `system.volume.get`
- `system.brightness.set`
- `app.open`
- `app.close`
- `browser.search`
- `browser.open`
- `screenshot.take`
- `time.now`
- `date.now`
- `clipboard.read`
- `clipboard.write`
- `file.read`
- `shell.run` (with confirm prompt)
- `agent.delegate` (bridge to local agent skills)

## RLM harness (data collection for the next SFT)

`rlm_harness/` is the loop that grows the SFT dataset. After the v1.49
LoRA is shipped, every new tool, every phrasing the user invents, every
edge case can be recorded as one short trace and folded into the next
training run:

```bash
# from the repo root
bash rlm.sh info                                # task counts + tool list
bash rlm.sh dataset --limit 20                  # collect 20 traces by hand
bash rlm.sh play    --limit 100 --also-sft      # log 100 model decisions
bash rlm.sh merge --source assistant/data/sft_extra.jsonl
python assistant/sft_qwen.py                    # retrain
```

The collected JSONL uses the same `{messages, tools}` schema as
`make_dataset.py`, so it drops straight into `sft_qwen.py` with no
glue.

## v1.49 changes

- Switched from `openwakeword.model.Model` + `cozy_v1.onnx` (no longer exists)
  to `livekit.wakeword.WakeWordModel` + `hey_cozy.onnx` (newer, better).
- Wake loop now accumulates a rolling 2 s audio buffer (livekit needs ≥ 2 s
  windows, unlike openwakeword's 80 ms streaming).
- Default threshold reads from `hey_cozy_eval.json` (uses the eval-optimal).
