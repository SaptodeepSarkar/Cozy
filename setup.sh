#!/usr/bin/env bash
# Cozy project setup - one-shot environment installer.
# Creates three independent venvs (one per component) with all deps.
# Run from the project root: bash setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY_VERSION="${PY_VERSION:-3.11}"
echo "=== Cozy project setup (Python $PY_VERSION) ==="
echo

# 0. System packages (Arch / Debian / Ubuntu).
# pyaudio needs portaudio headers; libsndfile is needed by soundfile/librosa;
# ffmpeg by librosa. python3.11 from system packages on Arch ships in extra/.
if command -v pacman >/dev/null 2>&1 && [[ "${SKIP_PACMAN:-0}" != "1" ]]; then
    echo "[setup] pacman -S python portaudio libsndfile ffmpeg ..."
    sudo pacman -S --needed --noconfirm \
        python python-pip \
        portaudio libsndfile ffmpeg sox \
        alsa-utils pipewire pipewire-alsa pipewire-pulse wireplumber \
        base-devel git curl || true
elif command -v apt >/dev/null 2>&1 && [[ "${SKIP_APT:-0}" != "1" ]]; then
    echo "[setup] apt-get install python3 portaudio libsndfile ffmpeg ..."
    sudo apt-get update
    sudo apt-get install -y \
        python3 python3-venv python3-pip \
        portaudio19-dev libsndfile1 ffmpeg sox \
        libnotify-bin alsa-utils pulseaudio-utils \
        build-essential git curl || true
fi

# 1. wakeword venv: livekit-wakeword, training deps
echo "[1/3] wakeword/.venv ..."
if [[ ! -d wakeword/.venv ]]; then
    uv venv wakeword/.venv --python "$PY_VERSION" --seed
fi
uv pip install --python wakeword/.venv/bin/python --quiet --upgrade pip
uv pip install --python wakeword/.venv/bin/python --quiet -e "wakeword/[listener,train,eval,export]"

# 2. stt-finetune venv: faster-whisper, transformers, torch
echo "[2/3] stt-finetune/.venv ..."
if [[ ! -d stt-finetune/.venv ]]; then
    uv venv stt-finetune/.venv --python "$PY_VERSION" --seed --system-site-packages
fi
uv pip install --python stt-finetune/.venv/bin/python --quiet --upgrade pip
uv pip install --python stt-finetune/.venv/bin/python --quiet \
    transformers datasets accelerate peft jiwer soundfile librosa \
    ctranslate2 faster-whisper pyarrow

# 3. assistant venv: wake + STT + LLM runtime deps
echo "[3/3] assistant/.venv ..."
if [[ ! -d assistant/.venv ]]; then
    uv venv assistant/.venv --python "$PY_VERSION" --seed
fi
uv pip install --python assistant/.venv/bin/python --quiet --upgrade pip
uv pip install --python assistant/.venv/bin/python --quiet \
    livekit-wakeword pyaudio sounddevice soundfile \
    faster-whisper librosa transformers torch torchaudio \
    huggingface-hub safetensors tokenizers pyyaml numpy \
    peft trl accelerate

# 4. Download the LLM base model if it's not already on disk.
# Qwen3-0.6B is small (~1.2 GB) and free. Required for `cozy` to start.
LLM_DIR="$ROOT/assistant/model/cozy-llm-v1"
if [[ ! -f "$LLM_DIR/model.safetensors" ]]; then
    if [[ "${SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
        echo "[setup] SKIP_MODEL_DOWNLOAD=1; skipping LLM fetch"
    else
        echo "[setup] Fetching Qwen3-0.6B base model into $LLM_DIR ..."
        assistant/.venv/bin/python - <<PY
import os
from pathlib import Path
from huggingface_hub import snapshot_download
target = Path("$LLM_DIR")
target.mkdir(parents=True, exist_ok=True)
snapshot_download(
    "Qwen/Qwen3-0.6B",
    local_dir=str(target),
    allow_patterns=[
        "*.json", "*.txt", "*.model", "*.tiktoken",
        "*.safetensors", "tokenizer*", "chat_template*",
    ],
)
print(f"[setup] LLM weights in {target}")
PY
    fi
fi

# Install the cozy shell alias if it isn't already sourced.
ALIAS_LINE="source \"$ROOT/cozy.shell\""
for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
    [ -f "$rc" ] || touch "$rc"
    if ! grep -qF "cozy.shell" "$rc" 2>/dev/null; then
        printf "\n# Cozy voice assistant\n%s\n" "$ALIAS_LINE" >> "$rc"
    fi
done
# Kokoro-82M is the TTS engine. It's a Python pip package and the
# model auto-downloads from hexgrad/Kokoro-82M on first speak().
# No system package needed.
echo "  TTS: Kokoro-82M (pip package, model downloads on first use)"
echo
# 4. Terminal UI: deterministic install from package-lock.json.
if command -v npm >/dev/null 2>&1; then
    echo "[ui] assistant/tui-node ..."
    npm --prefix assistant/tui-node ci --silent
else
    echo "[ui] skipped (Node.js 20+ and npm are required for the terminal UI)" >&2
fi

echo "=== Setup complete ==="
echo
echo "Three venvs created:"
echo "  wakeword/.venv/        - wake word training + inference"
echo "  stt-finetune/.venv/   - speech-to-text training + inference"
echo "  assistant/.venv/      - full voice assistant runtime"
echo
echo "Run the assistant:"
echo "  cd wakeword && ./.venv/bin/python ../assistant/runtime.py"
echo "  (or: bash run.sh)"
echo
echo "Train the wake word model:"
echo "  cd wakeword && source .venv/bin/activate"
echo "  uv run livekit-wakeword setup --config configs/hey_cozy_test.yaml --skip-acav"
echo "  uv run livekit-wakeword run configs/hey_cozy_test.yaml"
echo
echo "Train the STT model:"
echo "  cd stt-finetune && source env.sh && .venv/bin/python scripts/train_lora.py"
echo
echo "Train the LLM (function-calling SFT):"
echo "  cd assistant && .venv/bin/python sft_qwen.py"

if [[ "${COZY_SKIP_GLOBAL:-0}" != "1" ]]; then
    bash "$ROOT/install-global.sh"
fi
