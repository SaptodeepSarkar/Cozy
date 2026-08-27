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

# 1. wakeword venv: livekit-wakeword, training deps
echo "[1/3] wakeword/.venv ..."
if [[ ! -d wakeword/.venv ]]; then
    uv venv wakeword/.venv --python "$PY_VERSION"
fi
wakeword/.venv/bin/python -m pip install --quiet --upgrade pip
wakeword/.venv/bin/python -m pip install --quiet -e "wakeword/[listener,train,eval,export]"

# 2. stt-finetune venv: faster-whisper, transformers, torch
echo "[2/3] stt-finetune/.venv ..."
if [[ ! -d stt-finetune/.venv ]]; then
    uv venv stt-finetune/.venv --python "$PY_VERSION" --system-site-packages
fi
stt-finetune/.venv/bin/python -m pip install --quiet --upgrade pip
stt-finetune/.venv/bin/python -m pip install --quiet \
    transformers datasets accelerate peft jiwer soundfile librosa \
    ctranslate2 faster-whisper pyarrow

# 3. assistant venv: wake + STT + LLM runtime deps
echo "[3/3] assistant/.venv ..."
if [[ ! -d assistant/.venv ]]; then
    uv venv assistant/.venv --python "$PY_VERSION"
fi
assistant/.venv/bin/python -m pip install --quiet --upgrade pip
assistant/.venv/bin/python -m pip install --quiet \
    livekit-wakeword pyaudio sounddevice soundfile \
    faster-whisper librosa transformers torch torchaudio \
    huggingface-hub safetensors tokenizers pyyaml numpy

echo
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
