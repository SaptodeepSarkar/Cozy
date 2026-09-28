#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v git >/dev/null || ! command -v uv >/dev/null; then
  echo "Install git and uv first (https://docs.astral.sh/uv/getting-started/installation/)." >&2
  exit 1
fi

if [[ ! -f hermes-agent/setup-hermes.sh ]]; then
  git submodule update --init --recursive hermes-agent
fi

echo "[1/3] Preparing Hermes Agent runtime..."
(cd hermes-agent && bash setup-hermes.sh --runtime-only \
  --runtime-extras=wake-livekit,voice,audio-io,edge-tts)

if [[ -f wakeword/output/hey_cozy/hey_cozy.onnx ]]; then
  echo "[3/3] Installing Cozy wake-word model into Hermes home..."
  install -d "${HERMES_HOME:-$HOME/.hermes}/wakewords"
  install -m 0644 wakeword/output/hey_cozy/hey_cozy.onnx \
    "${HERMES_HOME:-$HOME/.hermes}/wakewords/hey_cozy.onnx"
else
  echo "[3/3] Cozy wake-word model not present; voice wake activation will need a model."
fi

V6_STT_MODEL="${COZY_V6_STT_MODEL:-$HOME/.local/share/vaani/models/v6-stt-whisper-ami-clean-eosfix-1000-20260928-export/ct2-int8-float16}"
if [[ -d "$V6_STT_MODEL" ]]; then
  echo "[voice] Configuring Hermes to use ArchFlow V6 local STT: $V6_STT_MODEL"
  (cd hermes-agent && ./hermes config set stt.provider local && \
    ./hermes config set stt.local.model "$V6_STT_MODEL" && \
    ./hermes config set stt.local.device auto && \
    ./hermes config set stt.local.compute_type int8_float16)
else
  echo "[voice] ArchFlow V6 STT export not found; configure it later with COZY_V6_STT_MODEL."
fi

ARCHFLOW_ROOT="${COZY_ARCHFLOW_ROOT:-$ROOT/../ArchFlow}"
V6_CLEANUP_PYTHON="$ARCHFLOW_ROOT/training/cleanup-llm/.venv/bin/python"
V6_CLEANUP_SCRIPT="$ARCHFLOW_ROOT/training/cleanup-llm/scripts/llm-server.py"
V6_CLEANUP_MODEL="$HOME/.local/share/vaani/cleanup/v5-formatter-smollm2-360m"
V6_CLEANUP_ADAPTER="$HOME/.local/share/vaani/cleanup/v6-seq2seq-20260927"
if [[ -x "$V6_CLEANUP_PYTHON" && -f "$V6_CLEANUP_SCRIPT" && \
      -f "$V6_CLEANUP_MODEL/model.safetensors" && -f "$V6_CLEANUP_ADAPTER/adapter_model.safetensors" ]]; then
  echo "[voice] Enabling ArchFlow V6 transcript cleanup (local sidecar)."
  (cd hermes-agent && \
    ./hermes config set stt.local.v6_cleanup.enabled true && \
    ./hermes config set stt.local.v6_cleanup.python "$V6_CLEANUP_PYTHON" && \
    ./hermes config set stt.local.v6_cleanup.script "$V6_CLEANUP_SCRIPT" && \
    ./hermes config set stt.local.v6_cleanup.model "$V6_CLEANUP_MODEL" && \
    ./hermes config set stt.local.v6_cleanup.adapter "$V6_CLEANUP_ADAPTER")
else
  echo "[voice] V6 cleanup adapter not found; leave cleanup disabled until ArchFlow's models/venv are ready."
fi

bash "$ROOT/install-global.sh"

echo
echo "Setup finished. Start the desktop with: ./cozy"
echo "The first desktop launch may download/build its UI and computer-use dependencies."
