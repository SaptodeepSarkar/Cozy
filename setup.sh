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
(cd hermes-agent && bash setup-hermes.sh --runtime-only)

echo "[2/3] Installing local voice dependencies..."
(cd hermes-agent && ./hermes pm install --extra wake-livekit --extra voice --extra audio-io)

if [[ -f wakeword/output/hey_cozy/hey_cozy.onnx ]]; then
  echo "[3/3] Installing Cozy wake-word model into Hermes home..."
  install -d "${HERMES_HOME:-$HOME/.hermes}/wakewords"
  install -m 0644 wakeword/output/hey_cozy/hey_cozy.onnx \
    "${HERMES_HOME:-$HOME/.hermes}/wakewords/hey_cozy.onnx"
else
  echo "[3/3] Cozy wake-word model not present; voice wake activation will need a model."
fi

echo
echo "Setup finished. Start the desktop with: ./cozy"
echo "The first desktop launch may download/build its UI and computer-use dependencies."
