#!/usr/bin/env bash
# Run the Cozy voice assistant.
# Tries the assistant venv first; falls back to wakeword venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -x assistant/.venv/bin/python ]]; then
    exec assistant/.venv/bin/python assistant/runtime.py "$@"
elif [[ -x wakeword/.venv/bin/python ]]; then
    exec wakeword/.venv/bin/python assistant/runtime.py "$@"
else
    echo "No venv found. Run: bash setup.sh" >&2
    exit 1
fi
