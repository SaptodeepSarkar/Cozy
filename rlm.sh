#!/usr/bin/env bash
# Shortcut to invoke the Cozy RLM harness from the repo root.
#   bash rlm.sh info
#   bash rlm.sh dataset --limit 10
#   bash rlm.sh play --limit 50
#   bash rlm.sh serve
#   bash rlm.sh merge --source <file>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Pick the first available venv with the assistant deps
if [[ -x assistant/.venv/bin/python ]]; then
    PY=assistant/.venv/bin/python
elif [[ -x wakeword/.venv/bin/python ]]; then
    PY=wakeword/.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    echo "No Python found. Run: bash setup.sh" >&2
    exit 1
fi

exec "$PY" -m assistant.rlm_harness "$@"
