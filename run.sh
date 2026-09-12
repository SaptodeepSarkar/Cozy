#!/usr/bin/env bash
# Run the Cozy voice assistant.
# Keep the documented entry point on the same OpenTUI interface as `cozy`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -x assistant/.venv/bin/python ]]; then
    echo "No venv found. Run: bash setup.sh" >&2
    exit 1
fi
exec "$ROOT/cozy" "$@"
