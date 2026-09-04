#!/usr/bin/env bash
# One-command, resumable SFT → RLVR → DPO → STT → benchmark pipeline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$ROOT/training_pipeline.py" "$@"
