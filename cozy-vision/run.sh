#!/usr/bin/env bash
# Cozy-Vision launcher. Use one of the subcommands below.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || {
  echo "No .venv found. Run bash setup.sh first." >&2
  exit 1
}

# Start ydotoold if available (faster than ydotool's fallback path)
# It needs the user to be in the 'input' group.
if ! pgrep -x ydotoold >/dev/null 2>&1; then
  if command -v ydotoold >/dev/null 2>&1; then
    echo "[run] starting ydotoold daemon"
    ydotoold >/tmp/ydotoold.log 2>&1 &
  fi
fi

# Honour dGPU pinning like the rest of Cozy
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# PYTORCH_CUDA_ALLOC_CONF helps with the 6 GB fragmentation
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Quiet the transformers progress bars
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

# When COZY_VISION_PAUSE_COZY=1, SIGSTOP all cosy assistant processes
# before running the vision task, and SIGCONT after. This frees ~1.7 GB
# of GPU VRAM (cozy-llm-v1) so the VLM and VLA can both fit.
# When unset, the vision task will run alongside cosy-llm-v1 (which may
# OOM; not recommended).
# When COZY_VISION_SWAP_COZY=1, kill any cozy-llm-v1 processes before
# the vision task and respawn them after. This frees ~1.7 GB of GPU
# VRAM. Set to 0 to keep cosy alive (and accept the OOM risk on the
# 6 GB dGPU).
export COZY_VISION_SWAP_COZY="${COZY_VISION_SWAP_COZY:-1}"
export COZY_VISION_RESPAWN_COZY="${COZY_VISION_RESPAWN_COZY:-1}"
# Cap each model to leave room for the cozy voice stack on the same dGPU.
# VLM Qwen2.5-VL-3B NF4 needs ~2.4 GB on GPU; VLA UI-TARS-2B NF4 needs ~1.5 GB.
# cosy-llm-v1 takes ~1.2 GB, STT ~0.5 GB, wake ~0.1 GB.
# Total: 2.4 + 1.5 + 1.2 + 0.5 + 0.1 = 5.7 GB of 6 GB.
export COZY_VISION_VLM_GPU_MEM="${COZY_VISION_VLM_GPU_MEM:-2.5GiB}"
export COZY_VISION_VLA_GPU_MEM="${COZY_VISION_VLA_GPU_MEM:-2.0GiB}"

exec python -m harness.cli "$@"
