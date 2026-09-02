#!/usr/bin/env bash
# Free up GPU VRAM by killing duplicate cozy-llm-v1 instances.
# 
# The cozy voice runtime (assistant/.venv) loads Qwen3-0.6B (1.28 GB
# on GPU) per process. If you have multiple cozies running, they
# each take 1.28 GB and your 6 GB dGPU fills up.
#
# This script:
#   1. Lists all cozy assistant processes
#   2. Kills all but the most recently started one (keeps the one
#      you probably launched yourself)
#
# Run with:  bash scripts/free_vram.sh
#
# Override the keep rule:  bash scripts/free_vram.sh --keep-all
# Override which to keep:  bash scripts/free_vram.sh --keep 3603808

set -euo pipefail

KEEP_ALL=0
KEEP_PID=""

for arg in "$@"; do
  case "$arg" in
    --keep-all) KEEP_ALL=1 ;;
    --keep)     shift; KEEP_PID="$1" ;;
    *)          echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

# Find cozy runtime PIDs (skip --harness-only which is a test runner)
PIDS=$(pgrep -f "assistant/.venv/bin/python.*assistant/runtime.py" | grep -v "\-\-harness-only" || true)

if [[ -z "$PIDS" ]]; then
  echo "No cozy runtime instances found."
  exit 0
fi

# Total GPU memory used by cozy
echo "Cozy runtime instances:"
TOTAL_MB=0
for pid in $PIDS; do
  MB=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
       | awk -v p="$pid" '$1==p{print $2}')
  MB=${MB:-0}
  cmdline=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
  echo "  pid=$pid  gpu=${MB}MB  cmd=$cmdline"
  TOTAL_MB=$((TOTAL_MB + MB))
done
echo "  TOTAL GPU: ${TOTAL_MB}MB"
echo ""

if [[ $KEEP_ALL -eq 1 ]]; then
  echo "--keep-all set, no kills"
  exit 0
fi

# Decide which PID to keep
if [[ -n "$KEEP_PID" ]]; then
  KEEP="$KEEP_PID"
else
  # Keep the most recently started one (largest etime)
  KEEP=$(for pid in $PIDS; do
    etime=$(ps -o etimes= -p $pid 2>/dev/null | tr -d ' ')
    echo "$etime $pid"
  done | sort -rn | head -1 | awk '{print $2}')
fi

echo "Keeping pid=$KEEP, killing the rest"
for pid in $PIDS; do
  if [[ "$pid" != "$KEEP" ]]; then
    echo "  killing $pid"
    kill -TERM "$pid" 2>/dev/null || true
  fi
done

# Give them a moment, then SIGKILL any stragglers
sleep 2
for pid in $PIDS; do
  if [[ "$pid" != "$KEEP" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "  SIGKILL $pid"
    kill -KILL "$pid" 2>/dev/null || true
  fi
done

echo ""
echo "After kill:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
