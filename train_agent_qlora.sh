#!/usr/bin/env bash
# Train only Cozy's larger action adapter on a 6 GB NVIDIA GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/assistant"
BASE="${COZY_AGENT_BASE:-Qwen/Qwen2.5-7B-Instruct}"
OUT="${COZY_AGENT_ADAPTER_OUT:-$ROOT/assistant/model/cozy-agent-adapter}"
RUN="${COZY_AGENT_RUN_DIR:-$ROOT/assistant/model/cozy-agent-runs}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,18p' "$ROOT/train_agent_qlora.sh"
  exit 0
fi

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "ERROR: an NVIDIA GPU is required for QLoRA training" >&2; exit 1;
}
if ! .venv/bin/python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  echo "ERROR: PyTorch cannot see a CUDA device; training was not started" >&2
  exit 1
fi
echo "Frozen base: $BASE"
echo "Adapter:     $OUT"
exec .venv/bin/python sft_qwen.py \
  --base "$BASE" --adapter-out "$OUT" --out "$RUN/merged-debug" \
  --run-dir "$RUN" --qlora --adapter-only \
  --epochs "${COZY_AGENT_EPOCHS:-2}" --batch-size 1 \
  --grad-accum "${COZY_AGENT_GRAD_ACCUM:-16}" \
  --learning-rate "${COZY_AGENT_LR:-2e-4}" \
  --max-length "${COZY_AGENT_MAX_LENGTH:-1024}" \
  --eval-steps 100 --lora-r 16 --workers 1 \
  ${COZY_AGENT_RESUME:+--resume}
