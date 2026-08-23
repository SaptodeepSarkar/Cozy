#!/usr/bin/env bash
# One-time environment setup for the Cozy wake-word pipeline.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

if [[ ! -d .venv ]]; then
  echo "[setup] creating virtual environment (.venv)"
  "$PYTHON" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools >/dev/null

echo "[setup] installing PyTorch (default build includes CUDA support)"
python -m pip install torch

echo "[setup] installing pipeline dependencies"
python -m pip install -r requirements.txt

echo "[setup] fetching openWakeWord feature models (melspectrogram + embeddings)"
python download_models.py --only-openwakeword || true

echo ""
echo "[setup] done. Next steps:"
echo "  bash run_all.sh smoke   # quick end-to-end check (~10 min)"
echo "  bash run_all.sh full    # full dataset + training"
