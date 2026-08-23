#!/usr/bin/env bash
# Environment setup for the Cozy wake-word pipeline.
# RESILIENT BY DESIGN:
#   * reruns are cheap - every installed piece is detected and skipped
#   * every pip call retries up to 5x with exponential backoff + long timeouts
#   * PyTorch falls back to the small CPU wheel if the CUDA build keeps failing
#   * openwakeword is installed with --no-deps because its tflite-runtime
#     dependency has NO wheels for Python 3.12 (we only use its ONNX paths)
set -euo pipefail
cd "$(dirname "$0")"

export PIP_DEFAULT_TIMEOUT=120
PYTHON="${PYTHON:-python3}"

if [[ ! -d .venv ]]; then
  echo "[setup] creating virtual environment (.venv)"
  "$PYTHON" -m venv .venv
fi

source .venv/bin/activate

pip_retry() {
  local attempt=1
  until python -m pip install "$@" --timeout 120 --retries 10; do
    if [[ $attempt -ge 5 ]]; then return 1; fi
    echo "[setup] pip failed (attempt $attempt/5) - backing off $((attempt * 15))s"
    sleep $((attempt * 15))
    attempt=$((attempt + 1))
  done
}

echo "[setup] bootstrapping pip tooling"
python -m pip install --upgrade pip wheel setuptools >/dev/null 2>&1 \
  || echo "[setup] (pip upgrade skipped - continuing)"

have() { python -c "import $1" >/dev/null 2>&1; }

install_with_retries() {
  local attempt
  for attempt in 1 2 3; do
    if pip_retry "$@"; then return 0; fi
    sleep 10
  done
  return 1
}

# --- torch (big one) ---
if have torch && [[ -f .venv/torch_flavor ]]; then
  echo "[setup] = torch already installed (flavor: $(cat .venv/torch_flavor))"
else
  echo "[setup] + installing PyTorch (CUDA build first, CPU fallback)"
  torch_ok=0
  for attempt in 1 2 3; do
    if pip_retry torch; then torch_ok=1; break; fi
    sleep 10
  done
  if [[ $torch_ok -eq 1 ]]; then
    echo cuda > .venv/torch_flavor
  else
    echo "[setup] CUDA torch kept failing -> falling back to CPU wheel"
    install_with_retries torch --index-url https://download.pytorch.org/whl/cpu
    echo cpu > .venv/torch_flavor
  fi
fi

# --- openwakeword WITHOUT tflite-runtime (py3.12 has no tflite wheels) ---
if have openwakeword; then
  echo "[setup] = openwakeword (already installed)"
else
  echo "[setup] + openwakeword (--no-deps: skips tflite-runtime, unavailable on py3.12)"
  install_with_retries --no-deps "openwakeword>=0.6.0" \
    || { echo "[setup] FAILED to install openwakeword" >&2; exit 1; }
fi

# --- everything else (import-name mapped) ---
declare -A PKGS=(
  [piper-sample-generator]=piper_sample_generator
  [numpy]=numpy
  [scipy]=scipy
  [soundfile]=soundfile
  [audiomentations]=audiomentations
  [onnxruntime]=onnxruntime
  [onnxscript]=onnxscript
  [scikit-learn]=sklearn
  [tqdm]=tqdm
  [requests]=requests
  [PyYAML]=yaml
  [sounddevice]=sounddevice
)

for req in "${!PKGS[@]}"; do
  mod="${PKGS[$req]}"
  if have "$mod"; then
    echo "[setup] = $req (already installed)"
  else
    echo "[setup] + installing $req"
    install_with_retries "$req" \
      || { echo "[setup] FAILED to install $req" >&2; exit 1; }
  fi
done

# --- openWakeWord feature models (cached; safe to rerun) ---
echo "[setup] fetching openWakeWord feature models"
for attempt in 1 2 3; do
  if python download_models.py --only-openwakeword; then
    break
  fi
  if [[ $attempt -eq 3 ]]; then
    echo "[setup] feature-model download still failing - rerun later; progress caches"
  fi
  sleep 15
done

echo ""
echo "[setup] done. Next steps:"
echo "  bash run_all.sh smoke   # quick end-to-end check (~10 min)"
echo "  bash run_all.sh full    # full dataset + training"
