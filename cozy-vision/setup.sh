#!/usr/bin/env bash
# Cozy-Vision one-shot installer.
#
# Sets up a Python venv with all ML + Pop!_OS driver deps and verifies
# the two pretrained model weights are present.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- 1. System packages (Pop!_OS / Ubuntu) -------------------------------
if command -v apt >/dev/null 2>&1; then
  if [[ "${SKIP_APT:-0}" != "1" ]]; then
    echo "[setup] apt-get install ydotool grim wl-clipboard xdotool ..."
    sudo apt-get update
    sudo apt-get install -y \
        python3.11 python3.11-venv python3.12 python3.12-venv \
        ydotool grim slurp wl-clipboard xdotool xwayland \
        i3-wm sway wlr-screencopy-unstable \
        libxkbcommon0 libxcb-cursor0 || true
  fi
fi

# --- 2. uv-based venv ---------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -d .venv ]]; then
  echo "[setup] creating venv"
  uv venv .venv --python 3.12
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup] pip install (this may take a few minutes for torch + transformers)"
uv pip install --upgrade pip
uv pip install \
    "torch>=2.4" \
    "torchvision" \
    "transformers>=4.46" \
    "accelerate>=1.0" \
    "qwen-vl-utils>=0.0.10" \
    "bitsandbytes>=0.43" \
    "peft>=0.13" \
    "trl>=0.12" \
    "huggingface-hub>=0.26" \
    "safetensors>=0.4" \
    "pillow>=10" \
    "opencv-python>=4.10" \
    "python-xlib>=0.33" \
    "pyyaml>=6" \
    "pydantic>=2" \
    "tyro>=0.8" \
    "requests>=2.32" \
    "numpy>=1.26" \
    "pandas>=2.2" \
    "av"  # for video processor in Qwen2.5-VL

# --- 3. Models (use HF CLI; aria2c is too unreliable on this mirror) ---
if [[ ! -f models/qwen2.5-vl-3b/model-00001-of-00002.safetensors ]] || \
   [[ ! -f models/ui-tars-2b-sft/model-00001-of-00002.safetensors ]]; then
  echo "[setup] downloading models (this is a one-time ~17 GB download, ~20 min)"
  python scripts/download_models.py
else
  echo "[setup] models already present"
fi

# --- 4. Seed the task JSONL ---------------------------------------------
if [[ ! -f data/sft_planner_seed.jsonl ]]; then
  python scripts/make_sft_seed.py
fi

# --- 5. UI-TARS-2B sharded index (workaround) ---------------------------
if [[ -f models/ui-tars-2b-sft/model-00001-of-00002.safetensors ]] && \
   [[ ! -f models/ui-tars-2b-sft/model.safetensors.index.json ]]; then
  echo "[setup] generating UI-TARS-2B safetensors index"
  python -c "
from safetensors import safe_open
import json, sys
def get_keys_and_size(p):
    keys, sz = [], 0
    with safe_open(p, framework='pt') as f:
        for k in f.keys():
            keys.append(k)
            t = f.get_tensor(k)
            sz += t.numel() * t.element_size()
    return keys, sz
k1, s1 = get_keys_and_size('models/ui-tars-2b-sft/model-00001-of-00002.safetensors')
k2, s2 = get_keys_and_size('models/ui-tars-2b-sft/model-00002-of-00002.safetensors')
wm = {k: 'model-00001-of-00002.safetensors' for k in k1}
wm.update({k: 'model-00002-of-00002.safetensors' for k in k2})
with open('models/ui-tars-2b-sft/model.safetensors.index.json', 'w') as f:
    json.dump({'metadata': {'total_size': s1+s2}, 'weight_map': wm}, f, indent=2)
print('done')
"
fi

# --- 6. ydotool daemon hint ----------------------------------------------
if ! systemctl --user is-active ydotool >/dev/null 2>&1; then
  echo
  echo "[setup] ydotool daemon is not running. To enable it for your user:"
  echo "  systemctl --user enable --now ydotool"
  echo "  sudo usermod -aG input \$USER   # then log out and back in"
  echo
fi

echo
echo "============================================================"
echo " Cozy-Vision setup complete"
echo "  venv:      $HERE/.venv"
echo "  models:    $HERE/models/{qwen2.5-vl-3b,ui-tars-2b-sft}"
echo "  next:      bash run.sh smoke       # sanity check both models"
echo "             bash run.sh plan 'open firefox'"
echo "             bash run.sh run 'close the current window'"
echo "============================================================"
