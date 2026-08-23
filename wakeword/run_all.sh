#!/usr/bin/env bash
# End-to-end: download models -> generate data -> train -> sanity-check.
# Usage: bash run_all.sh [full|smoke]    (default: full)
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-full}"

if [[ ! -d .venv ]]; then
  bash setup.sh
fi
source .venv/bin/activate

echo "[run_all/$MODE] step 1/3 - downloading models (best effort)"
timeout 900 python -u download_models.py \
  || echo "[run_all/$MODE] WARN: some downloads incomplete - continuing with cached models"

echo "[run_all/$MODE] step 2/3 - generating training data"
python -u generate_data.py --mode "$MODE"

echo "[run_all/$MODE] step 3/3 - training + exporting the wake word model"
python -u train_wakeword.py

echo ""
echo "[run_all/$MODE] finished. Try it live:"
echo "  python test_model.py --mic          # say 'cozy'"
echo "  python record_samples.py --num 20   # add your own voice, then retrain"
