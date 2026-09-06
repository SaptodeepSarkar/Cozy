#!/usr/bin/env bash
# Full retraining pipeline: SFT then DPO.
# Usage: bash retrain_all.sh
set -e
cd "$(dirname "$0")"

echo "=== Step 1: Generate datasets ==="
assistant/.venv/bin/python assistant/make_dataset_v2.py
assistant/.venv/bin/python assistant/make_dpo_pairs.py

echo "=== Step 2: SFT ==="
bash train_sft.sh

echo "=== Step 3: DPO ==="
bash train_dpo.sh

echo "=== Step 4: Smoke test ==="
assistant/.venv/bin/python assistant/smoke_test.py assistant/model/cozy-llm-v1
echo
echo "=== Done. Run:  bash run.sh --text  ==="
