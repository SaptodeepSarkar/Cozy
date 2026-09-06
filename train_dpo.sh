#!/usr/bin/env bash
# Run DPO training after SFT completes.
# Usage: bash train_dpo.sh
set -e
cd "$(dirname "$0")/assistant"

# Sanity: require SFT model
if [ ! -f model/cozy-llm-v1/model.safetensors ]; then
  echo "ERROR: model/cozy-llm-v1/model.safetensors missing"
  echo "Run:  bash train_sft.sh"
  exit 1
fi

mkdir -p model/cozy-llm-v1-dpo
.venv/bin/python dpo_light.py \
  --model model/cozy-llm-v1 \
  --pairs data/dpo_pairs.jsonl \
  --out model/cozy-llm-v1-dpo \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 4 \
  --lr 5e-5 \
  --max-length 1200 2>&1 | tee model/dpo.log
