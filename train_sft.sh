#!/usr/bin/env bash
# Run SFT on the clean v2 dataset.
# Usage: bash train_sft.sh
set -e
cd "$(dirname "$0")/assistant"

# Sanity: dataset
if [ ! -f data/sft_train.jsonl ]; then
  echo "ERROR: data/sft_train.jsonl missing"
  echo "Run:  .venv/bin/python make_dataset_v2.py"
  exit 1
fi

mkdir -p model/sft_runs
.venv/bin/python sft_qwen.py \
  --base Qwen/Qwen3-0.6B \
  --out model/cozy-llm-v1 \
  --adapter-out model/cozy-llm-v1-adapter \
  --run-dir model/sft_runs \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 4 \
  --learning-rate 1e-4 \
  --max-length 1024 \
  --eval-steps 200 \
  --lora-r 16 \
  --workers 2 2>&1 | tee model/sft.log
