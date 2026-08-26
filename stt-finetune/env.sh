# Source me: source env.sh  — keeps every model/dataset byte inside stt-finetune/
export HF_HOME="$PWD/.hf_cache"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- dGPU priority: always train/infer on the RTX 3050, never fall back ---
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_DEVICE_ORDER=FASTEST_FIRST
export TOKENIZERS_PARALLELISM=false

# --- FULLY LOCAL: never touch the network at train/inference time ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
