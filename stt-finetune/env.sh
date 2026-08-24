# Source me: source env.sh  — keeps every model/dataset byte inside stt-finetune/
export HF_HOME="$PWD/.hf_cache"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
