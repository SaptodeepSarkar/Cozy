# Source me: source env.sh  — keeps every model/dataset byte inside stt-finetune/
export HF_HOME="$PWD/.hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
