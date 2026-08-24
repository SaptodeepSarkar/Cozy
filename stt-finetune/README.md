# Cozy STT Finetune 🎤→📝

On-device finetuning of **Whisper-small** so it transcribes **your voice** and
**Indian English** far better than the stock model. Everything — base model,
dataset, venv, checkpoints and the final model — lives inside this folder.

Hardware target: NVIDIA RTX 3050 6GB (any ≥6GB CUDA GPU works).

## How it works

| Stage | Script | What it does |
|---|---|---|
| 0. Setup | `source env.sh` + `.venv` | project-local HF cache; reuses system CUDA torch |
| 1. Assets | `scripts/download_assets.py` + `scripts/build_indian_corpus.py` | `openai/whisper-small` + Indian-English corpora: **kaushalgawri/indian_accent_en_train** (~4.5k clips tagged accent=indian) & **skbose NPTEL Indian-English lectures** → decoded to plain wavs under `data/` |
| 2. Record | `python3 scripts/record_voice.py` | guided mic recorder, 6 themed sessions × 15 lines, resumable |
| 3. Manifests | `scripts/prepare_data.py` | merges FLEURS + your clips (auto up-samples yours ~12% of each epoch), holds out one session as personal test set |
| 4. Baseline | `scripts/baseline_eval.py` | WER of the stock model on the same eval set |
| 5. Train | `scripts/train_lora.py` | LoRA (r=32 on q/v projections), fp16, gradient checkpointing — fits in 6GB |
| 6. Export | `scripts/merge_export.py` | merges LoRA → plain HF model → **CTranslate2 int8_float16** for faster-whisper |
| 7. Use | `scripts/infer.py --mic 5` | transcribe from mic or file with your model |

## Quickstart

```bash
cd ~/Projects/Cozy/stt-finetune

# one-time
uv venv .venv --system-site-packages && \
uv pip install --python .venv/bin/python transformers datasets accelerate peft jiwer soundfile librosa ctranslate2 faster-whisper pyarrow
source env.sh
.venv/bin/python scripts/download_assets.py        # whisper-small base model
.venv/bin/python scripts/build_indian_corpus.py    # Indian-English corpora (~3.7 GB, one-time)

# record your voice (no venv needed) — run repeatedly across sittings
python3 scripts/record_voice.py            # next unfinished session
python3 scripts/record_voice.py --list     # progress overview

# build manifests + measure stock-model quality
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/baseline_eval.py --tag baseline

# finetune (~40-70 min on RTX 3050)
.venv/bin/python scripts/train_lora.py

# export + try it
.venv/bin/python scripts/merge_export.py
.venv/bin/python scripts/infer.py --mic 5
```

## Notes

- **Why LoRA?** Full finetuning of even whisper-small doesn't fit comfortably in
  6GB VRAM; LoRA adapters (<0.5% trainable params) match its quality for accent
  adaptation and merge cleanly back afterwards.
- **Your voice is private:** recordings stay in `recordings/` (gitignored).
- **Text normalization:** training labels use Whisper's English normalizer, so
  the finetuned model outputs clean lowercase text — ideal for an assistant
  intent router.
- Re-run `prepare_data.py` any time you add recordings, then retrain.
- The CTranslate2 export plugs straight into the Cozy roadmap's listener daemon:
  `WhisperModel("output/cozy_stt_v1_ct2_int8", compute_type="int8_float16")`.
