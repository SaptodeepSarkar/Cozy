# AGENTS.md — Cozy wake word agent notes

Context for any AI agent picking up the wake word subsystem.

## What this is

A vendored copy of [livekit-wakeword](https://github.com/livekit/livekit-wakeword)
v0.2.0, plus a trained `hey_cozy.onnx` model and a regeneration pipeline for
real-voice training data.

## Critical files

- `output/hey_cozy/hey_cozy.onnx` — the trained classifier. **Do not commit
  changes without re-evaluating.**
- `output/hey_cozy/hey_cozy_eval.json` — last eval metrics. If you retrain,
  this updates.
- `configs/hey_cozy_test.yaml` — training config used for the current model.
  Edit `target_phrases`, `n_samples`, `steps` to retrain.
- `extract_user_voice.py` — regenerates `user_voice/` from stt-finetune
  recordings and old git history.
- `test_model.py` — CLI for testing the model on wav / live mic.

## Build & verify

```bash
# from wakeword/
source .venv/bin/activate

# Smoke-test the current model
uv run livekit-wakeword eval configs/hey_cozy_test.yaml -m output/hey_cozy/hey_cozy.onnx

# Live-mic test
python test_model.py --mic

# Retrain
uv run livekit-wakeword setup --config configs/hey_cozy_test.yaml --skip-acav
uv run livekit-wakeword run configs/hey_cozy_test.yaml
```

## Conventions

- `clip_NNNNNN.wav` is the livekit augment stage's required filename pattern.
  Other names are silently skipped.
- Heavy data (recordings, `work/`, `.venv/`) is gitignored.
- The model file is git-ignored under `output*/` but we force-add it on
  release commits. See `git log --diff-filter=A -- 'output/hey_cozy/hey_cozy.onnx'`.

## Common gotchas

1. **`run_extraction` upstream bug**: `cli.py` calls `run_extraction(config)`
   but the function signature is `run_extraction(config, sess_options)`. We
   patched it in `src/livekit/wakeword/data/features.py` to make
   `sess_options` optional. Keep this patch when pulling new upstream commits.
2. **Augment stage uses regex `^clip_\d{6}\.wav$`**: any user-voice wavs must
   be renamed to this pattern before they are processed by the livekit
   augment stage.
3. **Training on GPU with insufficient VRAM**: the model is tiny but the
   augment stage loads all features into memory. If you train on a smaller
   GPU, lower `batch_n_per_class.ACAV100M_sample` from 1024 to 256.
4. **Threshold calibration**: the eval report's `optimal_threshold` is the
   FPPH-minimizing threshold. For real-mic use, a lower threshold (0.20-0.30)
   gives better recall at the cost of more false fires.

## Don't break these

- The `run_extraction` patch in `features.py` (see gotcha #1).
- The 2-second audio window contract: `WakeWordModel.predict` needs ≥ 2 s of
  audio, not 80 ms like openwakeword used.
- The `user_voice/` regeneration script: it pulls from `e2a4dda` commit for
  the old `data/cozy/recording_*.wav` files. Don't drop that commit.
