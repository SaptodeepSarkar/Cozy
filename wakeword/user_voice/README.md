# User voice training data

This directory contains WAV files extracted from your real voice recordings
to anchor the wake-word model on how YOU actually say "hey cozy" / "cozy",
and to provide a diverse set of negatives (your normal speech, no wake word).

## What's here

| Folder | Count | Source | Purpose |
|---|---|---|---|
| `user_cozy/` | 32 | `wakeword/data/cozy/recording_001-032.wav` (git history: commit `e2a4dda`) | Bare "cozy" recordings from your voice — strong positives |
| `user_pos_stt/` | 106 | `stt-finetune/recordings/session_{1,3,5,6,7,8,9,41}/*.wav` containing "cozy" | 2s windows around your "Hey Cozy" / "Cozy" commands |
| `user_neg_stt/` | 2068 | `stt-finetune/recordings/session_*/*.wav` NOT containing "cozy" | 2s windows of your normal speech, commands, etc. — hard negatives that match your acoustic environment |

## How to regenerate

The wavs are large (~143 MB) and git-ignored. Regenerate from sources with:

```bash
cd wakeword
.venv/bin/python extract_user_voice.py
```

## How to use in training

These wavs are added to livekit-wakeword's training data by copying them
into the output directory's `positive_train/` and `negative_train/` folders
(renamed to match livekit's `clip_NNNNNN.wav` pattern, which is required
by the livekit augment stage's `^clip_\d{6}\.wav$` filter).

See `train_cozynet_v2.py` (older) or the livekit `augment`/`train` pipeline
for the full workflow.
