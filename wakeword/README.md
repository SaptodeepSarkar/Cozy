# Wakeword Pipeline

Trains an openWakeWord-compatible detector for the word **cozy** entirely from
synthetic multi-speaker speech (Piper) plus your own microphone recordings,
then exports models/cozy_v1.onnx ready for real-time inference.

## Pipeline

    download_models.py      fetch TTS voices + feature models  -> work/
          |
    generate_data.py        synthesize positives/negatives     -> work/ + data/
          |
    train_wakeword.py       embed, train, export               -> models/cozy_v1.onnx
          |
    test_model.py           live mic / WAV verification

Or run everything at once:

    bash setup.sh             # one-time environment
    bash run_all.sh smoke     # ~10 min sanity pass (tiny counts)
    bash run_all.sh full      # full dataset (~18k clips) + training

## Scripts

| Script              | Purpose                                                        |
| ------------------- | -------------------------------------------------------------- |
| setup.sh            | create .venv, install PyTorch + dependencies                    |
| download_models.py  | Piper voices (6 accents), LibriTTS-R multi-speaker generator, openWakeWord feature models |
| generate_data.py    | synthesize all audio buckets (idempotent; --force regenerates)  |
| record_samples.py   | guided microphone recording into data/cozy                      |
| train_wakeword.py   | feature extraction, training, ONNX export, runtime sanity check |
| test_model.py       | score WAVs or listen live (--mic)                               |

## Data folders

### data/cozy/  (positives)
Recordings of the word **cozy**:
- recording_NNN.wav - YOUR voice, made via record_samples.py (make 20+)
- synth_NNN.wav - small synthetic demo subset copied by generate_data.py
The bulk of synthetic positives lives in work/synthetic/ (git-ignored) but is
regenerable any time.

### data/similar/  (hard negatives)
Words that sound close to cozy - nosy, rosy, Josie, Ozzie, dozy, posy, noisy -
so the model learns what is NOT cozy. Seeds are committed; the full per-word
sets are regenerated into work/similar/ during data generation.

## Configuration

Everything tunable lives in config.yaml: the wake word text and spelling
variants, the similar-words list, clip counts, Piper batch sizes and training
hyperparameters. After editing, rerun:

    python generate_data.py --force
    python train_wakeword.py

## Requirements

- Python 3.10-3.12, ffmpeg not required
- sudo apt install libportaudio2 (for microphone recording / live testing)
- ~4 GB disk in wakeword/work/ during a full run
- GPU optional but much faster (RTX 3050 trains comfortably); CPU works with
  smaller Piper batch sizes (handled automatically)

## Using the trained model

The exported ONNX plugs straight into openWakeWord:

    from openwakeword.model import Model
    model = Model(wakeword_models=["models/cozy_v1.onnx"],
                  inference_framework="onnx")
    score = model.predict(audio_chunk_int16_1280)   # 80 ms chunks

See test_model.py for a complete streaming example.

## Troubleshooting

- piper-sample-generator fails to install on exotic Pythons: use 3.11/3.12.
- No microphone in test_model.py: check libportaudio2 and arecord -L.
- Training data stale after config edits: add --force to generate_data.py.
- Everything is resumable: downloads cache, generated buckets skip when full.
