# Cozy Wake Word (livekit-wakeword)

This folder is a vendored copy of [livekit-wakeword](https://github.com/livekit/livekit-wakeword),
an open-source wake-word library with a modern **conv-attention** classifier head
(1D temporal convolutions + multi-head self-attention) on top of frozen
mel-spectrogram + speech-embedding ONNX feature extractors.

The previous custom pipeline (CozyNet v1/v2) was retired; this library is
maintained upstream and ships a pre-trained `hey_livekit.onnx` model out of
the box, plus a YAML-driven training pipeline to retrain for any wake word.

## Quick start (inference with the pre-trained model)

```bash
# Install
uv sync --all-extras                 # all deps (train + eval + export + listener)
# or, for inference only:
uv pip install livekit-wakeword pyaudio

# Run the live mic listener on the bundled "hey livekit" model
uv run python examples/listener.py
```

## Quick start (train your own wake word)

The repo-root `setup.sh` already created `wakeword/.venv` with all
training deps. To re-train the "hey_cozy" model:

```bash
cd wakeword
source .venv/bin/activate

# 1. Download TTS voices + feature extractors + background noise + RIRs
uv run livekit-wakeword setup --config configs/hey_cozy_test.yaml --skip-acav

# 2. Full pipeline: generate -> augment -> extract -> train -> export
uv run livekit-wakeword run configs/hey_cozy_test.yaml
```

Edit `configs/hey_cozy_test.yaml` (or copy `configs/test.yaml` for a small run):

```yaml
model_name: hey_cozy
target_phrases: ["hey cozy"]
custom_negative_phrases: ["hey dozy", "hey posy", ...]
```

## Test the trained model

```bash
# Test on a wav file
python test_model.py --wav path/to/recording.wav

# Live mic test (uses default threshold from hey_cozy_eval.json)
python test_model.py --mic

# Lower threshold for higher recall
python test_model.py --mic --threshold 0.20
```

The default threshold (0.30) was tuned on real voice. Adjust with
`--threshold` for your environment.

## Models

| Model | Path | Size | Trained on |
|---|---|---|---|
| **hey_cozy** ONNX | `output/hey_cozy/hey_cozy.onnx` | 122 KB | 138 user-voice + 500 synth positives, 2568 negatives |
| **hey_cozy** PyTorch | `output/hey_cozy/hey_cozy.pt` | 104 KB | (same) |
| hey_livekit (upstream) | `examples/resources/hey_livekit.onnx` | 953 KB | upstream's training set |

## hey_cozy validation metrics (16.91h held-out)

| Metric | Value |
|---|---|
| AUT (Area Under DET curve) | 0.0195 |
| FPPH (False Positives Per Hour) | 1.66 |
| Recall @ threshold 0.50 | 69.0% |
| Recall @ optimal threshold 0.60 | 50.3% |

## Repository layout

```
wakeword/
├── pyproject.toml       livekit-wakeword v0.2.0 (vendored, editable)
├── uv.lock              locked dep tree
├── README.md            this file
├── AGENTS.md            agent-specific notes
├── LICENSE               Apache 2.0
│
├── output/hey_cozy/     trained model artifacts
│   ├── hey_cozy.onnx
│   ├── hey_cozy.pt
│   ├── hey_cozy_eval.json
│   ├── hey_cozy_metrics.json
│   └── hey_cozy_det.png
│
├── user_voice/          real-voice training data (regenerable)
│   ├── user_cozy/       32 bare "cozy" recordings (from old data/cozy/)
│   ├── user_pos_stt/     106 2s windows from STT "cozy" recordings
│   ├── user_neg_stt/     2068 2s windows from non-cozy STT recordings
│   └── README.md
│
├── configs/             training YAMLs
│   ├── hey_cozy_test.yaml   our trained config
│   ├── prod.yaml             production scale
│   ├── test.yaml             quick smoke test
│   ├── prod_voxcpm.yaml      multilingual
│   └── test_voxcpm.yaml      multilingual smoke test
│
├── src/livekit/wakeword/   library source (editable)
│   ├── cli.py
│   ├── config.py
│   ├── inference/
│   │   ├── model.py       WakeWordModel class
│   │   └── listener.py    WakeWordListener (mic + sliding window)
│   ├── models/
│   │   ├── classifier.py  conv_attention + dnn + rnn heads
│   │   └── feature_extractor.py  mel + speech embedding ONNX
│   ├── data/
│   │   ├── generate.py    TTS synthesis (Piper / VoxCPM)
│   │   ├── augment.py     audio augmentation
│   │   ├── features.py    feature extraction (.npy)
│   │   └── tts/           Piper + VoxCPM TTS backends
│   ├── training/
│   │   └── trainer.py     3-phase adaptive training
│   ├── export/onnx.py     ONNX export
│   └── eval/evaluate.py   DET curve, AUT, FPPH, recall
│
├── examples/             usage examples
│   ├── inference.py
│   ├── listener.py
│   ├── ios_wakeword/      iOS/macOS Swift demo
│   └── resources/         pre-trained hey_livekit ONNX
│
├── docs/                 architecture + data + training docs
├── tests/                pytest suite
├── skypilot/             cloud GPU training config
├── swift/                iOS / macOS Swift package
├── test_model.py         live-mic / wav test CLI (this repo)
└── extract_user_voice.py  regenerate user_voice/ from sources
```

## Why livekit-wakeword?

- **conv-attention** head achieves 60x lower AUT and 100x fewer false
  positives per hour than a flat DNN, while detecting 17% more wake words
  (see upstream results).
- Backward compatible with openWakeWord ONNX models and library.
- One YAML drives the whole pipeline: TTS synthesis, augmentation, training,
  ONNX export, and DET-curve evaluation.
- Multilingual (30+ languages) via VoxCPM2 TTS.

## Upstream

https://github.com/livekit/livekit-wakeword — Apache 2.0
