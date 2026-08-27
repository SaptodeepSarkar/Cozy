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

```bash
# 1. Download TTS voices + feature extractors + background noise + RIRs
uv run livekit-wakeword setup --config configs/prod.yaml

# 2. Full pipeline: generate -> augment -> extract -> train -> export
uv run livekit-wakeword run configs/prod.yaml
```

Edit `configs/prod.yaml` (or copy `configs/test.yaml` for a small run):

```yaml
model_name: hey_cozy
target_phrases: ["hey cozy"]
custom_negative_phrases: ["hey cozy", "hey dozy", "hey posy", ...]
```

## Repository layout

| Path                              | Source                              |
|-----------------------------------|-------------------------------------|
| `src/livekit/wakeword/`           | The library (installable package)   |
| `configs/`                        | Sample YAML training configs        |
| `examples/`                       | Inference + listener scripts        |
| `examples/resources/*.onnx`       | Pre-trained "hey livekit" / "nihao livekit" classifiers |
| `docs/`                           | Architecture + data + training docs |
| `tests/`                          | Pytest suite                        |
| `swift/`                          | iOS / macOS Swift package           |
| `pyproject.toml` / `uv.lock`      | Python project + locked deps        |

## Why this and not a custom model?

- The **conv-attention** head achieves 60x lower AUT and 100x fewer false
  positives per hour than the flat DNN, while detecting 17% more wake words
  (see the [livekit-wakeword README](https://github.com/livekit/livekit-wakeword#why-livekit-wakeword)).
- Backward compatible with openWakeWord ONNX models and library.
- One YAML drives the whole pipeline: TTS synthesis, augmentation, training,
  ONNX export, and DET-curve evaluation.
- Multilingual (30+ languages) via VoxCPM2 TTS.

## Upstream

https://github.com/livekit/livekit-wakeword — Apache 2.0
