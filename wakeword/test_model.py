#!/usr/bin/env python3
"""Test the trained Cozy wake-word model on your microphone or WAV files.

Usage:
    python test_model.py --mic                    # live listening
    python test_model.py --mic --threshold 0.6
    python test_model.py --wav some_clip.wav [more.wav ...]
"""
from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "cozy_v1.onnx"
CHUNK = 1280  # 80 ms at 16 kHz - openWakeWord's native frame


def load_model(model_path: Path, threshold: float):
    from openwakeword.model import Model

    if not model_path.exists():
        raise SystemExit(
            f"Model not found: {model_path}\n"
            "Train it first: bash run_all.sh (see wakeword/README.md)"
        )
    model = Model(wakeword_models=[str(model_path)],
                  inference_framework="onnx")
    # openWakeWord 0.6: model names live on .models (prediction_buffer
    # only fills after the first predict call)
    name = next(iter(model.models.keys()))
    return model, name


def predict_file(model, name: str, path: Path, threshold: float) -> float:
    import soundfile as sf

    pcm, sr = sf.read(path, dtype="int16")
    assert sr == 16000, f"{path}: expected 16 kHz, got {sr}"
    model.reset()
    best = 0.0
    for i in range(0, len(pcm) - CHUNK + 1, CHUNK):
        scores = model.predict(pcm[i:i + CHUNK])
        best = max(best, float(scores[name]))
    verdict = "DETECTED ✔" if best >= threshold else "clean"
    print(f"{path.name:<40} score={best:.3f}  [{verdict}]")
    return best


def run_mic(model, name: str, threshold: float) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("Install mic deps: pip install sounddevice "
                         "(OS package: sudo apt install libportaudio2)")

    recent: collections.deque[np.ndarray] = collections.deque(maxlen=8)
    last_hit = 0.0

    def callback(indata, _frames, _time, _status):
        recent.append(indata.copy())

    print(f"Listening for 'cozy' (threshold={threshold}). Ctrl-C to stop.")
    with sd.InputStream(samplerate=16000, channels=1, dtype="int16",
                        blocksize=CHUNK, callback=callback):
        idle_print = time.time()
        while True:
            if not recent:
                time.sleep(0.01)
                continue
            chunk = recent.popleft()[:, 0]
            score = float(model.predict(chunk)[name])
            now = time.time()
            if score >= threshold and now - last_hit > 2.0:
                last_hit = now
                print(f"\n🟢 COZY detected (score={score:.3f}) - "
                      "assistant would wake now\n")
            elif now - idle_print > 5.0:
                idle_print = now
                print(f"  ... listening (peak score {score:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mic", action="store_true", help="listen live")
    parser.add_argument("--wav", nargs="*", type=Path, default=[],
                        help="score WAV file(s) instead")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model, name = load_model(args.model, args.threshold)

    if args.wav:
        for wav in args.wav:
            predict_file(model, name, wav, args.threshold)
    elif args.mic:
        run_mic(model, name, args.threshold)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
