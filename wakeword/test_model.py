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


def run_calibrate(seconds: float, model_path, threshold: float) -> None:
    """Record straight through the live-mode audio path, then score it."""
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        raise SystemExit("pip install sounddevice soundfile")
    secs = float(seconds)
    print("Recording " + format(secs, ".0f") + "s from your mic - say "
          "'cozy' a few times, then talk about anything else.")
    for tick in (3, 2, 1):
        print(tick, flush=True)
    pcm = sd.rec(int(secs * 16000), samplerate=16000, channels=1,
                 dtype="int16")
    sd.wait()
    out = Path("work") / "calib.wav"
    out.parent.mkdir(exist_ok=True)
    sf.write(str(out), pcm, 16000, subtype="PCM_16")
    peak = int(np.abs(pcm).max())
    print("saved " + str(out) + " | mic peak level "
          + str(peak) + "/32767"
          + (" (LOW - speak louder / check mic)" if peak < 800 else "")
          + (" (CLIPPING - lower mic gain!)" if peak > 32000 else ""))
    model, name = load_model(Path(model_path), threshold)
    pcm1 = pcm[:, 0] if pcm.ndim > 1 else pcm.reshape(-1)
    sec = 16000
    for start in range(0, len(pcm1) - sec + 1, sec):
        model.reset()
        best = 0.0
        for i in range(start, min(start + sec, len(pcm1) - 1280 + 1), 1280):
            best = max(best,
                       float(model.predict(pcm1[i:i + 1280])[name]))
        t = start // 16000
        bar = "#" * int(best * 40)
        print(format(t, "02d") + "s " + bar.ljust(40) + " "
              + format(best, ".3f")
              + ("  <-- WAKE" if best >= threshold else ""))


def run_mic(model, name: str, threshold: float, debug: bool = False) -> None:
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
    last_debug = 0.0
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
            elif debug and now - last_debug > 0.4:
                last_debug = now
                bar = "#" * int(min(score, 1.0) * 40)
                print("\r" + bar.ljust(42) + format(score, ".3f"),
                      end="", flush=True)
            elif not debug and now - idle_print > 5.0:
                idle_print = now
                print(f"  ... listening (peak score {score:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mic", action="store_true", help="listen live")
    parser.add_argument("--wav", nargs="*", type=Path, default=[],
                        help="score WAV file(s) instead")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--debug", action="store_true",
                        help="show a live score bar while listening")
    parser.add_argument("--calibrate", type=float, default=0,
                        metavar="SECONDS",
                        help="record SECONDS via the live audio path, save "
                             "work/calib.wav and score it per second")
    args = parser.parse_args()

    if args.calibrate:
        run_calibrate(args.calibrate, args.model, args.threshold)
        return

    model, name = load_model(args.model, args.threshold)

    if args.wav:
        for wav in args.wav:
            predict_file(model, name, wav, args.threshold)
    elif args.mic:
        run_mic(model, name, args.threshold, debug=args.debug)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
