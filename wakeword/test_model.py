#!/usr/bin/env python3
"""Test the trained 'hey_cozy' wake-word model on your microphone or WAV files.

Built on top of livekit-wakeword (https://github.com/livekit/livekit-wakeword).
The model lives at output/hey_cozy/hey_cozy.onnx and was trained via
  uv run livekit-wakeword run configs/hey_cozy_test.yaml

Real-time mic listening uses the upstream WakeWordListener (handles
audio capture, sliding windows, and debounce internally).

Usage:
    python test_model.py --mic                    # live listening (threshold 0.37)
    python test_model.py --mic --threshold 0.5    # custom threshold
    python test_model.py --wav some_clip.wav      # score a single wav
    python test_model.py --wav a.wav b.wav c.wav   # score multiple wavs
    python test_model.py --calibrate 8            # record 8s and score per-second
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "output" / "hey_cozy" / "hey_cozy.onnx"
DEFAULT_META = HERE / "output" / "hey_cozy" / "hey_cozy_eval.json"
SR = 16000
CHUNK = 32000  # 2 seconds — livekit-wakeword's recommended inference size


def load_model(model_path: Path):
    """Load the wake-word ONNX and return (model, model_name)."""
    from livekit.wakeword import WakeWordModel
    if not model_path.exists():
        raise SystemExit(
            f"Model not found: {model_path}\n"
            f"Train it first:\n"
            f"  cd {HERE}\n"
            f"  uv run livekit-wakeword setup --config configs/hey_cozy_test.yaml --skip-acav\n"
            f"  uv run livekit-wakeword run configs/hey_cozy_test.yaml"
        )
    model = WakeWordModel(models=[model_path])
    name = next(iter(model._classifiers.keys()))
    return model, name


def print_meta(model_path: Path, threshold: float) -> None:
    meta_path = model_path.with_name(model_path.stem + '_eval.json')
    if not meta_path.exists():
        print(f"loaded model | threshold={threshold} (no meta file)")
        return
    meta = json.loads(meta_path.read_text())
    print(f"loaded {meta_path.parent.name}.onnx | "
          f"threshold={threshold} | "
          f"AUT={meta.get('aut', 0):.4f}  FPPH={meta.get('fpph', 0):.2f}  "
          f"Recall={meta.get('recall', 0)*100:.1f}%")


# ----------------------------------------------------------------- --wav mode

def run_wav(paths, threshold):
    """Score WAV files using the stateless model.predict() pattern
    (matches examples/inference.py from upstream livekit-wakeword)."""
    model, name = load_model(DEFAULT_MODEL)
    for wav in paths:
        pcm, sr = sf.read(str(wav), dtype="int16")
        if sr != SR:
            from scipy.signal import resample_poly
            pcm = resample_poly(pcm.astype(np.int16), SR, sr).astype(np.int16)
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1).astype(np.int16)
        STRIDE = 1280 * 4  # 320ms
        if len(pcm) < CHUNK:
            pcm = np.pad(pcm, (0, CHUNK - len(pcm)))
        best = 0.0
        for start in range(0, len(pcm) - CHUNK + 1, STRIDE):
            chunk = pcm[start:start + CHUNK]
            scores = model.predict(chunk)
            best = max(best, float(scores[name]))
        verdict = "DETECTED" if best >= threshold else "clean"
        print(f"{wav.name:<48} peak={best:.3f}  [{verdict}]")


# --------------------------------------------------------------- --calibrate

def run_calibrate(seconds, threshold):
    """Record N seconds from the mic, then score each 2s window."""
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("Install mic deps: pip install sounddevice")
    model, name = load_model(DEFAULT_MODEL)
    secs = float(seconds)
    print(f"Recording {secs:.0f}s from your mic - say 'hey cozy' a few "
          f"times, then talk about anything else.")
    for tick in (3, 2, 1):
        print(tick, flush=True)
    pcm = sd.rec(int(secs * SR), samplerate=SR, channels=1, dtype="int16")
    sd.wait()
    out = HERE / "work" / "calib_hey_cozy.wav"
    out.parent.mkdir(exist_ok=True)
    sf.write(str(out), pcm, SR, subtype="PCM_16")
    peak = int(np.abs(pcm).max())
    print(f"saved {out} | mic peak {peak}/32767"
          + (" (LOW - speak louder / check mic)" if peak < 800 else "")
          + (" (CLIPPING - lower mic gain!)" if peak > 32000 else ""))
    pcm1 = pcm[:, 0] if pcm.ndim > 1 else pcm.reshape(-1)
    pcm1 = pcm1.astype(np.int16)
    for start in range(0, len(pcm1) - SR * 2 + 1, SR):
        seg = pcm1[start:start + SR * 2]
        s = float(model.predict(seg)[name])
        t = start // SR
        bar = "#" * int(s * 40)
        print(f"{t:02d}s {bar.ljust(40)} {s:.3f}"
              + ("  <-- WAKE" if s >= threshold else ""))


# ---------------------------------------------------------------- --mic mode

def run_mic(threshold):
    """Live mic listening via the upstream WakeWordListener.

    This is the same listener shown in examples/listener.py — handles
    audio capture, sliding windows, and debounce internally.
    """
    from livekit.wakeword import WakeWordListener
    model, name = load_model(DEFAULT_MODEL)

    async def main():
        async with WakeWordListener(model, threshold=threshold, debounce=2.0) as listener:
            print(f"Listening for '{name}' (threshold={threshold}). "
                  f"Ctrl-C to stop.")
            while True:
                d = await listener.wait_for_detection()
                print(f"\n[WAKE] Detected {d.name}! (confidence={d.confidence:.2f})")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nstopped.")


# ----------------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(
        description="Test the hey_cozy wake-word model on mic or wav files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mic", action="store_true", help="listen live")
    parser.add_argument("--wav", nargs="*", type=Path, default=[],
                        help="score WAV file(s) instead")
    parser.add_argument("--calibrate", type=float, default=0, metavar="SECONDS",
                        help="record SECONDS via the live audio path, save "
                             "work/calib_hey_cozy.wav and score it per second")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.37,
                        help="wake trigger threshold (0-1, default 0.37)")
    args = parser.parse_args()

    print_meta(args.model, args.threshold)

    if args.wav:
        run_wav(args.wav, args.threshold)
    elif args.calibrate:
        run_calibrate(args.calibrate, args.threshold)
    elif args.mic:
        run_mic(args.threshold)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
