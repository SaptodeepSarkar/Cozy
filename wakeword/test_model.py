#!/usr/bin/env python3
"""Test the trained CozyNet v2 wake-word model on your microphone or WAV files.

This is the new lightweight model (cozynet_v2.onnx) trained on YOUR real voice
plus 32 user "cozy" recordings from data/cozy.

Inference flow (matches training exactly):
  1. Read 16kHz int16 mono PCM
  2. Compute 32-bin mel spectrogram via openWakeWord's AudioFeatures
  3. Per-bin normalize using saved MEAN/STD
  4. Slide a 1.0s window (97 mel frames) hop=19 (0.2s) across the clip
  5. Sigmoid the model's logit output -> probability
  6. Peak score is the wake-word score for the clip

Two safety guards in front of the model:
  - Energy gate: window RMS must be >= ENERGY_RMS_MIN (default 500).
    This silences the model on pure silence / quiet background noise where
    the BN-collapsed score would otherwise leak through (~0.3-0.6).
  - Default threshold 0.7: real "Hey Cozy" scores 0.85-0.99 in practice;
    silence / low-noise scored 0.0 with the energy gate active.

Usage:
    python test_model.py --mic                    # live listening (threshold 0.7)
    python test_model.py --mic --threshold 0.5    # custom threshold
    python test_model.py --wav some_clip.wav      # score a single wav
    python test_model.py --self-test              # score held-out val set
    python test_model.py --calibrate 8            # record 8s and score per-second
    python test_model.py --mic --debug            # live score bar
    python test_model.py --no-energy-gate         # disable RMS gate (debug)
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "cozynet_v2.onnx"
DEFAULT_META = HERE / "models" / "cozynet_v2_meta.json"
CHUNK = 1280  # 80 ms - the audio capture granularity
WIN_FRAMES = 97  # ~1.0s of mel frames
HOP_FRAMES = 19  # ~0.2s of mel frames
SR = 16000

# Audio energy gate: if a window's RMS is below this, skip inference
# and report 0.0. This prevents the model from firing on pure silence
# (where BN collapsed score leaks through as ~0.3-0.6).
# Real speech RMS at 16kHz int16 is typically 1000-5000; ambient room
# noise is 100-500. 500 is a safe default.
ENERGY_RMS_MIN = 500

# Per-mel-frame energy floor: a 1s window must have at least this
# fraction of frames with energy above the floor. Keeps the model
# honest on partial-silence windows.
MEL_ACTIVE_MIN = 0.20  # at least 20% of frames must have audio

_af = None  # lazy-init AudioFeatures


def _audio_features():
    global _af
    if _af is None:
        from openwakeword.model import AudioFeatures
        _af = AudioFeatures()
    return _af


def _stable_mel(pcm_int16: np.ndarray) -> np.ndarray:
    """Compute a mel spectrogram of a fixed length, regardless of input.

    openWakeWord's mel extractor returns one frame fewer than the
    expected count for some input lengths (off-by-one). We pad the
    audio to 24000 samples (1.5s) which gives 147 frames reliably,
    then trim/pad to a known shape.
    """
    af = _audio_features()
    if pcm_int16.ndim > 1:
        pcm_int16 = pcm_int16.mean(axis=1)
    pcm_int16 = pcm_int16.astype(np.int16)
    target = 24000  # 1.5s -> 147 mel frames
    if len(pcm_int16) < target:
        p = np.pad(pcm_int16, (0, target - len(pcm_int16)))
    else:
        p = pcm_int16[:target]
    out = af._get_melspectrogram_batch(p[None, :], batch_size=8, ncpu=4)
    return np.asarray(out)[0].astype(np.float32)


def _window_rms(pcm_int16: np.ndarray) -> float:
    return float(np.sqrt(np.mean(pcm_int16.astype(np.float32) ** 2)))


def _mel_active_fraction(m: np.ndarray) -> float:
    """Fraction of frames whose mean energy > 1.5x silence floor.

    Silence mel is constant ~-8.0; speech mel has variance. Use 0.5
    as a robust threshold: any frame above 0.5 is 'active'.
    """
    frame_means = m.mean(axis=1)
    return float((frame_means > 0.5).mean())


class CozyDetector:
    def __init__(self, model_path: Path, meta_path: Path, threshold: float,
                 energy_gate: bool = True):
        if not model_path.exists():
            raise SystemExit(
                f"Model not found: {model_path}\n"
                f"Train it first: python train_cozynet_v2.py"
            )
        self.meta = json.loads(meta_path.read_text())
        self.T = self.meta["time_frames"]
        self.BINS = self.meta["mel_bins"]
        self.MEAN = np.array(self.meta["mel_mean"], dtype=np.float32)
        self.STD = np.array(self.meta["mel_std"], dtype=np.float32)
        self.sess = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name
        self.threshold = float(threshold)
        self.energy_gate = bool(energy_gate)
        # rolling 1.0s buffer
        self.buf = np.zeros(SR, dtype=np.int16)
        self.buf_fill = 0

    def reset(self) -> None:
        self.buf[:] = 0
        self.buf_fill = 0

    def _score_mel(self, m: np.ndarray) -> float:
        """Run sliding-window inference on a mel tensor. Returns peak score."""
        if m.shape[0] < self.T:
            m = np.pad(m, ((0, self.T - m.shape[0]), (0, 0)))
        best = 0.0
        for s0 in range(0, m.shape[0] - self.T + 1, HOP_FRAMES):
            win = m[s0:s0 + self.T]
            if win.shape[0] < self.T:
                win = np.pad(win, ((0, self.T - win.shape[0]), (0, 0)))
            win = (win - self.MEAN) / self.STD
            xb = win[None, None].astype(np.float32)
            logit = float(self.sess.run([self.out], {self.inp: xb})[0].reshape(-1)[0])
            score = 1.0 / (1.0 + np.exp(-logit))
            if score > best:
                best = score
        return best

    def _gate(self, pcm_int16: np.ndarray, m: np.ndarray) -> bool:
        """Return True if the window passes the energy gate."""
        if not self.energy_gate:
            return True
        rms = _window_rms(pcm_int16)
        if rms < ENERGY_RMS_MIN:
            return False
        # also require a few active mel frames
        if _mel_active_fraction(m) < MEL_ACTIVE_MIN:
            return False
        return True

    def feed_chunk(self, chunk_int16: np.ndarray) -> float:
        """Feed an 80ms audio chunk (int16). Returns wake score."""
        chunk_int16 = chunk_int16.astype(np.int16)
        self.buf = np.roll(self.buf, -len(chunk_int16))
        self.buf[-len(chunk_int16):] = chunk_int16
        self.buf_fill = min(SR, self.buf_fill + len(chunk_int16))
        if self.buf_fill < SR:
            return 0.0
        if not self._gate(self.buf, np.zeros((1, self.BINS), dtype=np.float32)):
            return 0.0
        m = _stable_mel(self.buf)
        if not self._gate(self.buf, m):
            return 0.0
        return self._score_mel(m)

    def score_wav(self, path: Path) -> tuple[float, list[tuple[int, float]]]:
        """Score a wav file. Returns (peak_score, [(time_ms, score), ...])."""
        pcm, sr = sf.read(str(path), dtype="int16")
        if sr != SR:
            from scipy.signal import resample_poly
            pcm = resample_poly(pcm.astype(np.int16), SR, sr).astype(np.int16)
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1).astype(np.int16)
        if len(pcm) < SR:
            pcm = np.pad(pcm, (0, SR - len(pcm)))
        return self.score_pcm(pcm)

    def score_pcm(self, pcm: np.ndarray) -> tuple[float, list[tuple[int, float]]]:
        """Score a raw int16 PCM array (>=1s). Returns peak + per-second timeline."""
        pcm = pcm.astype(np.int16)
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1).astype(np.int16)
        if len(pcm) < SR:
            pcm = np.pad(pcm, (0, SR - len(pcm)))
        timeline = []
        best = 0.0
        for start in range(0, len(pcm) - SR + 1, SR):
            seg = pcm[start:start + SR]
            m = _stable_mel(seg)
            gated = self._gate(seg, m)
            if gated:
                s = self._score_mel(m)
            else:
                s = 0.0
            t_sec = start / SR
            timeline.append((int(t_sec * 1000), s))
            if s > best:
                best = s
        return best, timeline


# ----------------------------------------------------------------- CLI modes

def run_self_test(det: CozyDetector) -> None:
    """Score the held-out val_pos and val_neg wavs (real voice only)."""
    val_pos = sorted((HERE / "work" / "val_pos").glob("*.wav"))
    val_neg = sorted((HERE / "work" / "val_neg").glob("*.wav"))
    print(f"\n=== VAL POS (real voice, held out) - {len(val_pos)} files ===")
    pos_scores = []
    for p in val_pos:
        s, _ = det.score_wav(p)
        pos_scores.append(s)
        print(f"  {p.name:<48} {s:.3f}  {'FIRE' if s >= det.threshold else 'silent'}")
    print(f"\n=== VAL NEG (real voice, held out) - {len(val_neg)} files ===")
    neg_scores = []
    for p in val_neg:
        s, _ = det.score_wav(p)
        neg_scores.append(s)
    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)
    print(f"  pos: min={pos_scores.min():.3f} median={np.median(pos_scores):.3f} max={pos_scores.max():.3f}")
    print(f"  neg: min={neg_scores.min():.3f} median={np.median(neg_scores):.3f} max={neg_scores.max():.3f}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        tpr = float((pos_scores >= thr).mean())
        fpr = float((neg_scores >= thr).mean())
        marker = "  <-- current default" if abs(thr - det.threshold) < 0.01 else ""
        print(f"  thr={thr}: TPR={tpr*100:5.1f}%  FPR={fpr*100:5.1f}%{marker}")


def run_wav(det: CozyDetector, paths: list[Path], verbose: bool) -> None:
    for wav in paths:
        s, timeline = det.score_wav(wav)
        verdict = "DETECTED \u2714" if s >= det.threshold else "clean"
        print(f"{wav.name:<48} peak={s:.3f}  [{verdict}]")
        if verbose:
            for t_ms, sc in timeline:
                bar = "#" * int(sc * 30)
                fired = " <-- FIRE" if sc >= det.threshold else ""
                print(f"  t={t_ms:>5}ms  {bar:<30} {sc:.3f}{fired}")


def run_calibrate(seconds: float, det: CozyDetector) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("Install mic deps: pip install sounddevice")
    secs = float(seconds)
    print(f"Recording {secs:.0f}s from your mic - say 'hey cozy' a few "
          f"times, then talk about anything else.")
    for tick in (3, 2, 1):
        print(tick, flush=True)
    pcm = sd.rec(int(secs * SR), samplerate=SR, channels=1, dtype="int16")
    sd.wait()
    out = HERE / "work" / "calib_v2.wav"
    out.parent.mkdir(exist_ok=True)
    sf.write(str(out), pcm, SR, subtype="PCM_16")
    peak = int(np.abs(pcm).max())
    print(f"saved {out} | mic peak {peak}/32767"
          + (" (LOW - speak louder / check mic)" if peak < 800 else "")
          + (" (CLIPPING - lower mic gain!)" if peak > 32000 else ""))
    pcm1 = pcm[:, 0] if pcm.ndim > 1 else pcm.reshape(-1)
    pcm1 = pcm1.astype(np.int16)
    s, timeline = det.score_pcm(pcm1)
    for t_ms, sc in timeline:
        bar = "#" * int(sc * 40)
        t = t_ms // 1000
        print(f"{t:02d}s {bar.ljust(40)} {sc:.3f}"
              + ("  <-- WAKE" if sc >= det.threshold else ""))
    print(f"\noverall peak: {s:.3f}")


def run_mic(det: CozyDetector, debug: bool) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("Install mic deps: pip install sounddevice")
    recent: collections.deque = collections.deque(maxlen=8)
    last_hit = 0.0

    def callback(indata, _frames, _time, _status):
        recent.append(indata.copy())

    print(f"\nListening for 'hey cozy' (threshold={det.threshold}, "
          f"energy_gate={det.energy_gate}). Ctrl-C to stop.\n")
    last_debug = 0.0
    with sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                        blocksize=CHUNK, callback=callback):
        idle_print = time.time()
        while True:
            if not recent:
                time.sleep(0.01)
                continue
            chunk = recent.popleft()[:, 0]
            score = det.feed_chunk(chunk)
            now = time.time()
            if score >= det.threshold and now - last_hit > 2.0:
                last_hit = now
                print(f"\n\U0001F7E2 COZY detected (score={score:.3f}) - "
                      f"assistant would wake now\n")
            elif debug and now - last_debug > 0.2:
                last_debug = now
                bar = "#" * int(min(score, 1.0) * 40)
                print(f"\r{bar:<40} {score:.3f}", end="", flush=True)
            elif not debug and now - idle_print > 8.0:
                idle_print = now
                rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
                print(f"  ... listening (peak score {score:.3f}, "
                      f"chunk rms {rms:.0f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the CozyNet v2 wake-word model on mic or wav files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mic", action="store_true", help="listen live")
    parser.add_argument("--wav", nargs="*", type=Path, default=[],
                        help="score WAV file(s) instead")
    parser.add_argument("--self-test", action="store_true",
                        help="score the held-out val_pos/val_neg wavs (real voice)")
    parser.add_argument("--calibrate", type=float, default=0, metavar="SECONDS",
                        help="record SECONDS via the live audio path, save "
                             "work/calib_v2.wav and score it per second")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="wake trigger threshold (0-1, default 0.7)")
    parser.add_argument("--debug", action="store_true",
                        help="show a live score bar while listening")
    parser.add_argument("--verbose", action="store_true",
                        help="show per-second scores for --wav mode")
    parser.add_argument("--no-energy-gate", action="store_true",
                        help="disable the RMS-based silence gate (debug)")
    parser.add_argument("--gate-rms", type=int, default=ENERGY_RMS_MIN,
                        help=f"minimum window RMS to run the model "
                             f"(default {ENERGY_RMS_MIN}). Lower = more "
                             f"sensitive to quiet audio, higher = stricter.")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"Model not found: {args.model}\n"
            f"Train it first: python train_cozynet_v2.py"
        )
    # override the gate threshold
    import test_model as _self
    _self.ENERGY_RMS_MIN = args.gate_rms
    det = CozyDetector(args.model, args.meta, args.threshold,
                       energy_gate=not args.no_energy_gate)
    print(f"loaded {args.model.name} | T={det.T} bins={det.BINS} "
          f"thr={det.threshold} gate={'on' if det.energy_gate else 'off'} "
          f"gate_rms={args.gate_rms} "
          f"| val_auc={det.meta.get('val_auc_real_voice', '?')}")

    if args.self_test:
        run_self_test(det)
    elif args.wav:
        run_wav(det, args.wav, args.verbose)
    elif args.calibrate:
        run_calibrate(args.calibrate, det)
    elif args.mic:
        run_mic(det, args.debug)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
