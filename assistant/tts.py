"""Cozy TTS - Kokoro-82M wrapper with a non-blocking speak().

Kokoro is a small (~80 MB) high-quality neural TTS model that runs
fully on CPU. The default voice is "af_heart" (warm female US English).
The model auto-downloads from hexgrad/Kokoro-82M on first use.

Design:
  - speak() is non-blocking; runs on a background thread.
  - The pipeline is lazy-loaded on first call (saves ~2 GB RAM if you
    never speak).
  - Common affirmations are cached as WAVs for instant replay.
  - If Kokoro fails to load (e.g. disk full, no network), Cozy falls
    back to a quiet no-op (the runtime never crashes on TTS issues).

Config: assistant/voice.cfg
  voice = af_heart       # any Kokoro voice id
  speed = 1.0            # 0.5 - 2.0
  lang = a               # a = American English, b = British English
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

DEFAULT_VOICE = "af_heart"
DEFAULT_SPEED = 1.0
DEFAULT_LANG = "a"  # American English

_CFG_PATH = Path(__file__).resolve().parent / "voice.cfg"
_CACHE_DIR = Path("/tmp/cozy_tts_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_pipeline = None
_pipeline_lock = threading.Lock()
_worker_started = False

import queue as _queue
_speak_queue: "_queue.Queue[str | None]" = _queue.Queue()


def _load_cfg():
    cfg = {
        "voice": DEFAULT_VOICE,
        "speed": DEFAULT_SPEED,
        "lang": DEFAULT_LANG,
    }
    if _CFG_PATH.exists():
        for line in _CFG_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            try:
                v = float(v)
            except ValueError:
                pass
            cfg[k] = v
    return cfg


def _get_pipeline():
    """Lazy-load the Kokoro pipeline. Cached after first call."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        # Suppress noisy HF progress bars
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        try:
            from kokoro import KPipeline
            cfg = _load_cfg()
            print(f"[tts] loading Kokoro pipeline (voice={cfg['voice']}, lang={cfg['lang']})...", flush=True)
            _pipeline = KPipeline(lang_code=cfg["lang"])
            print(f"[tts] Kokoro ready (voice={cfg['voice']})", flush=True)
            return _pipeline
        except Exception as exc:
            print(f"[tts] Kokoro load failed: {exc}", flush=True)
            print("[tts] falling back to silent mode", flush=True)
            _pipeline = None
            return None


def is_available() -> bool:
    """Return True if Kokoro can be loaded."""
    try:
        from kokoro import KPipeline  # noqa: F401
        return True
    except ImportError:
        return False


def _synthesize(text: str) -> "tuple[object, int] | None":
    """Synthesize text. Returns (samples, sample_rate) or None on failure."""
    p = _get_pipeline()
    if p is None:
        return None
    cfg = _load_cfg()
    voice = cfg["voice"]
    speed = float(cfg["speed"])
    try:
        # KPipeline returns (graphemes, phonemes, audio) tuples.
        for _gs, _ps, audio in p(text, voice=voice, speed=speed):
            if audio is not None:
                return audio, 24000
    except Exception as exc:
        print(f"[tts] synth error: {exc}")
    return None


def _play(samples, sr: int) -> None:
    try:
        import sounddevice as sd
        import soundfile as sf
        if samples is None:
            return
        # If cached, load from file
        if isinstance(samples, (str, Path)):
            data, file_sr = sf.read(str(samples), dtype="float32")
            sr = file_sr
        else:
            data = samples
        sd.play(data, samplerate=sr, blocking=True)
    except Exception as exc:
        print(f"[tts] play failed: {exc}")


def _ensure_worker():
    global _worker_started
    if _worker_started:
        return
    def loop():
        while True:
            text = _speak_queue.get()
            if text is None:
                break
            try:
                # Cache hit?
                safe = "".join(c if c.isalnum() else "_" for c in text)[:64]
                cache_path = _CACHE_DIR / f"{safe}.wav"
                if cache_path.exists():
                    _play(str(cache_path), 24000)
                else:
                    out = _synthesize(text)
                    if out is not None:
                        samples, sr = out
                        # Save to cache for next time
                        try:
                            import soundfile as sf
                            sf.write(str(cache_path), samples, sr)
                        except Exception:
                            pass
                        _play(samples, sr)
            except Exception as exc:
                print(f"[tts] worker error: {exc}")
            finally:
                _speak_queue.task_done()
    t = threading.Thread(target=loop, daemon=True, name="cozy-tts")
    t.start()
    _worker_started = True


def speak(text: str, blocking: bool = False) -> None:
    """Speak ``text``. Non-blocking by default; runs on a background thread."""
    if not text or not text.strip():
        return
    if not is_available():
        return
    _ensure_worker()
    _speak_queue.put(text)
    if blocking:
        _speak_queue.join()


def say(text: str) -> None:
    """Convenience: block until text is finished speaking."""
    speak(text, blocking=True)


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "Cozy is online."
    print(f"[tts] speaking: {text}")
    say(text)
    print("[tts] done")
