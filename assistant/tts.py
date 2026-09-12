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
import hashlib
import json
import sys
import threading
import warnings
import re
from pathlib import Path

DEFAULT_VOICE = "af_heart"
DEFAULT_SPEED = 1.0
DEFAULT_LANG = "a"  # American English

_CFG_PATH = Path(__file__).resolve().parent / "voice.cfg"
_CACHE_DIR = Path("/tmp/cozy_tts_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_pipeline = None
_pipeline_error = None
_pipeline_lock = threading.Lock()
_worker_started = False
_worker_thread = None
_speaking = threading.Event()

import queue as _queue
_speak_queue: "_queue.Queue[str | None]" = _queue.Queue(maxsize=8)

warnings.filterwarnings("ignore", message=r".*dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=r".*weight_norm.*deprecated.*")

def _runtime_error(message: str) -> None:
    if os.environ.get("COZY_TUI_MODE") != "node":
        return
    try:
        sys.stdout.write(json.dumps({"kind": "error", "msg": message}) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def _runtime_event(kind: str, **fields) -> None:
    if os.environ.get("COZY_TUI_MODE") != "node":
        return
    try:
        sys.stdout.write(json.dumps({"kind": kind, **fields}) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


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
    global _pipeline, _pipeline_error
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        # Suppress noisy HF progress bars
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        try:
            import spacy
            if not spacy.util.is_package("en_core_web_sm"):
                raise RuntimeError(
                    "Kokoro requires spaCy model en_core_web_sm; run bash setup.sh")
            from kokoro import KPipeline
            cfg = _load_cfg()
            print(f"[tts] loading Kokoro pipeline (voice={cfg['voice']}, lang={cfg['lang']})...", flush=True)
            _pipeline = KPipeline(
                lang_code=cfg["lang"],
                repo_id="hexgrad/Kokoro-82M",
                device="cpu",
            )
            # Loading the pipeline alone does not load/validate the selected
            # voice. Do that during startup so the first reply cannot discover
            # a missing or corrupt voice file.
            _pipeline.load_voice(str(cfg["voice"]))
            _pipeline_error = None
            print(f"[tts] Kokoro ready (voice={cfg['voice']})", flush=True)
            return _pipeline
        except Exception as exc:
            print(f"[tts] Kokoro load failed: {exc}", flush=True)
            _runtime_error(f"TTS unavailable: {exc}")
            print("[tts] falling back to silent mode", flush=True)
            _pipeline = None
            _pipeline_error = str(exc)
            return None


def is_available() -> bool:
    """Return True if Kokoro can be loaded."""
    try:
        from kokoro import KPipeline  # noqa: F401
        return True
    except ImportError:
        return False


def warmup() -> bool:
    """Load Kokoro during the startup screen instead of on the first reply."""
    return _get_pipeline() is not None


def initialization_error() -> str:
    """Return the last Kokoro initialization error for startup diagnostics."""
    return str(_pipeline_error or "")


def _synthesize(text: str) -> "tuple[object, int] | None":
    """Synthesize text. Returns (samples, sample_rate) or None on failure."""
    p = _get_pipeline()
    if p is None:
        return None
    cfg = _load_cfg()
    voice = cfg["voice"]
    speed = float(cfg["speed"])
    try:
        chunks = list(_synthesize_chunks(text, pipeline=p, voice=voice, speed=speed))
        if chunks:
            import numpy as np
            return np.concatenate([np.asarray(chunk).reshape(-1) for chunk in chunks]), 24000
    except Exception as exc:
        print(f"[tts] synth error: {exc}")
    return None


def _synthesize_chunks(text: str, *, pipeline=None, voice=None, speed=None):
    """Yield Kokoro audio as soon as each generated segment is available."""
    p = pipeline or _get_pipeline()
    if p is None:
        return
    cfg = _load_cfg()
    for _gs, _ps, audio in p(
        text, voice=voice or cfg["voice"], speed=speed or float(cfg["speed"]),
    ):
        if audio is not None:
            yield audio


def _text_chunks(text: str, max_chars: int = 180) -> list[str]:
    """Split replies into natural speech chunks so the first sentence starts fast."""
    pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text.strip()) if piece.strip()]
    chunks = []
    for piece in pieces:
        while len(piece) > max_chars:
            cut = piece.rfind(" ", 0, max_chars)
            cut = cut if cut > 40 else max_chars
            chunks.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            chunks.append(piece)
    return chunks or [text.strip()]


def _cache_path(text: str) -> Path:
    safe = "".join(c if c.isalnum() else "_" for c in text)[:40]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return _CACHE_DIR / f"{safe}_{digest}.wav"


def _play(samples, sr: int) -> None:
    try:
        import soundfile as sf
        from audio_io import play
        if samples is None:
            return
        # If cached, load from file
        if isinstance(samples, (str, Path)):
            data, file_sr = sf.read(str(samples), dtype="float32")
            sr = file_sr
        else:
            data = samples
        play(data, sr)
    except Exception as exc:
        print(f"[tts] play failed: {exc}")


def _ensure_worker():
    global _worker_started, _worker_thread
    if _worker_started:
        return
    def loop():
        while True:
            text = _speak_queue.get()
            if text is None:
                _speak_queue.task_done()
                break
            try:
                _speaking.set()
                _runtime_event("tts_start", text=text)
                # Cache hit?
                cache_path = _cache_path(text)
                if cache_path.exists():
                    _play(str(cache_path), 24000)
                else:
                    cfg = _load_cfg()
                    generated = []
                    for audio in _synthesize_chunks(text, voice=cfg["voice"], speed=float(cfg["speed"])):
                        import numpy as np
                        samples = np.asarray(audio).reshape(-1)
                        generated.append(samples)
                        # Playback starts on the first Kokoro segment instead
                        # of waiting for the full response to synthesize.
                        _play(samples, 24000)
                    if generated:
                        try:
                            import soundfile as sf
                            sf.write(str(cache_path), np.concatenate(generated), 24000)
                        except Exception:
                            pass
            except Exception as exc:
                print(f"[tts] worker error: {exc}")
            finally:
                _speaking.clear()
                _runtime_event("tts_done")
                _speak_queue.task_done()
    t = threading.Thread(target=loop, daemon=True, name="cozy-tts")
    t.start()
    _worker_thread = t
    _worker_started = True


def shutdown(timeout: float = 5.0) -> bool:
    """Stop the speech worker before native libraries are finalized."""
    global _pipeline, _worker_started, _worker_thread
    if _worker_started:
        # Discard queued speech during shutdown so the sentinel is guaranteed
        # to fit and exit is not delayed by several old replies.
        while True:
            try:
                _speak_queue.get_nowait()
                _speak_queue.task_done()
            except _queue.Empty:
                break
        _speak_queue.put_nowait(None)
        if _worker_thread is not None:
            _worker_thread.join(timeout=timeout)
            if _worker_thread.is_alive():
                return False
    _worker_thread = None
    _worker_started = False
    _pipeline = None
    return True


def speak(text: str, blocking: bool = False) -> None:
    """Speak ``text``. Non-blocking by default; runs on a background thread."""
    if not text or not text.strip():
        return
    if not is_available():
        return
    _ensure_worker()
    for piece in _text_chunks(text):
        try:
            _speak_queue.put_nowait(piece)
        except _queue.Full:
            _runtime_error("TTS queue is full; dropping reply")
            break
    if blocking:
        _speak_queue.join()


def is_speaking() -> bool:
    # Queue accounting closes the small gap between speak() enqueueing a
    # reply and the worker setting _speaking. Without it, the mic can capture
    # the first syllable of Cozy's own response.
    return _speaking.is_set() or _speak_queue.unfinished_tasks > 0


def say(text: str) -> None:
    """Convenience: block until text is finished speaking."""
    speak(text, blocking=True)


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "Cozy is online."
    print(f"[tts] speaking: {text}")
    say(text)
    print("[tts] done")
