"""PipeWire-native audio I/O for Cozy.

The desktop PipeWire graph is accessed through pipewire-pulse (``parec`` and
``paplay``).  Cozy never opens an ALSA hardware device and never changes the
user's configured defaults.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable

import numpy as np


def _run_text(*args: str) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def pipewire_defaults() -> tuple[str, str]:
    """Return the current PipeWire/Pulse default ``(source, sink)`` names."""
    return (_run_text("pactl", "get-default-source"),
            _run_text("pactl", "get-default-sink"))


def _sources() -> list[str]:
    names = []
    for line in _run_text("pactl", "list", "short", "sources").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            names.append(fields[1])
    return names


def capture_source() -> str:
    """Choose a PipeWire source without disrupting Bluetooth playback.

    A Bluetooth headset cannot normally keep its high-quality A2DP playback
    profile while its microphone is active.  If both desktop defaults point
    at the same Bluetooth headset, use the existing RNNoise/built-in source
    for Cozy.  This keeps Spotify and other media on the default A2DP sink.
    Set ``COZY_ALLOW_BLUETOOTH_MIC=1`` to explicitly opt into headset mode.
    """
    source, sink = pipewire_defaults()
    if (os.environ.get("COZY_ALLOW_BLUETOOTH_MIC") == "1" or
            not source.startswith("bluez_input.") or
            not sink.startswith("bluez_output.")):
        return source or "@DEFAULT_SOURCE@"

    available = _sources()
    preferred = (
        "effect_output.cozy-rnnoise",
        "effect_output.rnnoise",
    )
    for name in preferred:
        if name in available:
            return name
    for name in available:
        if name.startswith("alsa_input.") and not name.endswith(".monitor"):
            return name
    return source or "@DEFAULT_SOURCE@"


def source_muted(source: str) -> bool | None:
    value = _run_text("pactl", "get-source-mute", source).lower()
    if value.endswith("yes"):
        return True
    if value.endswith("no"):
        return False
    return None


def playback_sink() -> str:
    """Return the current PipeWire default sink without modifying it."""
    _source, sink = pipewire_defaults()
    return sink or "@DEFAULT_SINK@"


class PipeWireInputStream:
    """Small ``sounddevice.InputStream``-like wrapper around ``parec``."""

    def __init__(self, *, samplerate: int, channels: int, dtype: str,
                 blocksize: int, callback: Callable):
        if dtype != "int16" or channels != 1:
            raise ValueError("Cozy capture requires mono int16 audio")
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.callback = callback
        self.source = capture_source()
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: str | None = None

    def __enter__(self):
        command = [
            "parec", f"--device={self.source}", "--raw", "--format=s16le",
            f"--rate={self.samplerate}", "--channels=1",
            f"--latency-msec={max(20, round(1000 * self.blocksize / self.samplerate))}",
        ]
        self._process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self._thread = threading.Thread(
            target=self._read_loop, name="cozy-pipewire-capture", daemon=True)
        self._thread.start()
        return self

    def _read_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        byte_count = self.blocksize * 2
        pending = b""
        try:
            while not self._stop.is_set():
                data = self._process.stdout.read(byte_count - len(pending))
                if not data:
                    if not self._stop.is_set():
                        self._error = "Microphone stream stopped; check the input device and restart Cozy"
                    break
                pending += data
                if len(pending) < byte_count:
                    continue
                pcm = np.frombuffer(pending, dtype="<i2").copy().reshape(-1, 1)
                pending = b""
                self.callback(pcm, len(pcm), None, None)
        except Exception as exc:
            self._error = f"Microphone capture failed: {exc}"

    def check(self) -> None:
        if self._error:
            raise RuntimeError(self._error)
        if self._process is not None and self._process.poll() is not None:
            raise RuntimeError("Microphone process exited; check the input device and restart Cozy")

    def __exit__(self, _exc_type, _exc, _tb):
        self._stop.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1)
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._process is not None:
            if self._process.stdout: self._process.stdout.close()
            if self._process.stderr: self._process.stderr.close()


def play(samples, samplerate: int) -> None:
    """Play samples through PipeWire's current default sink, synchronously."""
    import soundfile as sf

    temporary: Path | None = None
    if isinstance(samples, (str, Path)):
        audio_path = Path(samples)
    else:
        fd, name = tempfile.mkstemp(prefix="cozy_tts_", suffix=".wav")
        os.close(fd)
        temporary = Path(name)
        sf.write(str(temporary), samples, samplerate)
        audio_path = temporary
    try:
        # The previous timeout patch accidentally referenced names from the
        # TTS wrapper (`data`/`sr`) that do not exist in this function, so
        # every real playback failed before paplay was started.
        try:
            duration = float(sf.info(str(audio_path)).duration)
        except Exception:
            duration = 0.0
        result = subprocess.run(
            ["paplay", f"--device={playback_sink()}", str(audio_path)],
            capture_output=True, text=True, timeout=max(10.0, duration + 5.0))
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "PipeWire playback failed")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
