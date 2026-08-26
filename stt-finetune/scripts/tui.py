#!/usr/bin/env python
"""Cozy STT live TUI — Silero-VAD streaming transcription with a rich UI.

    .venv/bin/python scripts/tui.py                 # mic stream, auto engine
    .venv/bin/python scripts/tui.py --engine hf     # verbatim Hinglish verdicts
    .venv/bin/python scripts/tui.py --simulate recordings/session_1/000.wav

Layout: status header + live level meter + verdict history table.
Ctrl+C quits. Models load once before listening starts.
"""
import argparse
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import Engines  # noqa: E402

STATE = {
    "status": "loading",
    "level": 0.0,
    "peak": 0,
    "clip": False,
    "utter_s": 0.0,
    "last": "",
    "engine": "-",
    "latency_ms": 0,
    "history": deque(maxlen=14),
    "count": 0,
}


def level_bar(rms, width=28):
    blocks = " ▁▂▃▄▅▆▇█"
    n = min(width, int((min(rms, 8000) / 8000) ** 0.6 * width))
    bar = "█" * n + "░" * (width - n)
    color = "green" if rms < 4000 else ("yellow" if rms < 7000 else "red")
    return f"[{color}]{bar}[/{color}]"


def render():
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    s = STATE
    status_color = {"listening": "yellow", "transcribing": "cyan",
                    "ready": "green", "loading": "dim"}.get(s["status"], "white")
    head = Table.grid(padding=(0, 2))
    head.add_column(justify="left")
    head.add_column(justify="right")
    status_line = Text()
    status_line.append("● ", style=status_color)
    status_line.append(s["status"].upper(), style=f"bold {status_color}")
    if s["status"] == "listening":
        status_line.append(f"  {s['utter_s']:0.1f}s", style="yellow")
    head.add_row(status_line,
                 Text(f"engine={s['engine']}  verdicts={s['count']}",
                      style="dim"))
    head.add_row(Text(f"level {level_bar(s['level'])}  peak={s['peak']}"),
                 Text(("⚠ CLIPPING — lower mic gain" if s["clip"]
                       else f"latency {s['latency_ms']}ms"),
                      style="bold red" if s["clip"] else "dim"))

    hist = Table(expand=True, box=None, pad_edge=False)
    hist.add_column("#", width=4, style="dim")
    hist.add_column("eng", width=5)
    hist.add_column("ms", width=6, justify="right")
    hist.add_column("transcript", overflow="fold")
    for i, (eng, ms, txt) in enumerate(reversed(s["history"])):
        hist.add_row(str(len(s["history"]) - i),
                     "[cyan]hf[/cyan]" if eng == "hf" else "[green]ct2[/green]",
                     f"{ms:.0f}", txt)

    return Group(
        Panel(head, title="[b]🎤 Cozy STT Live[/b] — Silero VAD stream",
              subtitle="Ctrl+C to quit", subtitle_align="right"),
        Panel(hist, title="verdict history"),
    )


def audio_worker(q, engines, engine, max_len):
    import numpy as np
    from silero_vad import VADIterator, load_silero_vad

    vad = VADIterator(load_silero_vad(), sampling_rate=16000, threshold=0.45,
                      min_silence_duration_ms=650, speech_pad_ms=120)
    BLOCK = 512
    preroll, speech_buf = [], []
    speaking = False
    t_start = 0.0

    while True:
        pcm = np.frombuffer(q.get(), dtype=np.int16)
        rms = float(np.abs(pcm.astype(np.float32)).mean())
        STATE["level"], STATE["peak"] = rms, max(int(np.abs(pcm).max()), 0)
        if STATE["peak"] >= 32000:
            STATE["clip"] = True
        chunk = pcm.astype(np.float32) / 32768.0
        event = vad(chunk)

        if not speaking:
            preroll.append(chunk)
            if len(preroll) > 12:
                preroll.pop(0)
            if event and "start" in event:
                speaking, speech_buf, preroll = True, list(preroll), []
                t_start = time.time()
                STATE["status"] = "listening"
        else:
            speech_buf.append(chunk)
            STATE["utter_s"] = time.time() - t_start
            done = (event and "end" in event) or STATE["utter_s"] > max_len
            if done:
                audio = np.concatenate(speech_buf) if speech_buf else None
                speech_buf, speaking = [], False
                STATE["status"], STATE["utter_s"] = "transcribing", 0.0
                if audio is not None and len(audio) > 4800:
                    t0 = time.time()
                    text, used = engines.transcribe_array(audio, engine)
                    ms = (time.time() - t0) * 1000
                    STATE["last"], STATE["engine"] = text, used
                    STATE["latency_ms"] = int(ms)
                    STATE["history"].append((used, ms, text or "(silence)"))
                    STATE["count"] += 1
                STATE["status"] = "ready"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=["auto", "ct2", "hf"], default="auto")
    ap.add_argument("--max-len", type=float, default=60.0)
    ap.add_argument("--simulate", help="feed a wav through the same pipeline "
                                       "(no mic) and exit after first verdict")
    args = ap.parse_args()

    from rich.live import Live
    print("[tui] loading engines once ...")
    engines = Engines()
    engines.warmup(args.engine)
    STATE["status"] = "ready"

    q = queue.Queue()
    if args.simulate:
        import librosa
        import numpy as np
        from silero_vad import VADIterator, load_silero_vad

        audio, _ = librosa.load(args.simulate, sr=16000, mono=True)
        audio = np.concatenate([audio, np.zeros(12000, dtype=np.float32)])
        vad = VADIterator(load_silero_vad(), sampling_rate=16000,
                          threshold=0.45, min_silence_duration_ms=650,
                          speech_pad_ms=120)
        threading.Thread(target=lambda: [q.put(b) for b in
                         [(audio[i:i+512] * 32767).astype(np.int16).tobytes()
                          for i in range(0, len(audio) - 511, 512)]],
                         daemon=True).start()
    else:
        import sounddevice as sd
        # sounddevice starts on enter; the "with" context opens the stream
        sd.RawInputStream(samplerate=16000, blocksize=512, dtype="int16",
                          channels=1, callback=lambda i, f, t, s:
                          q.put(bytes(i)))

    worker = threading.Thread(target=audio_worker,
                              args=(q, engines, args.engine, args.max_len),
                              daemon=True)
    worker.start()

    try:
        with Live(render(), refresh_per_second=12, screen=False) as live:
            while True:
                time.sleep(0.08)
                live.update(render())
                if args.simulate and STATE["count"] >= 1:
                    time.sleep(1.0)
                    break
    except KeyboardInterrupt:
        pass
    print("\n[tui] bye")


if __name__ == "__main__":
    main()
