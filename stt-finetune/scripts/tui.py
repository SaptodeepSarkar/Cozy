#!/usr/bin/env python3
"""
Cozy STT — Prime-Agent-style live TUI for testing the finetuned model.

Layout:
  ┌──────────────────────────────────────────────────────────────┐
  │ 🎤 COZY STT  • engine: hf  • uptime 00:42  • verdicts: 7    │
  │ ● LISTENING  2.3s              [ct2] 178ms                  │
  │ ▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░ -18 dB  ⚠ clipping               │
  │  ╭▁▂▃▅▇▅▃▂▁▂▃▅▇▆▄▂▁▂▃▅▇▅▃▂▁╮  waveform                  │
  │ ┌─ LAST VERDICT ─────────────────────────────────────────┐ │
  │ │ Hey Cozy, what time is it right now?                  │ │
  │ └────────────────────────────────────────────────────────┘ │
  │ HISTORY                                                    │
  │ #  eng  ms   transcript                                    │
  │ 7  hf   178  Hey Cozy, what time is it right now?         │
  │ ...                                                        │
  └──────────────────────────────────────────────────────────────┘
  Ctrl+C to quit • --engine hf • --max-len 60s
"""
import argparse
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import Engines  # noqa: E402

# ---------- shared state (worker thread writes, UI thread reads) ----
class State:
    def __init__(self):
        self.status = "loading"          # loading / ready / listening / transcribing
        self.utter_s = 0.0
        self.t_start = time.time()
        self.engine_pref = "hf"
        self.last_eng = "-"
        self.last_ms = 0
        self.last_text = ""
        self.peak = 0
        self.clip_warn = False
        self.history = deque(maxlen=18)  # (eng, ms, text, ts)
        self.count = 0
        self.total_ms = 0
        self.wave = deque(maxlen=60)     # recent RMS values for waveform
        self.wave.append(0.0)

    @property
    def avg_ms(self):
        return self.total_ms / max(1, self.count)

    @property
    def uptime_s(self):
        return time.time() - self.t_start


S = State()


# ---------- visual primitives -----------------------------------------
def level_color(rms):
    if rms < 1500: return "green"
    if rms < 4500: return "yellow"
    if rms < 7000: return "dark_orange"
    return "bold red"


def vu_meter(rms, width=24):
    n = min(width, int((min(rms, 9000) / 9000) ** 0.5 * width))
    bar = "▓" * n + "░" * (width - n)
    db = 20 * math.log10(max(rms, 1) / 32768.0)
    return f"[{level_color(rms)}]{bar}[/{level_color(rms)}] {db:+.0f} dB"


def waveform(wave, height=6, width=42):
    """Render the recent RMS window as ASCII bars."""
    if not wave:
        return ""
    cols = list(wave)[-width:]
    # normalize to height
    peak = max(cols) or 1.0
    rows = []
    for h in range(height - 1, -1, -1):
        line = []
        for v in cols:
            norm = v / peak
            if norm >= h / height:
                line.append("█")
            else:
                line.append(" ")
        rows.append("".join(line))
    return "\n".join(rows)


def pulse_dot(status):
    colors = {
        "loading": "dim",
        "ready": "bold green",
        "listening": "bold yellow",
        "transcribing": "bold cyan",
    }
    return f"[{colors.get(status, 'white')}]●[/{colors.get(status, 'white')}]"


# ---------- main render ----------------------------------------------
def render():
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    s = S
    mins, secs = divmod(int(s.uptime_s), 60)
    header = Text()
    header.append("🎤 COZY STT  ", style="bold white")
    header.append(f"• engine: {s.engine_pref}", style="dim")
    header.append(f"  • uptime {mins:02d}:{secs:02d}", style="dim")
    header.append(f"  • verdicts: {s.count}", style="bold cyan")
    if s.count:
        header.append(f"  • avg {s.avg_ms:.0f}ms", style="dim")

    # status + timer
    status_line = Text()
    status_line.append_text(Text.from_markup(pulse_dot(s.status)))
    status_line.append(f" {s.status.upper()}", style="bold")
    if s.status == "listening":
        status_line.append(f"  {s.utter_s:0.1f}s", style="yellow")
    elif s.status == "transcribing":
        status_line.append(" …", style="cyan")
    if s.last_eng != "-":
        eng_color = "green" if s.last_eng == "ct2" else "cyan"
        status_line.append(
            f"    last: [{eng_color}]{s.last_eng}[/{eng_color}] {s.last_ms}ms",
            style="dim",
        )

    # VU meter
    clip = Text()
    if s.clip_warn:
        clip.append("  ⚠ CLIPPING — lower mic gain", style="bold red")
    vu = Text.from_markup(vu_meter(s.peak)) + clip

    # waveform
    wave_panel = Panel(
        waveform(list(s.wave)),
        title="waveform",
        title_align="left",
        border_style="dim",
        width=44,
        height=8,
    )

    # big last verdict
    last_text = s.last_text or "— waiting for first verdict —"
    big = Panel(
        Text(last_text, style="bold", justify="center"),
        title="LAST VERDICT",
        border_style="cyan" if s.last_eng == "hf" else "green",
        padding=(1, 2),
    )

    # history
    hist = Table(expand=True, box=None, pad_edge=False, show_header=True)
    hist.add_column("#", width=3, style="dim")
    hist.add_column("eng", width=5)
    hist.add_column("ms", width=5, justify="right")
    hist.add_column("transcript", overflow="fold")
    for i, (eng, ms, txt, ts) in enumerate(reversed(s.history)):
        h, m = divmod(int(time.time() - ts), 60)
        eng_styled = (f"[green]{eng}[/green]" if eng == "ct2"
                      else f"[cyan]{eng}[/cyan]")
        hist.add_row(
            str(len(s.history) - i),
            eng_styled,
            f"{ms:.0f}",
            txt,
        )

    footer = Text()
    footer.append("Ctrl+C", style="bold")
    footer.append(" quit", style="dim")
    footer.append("  •  ", style="dim")
    footer.append(f"--engine {s.engine_pref}", style="dim")
    footer.append("  •  --max-len 60s", style="dim")

    return Group(
        Panel(header, border_style="bright_blue"),
        Panel(Group(status_line, vu, wave_panel),
              border_style="bright_blue"),
        big,
        Panel(hist, title="HISTORY", border_style="bright_blue"),
        footer,
    )


# ---------- audio thread ---------------------------------------------
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
        S.wave.append(rms)
        S.peak = max(int(np.abs(pcm).max()), 0)
        if S.peak >= 32000:
            S.clip_warn = True
        else:
            S.clip_warn = False

        chunk = pcm.astype(np.float32) / 32768.0
        event = vad(chunk)

        if not speaking:
            preroll.append(chunk)
            if len(preroll) > 12:
                preroll.pop(0)
            if event and "start" in event:
                speaking, speech_buf, preroll = True, list(preroll), []
                t_start = time.time()
                S.status = "listening"
        else:
            speech_buf.append(chunk)
            S.utter_s = time.time() - t_start
            done = (event and "end" in event) or S.utter_s > max_len
            if done:
                audio = np.concatenate(speech_buf) if speech_buf else None
                speech_buf, speaking = [], False
                S.status = "transcribing"
                if audio is not None and len(audio) > 4800:
                    t0 = time.time()
                    text, used = engines.transcribe_array(audio, engine)
                    ms = (time.time() - t0) * 1000
                    S.last_eng, S.last_ms, S.last_text = used, int(ms), text
                    S.history.append((used, ms, text or "(silence)", time.time()))
                    S.count += 1
                    S.total_ms += ms
                S.status = "ready"


# ---------- main -----------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["auto", "ct2", "hf"], default="hf",
                    help="default: hf (Hinglish-aware). Use auto/ct2 for fast English.")
    ap.add_argument("--max-len", type=float, default=60.0,
                    help="force-flush an utterance after this many seconds")
    ap.add_argument("--simulate",
                    help="feed a wav through the same pipeline (no mic) and exit "
                         "after first verdict — for quick smoke tests")
    args = ap.parse_args()

    S.engine_pref = args.engine

    from rich.live import Live
    import numpy as np

    # Load engines FIRST so the user doesn't wait during first utterance
    print("[tui] loading engines once ...", file=sys.stderr, flush=True)
    engines = Engines()
    engines.warmup(args.engine)
    S.status = "ready"

    q = queue.Queue()

    if args.simulate:
        import librosa
        from silero_vad import VADIterator, load_silero_vad

        audio, _ = librosa.load(args.simulate, sr=16000, mono=True)
        audio = np.concatenate([audio, np.zeros(12000, dtype=np.float32)])

        def feeder():
            for i in range(0, len(audio) - 511, 512):
                q.put((audio[i:i + 512] * 32767).astype(np.int16).tobytes())

        threading.Thread(target=feeder, daemon=True).start()
        worker = threading.Thread(target=audio_worker,
                                  args=(q, engines, args.engine, args.max_len),
                                  daemon=True)
        worker.start()
    else:
        import sounddevice as sd
        # open the mic stream in a context manager — it actually starts
        # when the `with` block is entered
        print(f"[tui] opening mic @ 16 kHz, blocksize 512 ...", file=sys.stderr, flush=True)
        try:
            stream = sd.RawInputStream(samplerate=16000, blocksize=512,
                                       dtype="int16", channels=1,
                                       callback=lambda i, f, t, s:
                                       q.put(bytes(i)))
        except Exception as e:
            sys.exit(f"[tui] could not open mic: {e}\n"
                     "    check `arecord -l` and PulseAudio source selection")
        with stream:
            worker = threading.Thread(target=audio_worker,
                                      args=(q, engines, args.engine,
                                            args.max_len), daemon=True)
            worker.start()
            try:
                with Live(render(), refresh_per_second=12, screen=False) as live:
                    while True:
                        time.sleep(0.08)
                        live.update(render())
                        if args.simulate and S.count >= 1:
                            time.sleep(1.0)
                            break
            except KeyboardInterrupt:
                pass
            print("\n[tui] bye", file=sys.stderr, flush=True)
            return

    # simulate mode (no `with` for the mic)
    try:
        with Live(render(), refresh_per_second=12, screen=False) as live:
            while True:
                time.sleep(0.08)
                live.update(render())
                if args.simulate and S.count >= 1:
                    time.sleep(1.0)
                    break
    except KeyboardInterrupt:
        pass
    print("\n[tui] bye", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
