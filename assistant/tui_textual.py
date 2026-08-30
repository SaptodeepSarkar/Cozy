"""Cozy TUI v3 - product-grade single-focus UI per the user's spec.

Layout (no background, no panels, just text):
  +----------------------------------------------------------+
  |  |\__/,|   (`\\                         Cozy ui            |  cat top-left (animated)
  |  _.|o o  |_   ) )                       v0.1               |  title right
  |  -(((---(((--------                                       |
  |                                                           |
  |  [wake●●●][stt○○○][llm○○○][tts○○○]                       |  progress bar
  |                                                           |
  |  Cozy ui                                                   |  status
  |  v0.1                                                      |
  |  [ready]                                                   |  state
  |                                                           |
  |  > _                                                       |  text input (always)
  +----------------------------------------------------------+

States:
  0. LOADING    - progress bar fills, no "say hey cozy" text
  1. IDLE       - all 4 segments ●, "Cozy ui v0.1 [ready]"
  2. LISTENING  - audio visualizer in input area, top-left "listening"
  3. STT        - solid box with streaming text, "working" spinner
  4. LLM        - "thinking" spinner, then tool calls as dim text with
                   glowing indicators
  5. DONE       - LLM's final answer in BRIGHT WHITE, no "tts" text

The cat animates through 6 frames when waiting. Faster when listening.
Faster still when "thinking". Stops on the last frame when "done".
"""
from __future__ import annotations

import sys
import os
import threading
import time
import queue
import re
import math
from pathlib import Path
from typing import Optional

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from textual.app import App, ComposeResult
from textual.widgets import Static, Input
from textual.containers import Vertical, Horizontal
from textual.reactive import reactive
from textual.binding import Binding
from rich.text import Text
from rich.align import Align


# ── colors
COLOR_MUTED       = "#5c6370"
COLOR_BG_DIM      = "#3b3b3b"
COLOR_BRIGHT      = "#d0d6e0"
COLOR_WHITE       = "#ffffff"
COLOR_BLUE        = "#7aa2f7"
COLOR_GREEN       = "#c3e88d"
COLOR_PINK        = "#ff9b9b"
COLOR_RED         = "#ff6b6b"
COLOR_YELLOW      = "#e0c07c"


# ── cat animation (Hermes-style 6-frame cycle)
# Each frame: a different pose. The cycle speed is controlled by the
# state.
CAT_FRAMES = [
    # Frame 0: standing
    r"""
      |\__/,|   (`\
    _.|o o  |_   ) )
   -(((---(((--------
    """,
    # Frame 1: blinking
    r"""
      |\__/,|   (-\
    _.|o -  |_   ) )
   -(((---(((--------
    """,
    # Frame 2: tail flick
    r"""
      |\__/,|   (`/
    _.|o o  |_   ) )
   -(((---(((--------
    """,
    # Frame 3: alert
    r"""
      |\__/,|   (`\
    _.|o O  |_   ) )!
   -(((---(((--------
    """,
    # Frame 4: thinking
    r"""
      |\__/,|   (`\
    _.|o o  |_   ) ?
   -(((---(((-------
    """,
    # Frame 5: tail curled
    r"""
      |\__/,|   (`)
    _.|o o  |_   ) )
   -(((---(((--------
    """,
]


# ── spinners (Claude-code style)
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_DOTS = [".", "..", "...", "...."]


# ── progress bar characters
PROG_DONE      = "●"   # loaded
PROG_LOADING   = "◐"   # loading
PROG_PENDING   = "○"   # not started
PROG_FAILED    = "✗"   # failed


def strip_special_tokens(text):
    text = re.sub(r"<\\|im_end\\|>", "", text)
    text = re.sub(r"<\\|im_start\\|>", "", text)
    text = re.sub(r"<\\|endoftext\\|>", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()


class EventBus:
    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self.last_event_ts = 0.0

    def emit(self, kind, **fields):
        self.q.put({"kind": kind, "ts": time.time(), **fields})
        self.last_event_ts = time.time()

    def get_nowait(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None


class ProgressBar(Static):
    """4-segment progress bar: [wake●][stt●][llm●][tts●]

    State per segment: pending (○), loading (◐), done (●), failed (✗).
    Emits a "warmup_progress" event for each segment change.
    """
    segments = ["wake", "stt", "llm", "tts"]
    states = reactive({})  # name -> "pending" | "loading" | "done" | "failed"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.states = {n: "pending" for n in self.segments}

    def update_segment(self, name, state):
        if name in self.states:
            self.states[name] = state
            self.refresh()

    def render(self):
        parts = []
        for name in self.segments:
            state = self.states.get(name, "pending")
            if state == "done":
                mark, color = PROG_DONE, COLOR_GREEN
            elif state == "loading":
                mark, color = PROG_LOADING, COLOR_YELLOW
            elif state == "failed":
                mark, color = PROG_FAILED, COLOR_RED
            else:
                mark, color = PROG_PENDING, COLOR_MUTED
            seg = Text()
            seg.append(f"[{name}]", style="dim")
            seg.append(mark * 3, style=color)
            parts.append(seg)
            if name != self.segments[-1]:
                parts.append(Text("  "))
        line = Text()
        for i, p in enumerate(parts):
            line.append_text(p)
        return line


class Cat(Static):
    """The animated cat. Cycles through CAT_FRAMES at a rate that
    depends on the state."""
    state = reactive("loading")
    frame_idx = reactive(0)

    def on_mount(self):
        self._tick_interval = self.set_interval(0.4, self._tick)

    def _tick(self):
        self.frame_idx = (self.frame_idx + 1) % len(CAT_FRAMES)
        self.refresh()

    def watch_state(self, old, new):
        # Speed up the animation in active states
        try:
            self._tick_interval.stop()
        except Exception:
            pass
        if new in ("listening", "thinking"):
            self._tick_interval = self.set_interval(0.15, self._tick)
        elif new in ("speaking", "loading"):
            self._tick_interval = self.set_interval(0.25, self._tick)
        else:
            self._tick_interval = self.set_interval(0.6, self._tick)

    def render(self):
        idx = self.frame_idx
        return Text(CAT_FRAMES[idx], style=COLOR_BRIGHT)


class AudioVisualizer(Static):
    """A simple bar visualizer for the audio input during LISTENING.
    The input area becomes this visualizer when wake fires."""
    level = reactive(0.0)  # 0..1

    def render(self):
        bar_len = 30
        # Pulse the bar based on level
        n = int(self.level * bar_len)
        # Center-symmetric visualizer
        left = bar_len // 2
        right = bar_len - left
        # Fill from the center outward based on level
        fill = int(self.level * left)
        bar = list(" " * bar_len)
        for i in range(fill):
            bar[left - 1 - i] = "▌"
            if left + i < bar_len:
                bar[left + i] = "▐"
        return Text("[" + "".join(bar) + "]", style=COLOR_BLUE)


class StatusPanel(Static):
    """The middle text area: "Cozy ui", version, current state text."""
    title = reactive("Cozy ui")
    version = reactive("v0.1")
    state_text = reactive("")
    spinner_idx = reactive(0)
    state = reactive("loading")

    def on_mount(self):
        self.set_interval(0.15, self._tick_spinner)

    def _tick_spinner(self):
        self.spinner_idx = (self.spinner_idx + 1) % len(SPINNER_DOTS)

    def render(self):
        t = Text()
        t.append(self.title, style="bold " + COLOR_BRIGHT)
        t.append("\\n")
        t.append(self.version, style="dim")
        t.append("\\n")
        if self.state in ("thinking", "loading"):
            t.append("[" + SPINNER_DOTS[self.spinner_idx].ljust(4) + "]", style=COLOR_YELLOW)
        elif self.state == "listening":
            t.append("[", style=COLOR_MUTED)
            t.append("listening", style=COLOR_BLUE)
            t.append("]", style=COLOR_MUTED)
        elif self.state == "speaking":
            t.append("[", style=COLOR_MUTED)
            t.append("working", style=COLOR_PINK)
            t.append("]", style=COLOR_MUTED)
        elif self.state == "done":
            t.append("[", style=COLOR_MUTED)
            t.append("ready", style=COLOR_GREEN)
            t.append("]", style=COLOR_MUTED)
        elif self.state == "error":
            t.append("[", style=COLOR_MUTED)
            t.append("error", style=COLOR_RED)
            t.append("]", style=COLOR_MUTED)
        else:
            t.append("[", style=COLOR_MUTED)
            t.append("ready", style=COLOR_GREEN)
            t.append("]", style=COLOR_MUTED)
        # Current state text (the assistant's reply during done/thinking)
        if self.state_text:
            t.append("\\n")
            t.append(self.state_text, style=COLOR_WHITE if self.state == "done" else COLOR_BRIGHT)
        return t


class TextInputOrVisualizer(Static):
    """The bottom text-input area.

    In IDLE: shows a prompt with cursor.
    In LISTENING: shows the audio visualizer.
    In THINKING: shows streaming STT text in a solid box.
    In DONE: shows the LLM's final answer.
    """
    mode = reactive("idle")  # "idle" | "listening" | "thinking" | "done" | "stt"
    text = reactive("")
    audio_level = reactive(0.0)
    tool_calls = reactive([])  # list of {name, state, output}

    def render(self):
        if self.mode == "listening":
            return self._render_visualizer()
        if self.mode == "thinking":
            return self._render_stt_box()
        if self.mode == "stt":
            return self._render_stt_box()
        if self.mode == "done":
            return self._render_done()
        return self._render_idle()

    def _render_idle(self):
        t = Text()
        t.append("> ", style=COLOR_MUTED)
        t.append("_", style=COLOR_BRIGHT)
        return t

    def _render_visualizer(self):
        bar_len = 40
        n = int(self.audio_level * bar_len)
        left = bar_len // 2
        right = bar_len - left
        fill = int(self.audio_level * left)
        bar = list(" " * bar_len)
        for i in range(fill):
            if left - 1 - i >= 0:
                bar[left - 1 - i] = "▌"
            if left + i < bar_len:
                bar[left + i] = "▐"
        t = Text()
        t.append("> ", style=COLOR_MUTED)
        t.append("[" + "".join(bar) + "]", style=COLOR_BLUE)
        t.append(" ", style=COLOR_MUTED)
        t.append("listening", style=COLOR_BLUE)
        return t

    def _render_stt_box(self):
        t = Text()
        t.append("> ", style=COLOR_MUTED)
        if self.text:
            # Solid box around the text. Top + bottom border, side bars.
            lines = self.text.split("\\n")
            t.append("┌" + "─" * (max(len(l) for l in lines) + 2) + "┐", style=COLOR_BG_DIM)
            t.append("\\n")
            for line in lines:
                t.append("│ ", style=COLOR_BG_DIM)
                t.append(line, style=COLOR_BRIGHT)
                t.append(" │", style=COLOR_BG_DIM)
                t.append("\\n")
            t.append("└" + "─" * (max(len(l) for l in lines) + 2) + "┘", style=COLOR_BG_DIM)
        else:
            t.append("┌──────────────────────┐", style=COLOR_BG_DIM)
            t.append("\\n")
            t.append("│  ", style=COLOR_BG_DIM)
            t.append(SPINNER_DOTS[self.spinner_idx()], style=COLOR_YELLOW)
            t.append(" listening     │", style=COLOR_BG_DIM)
            t.append("\\n")
            t.append("└──────────────────────┘", style=COLOR_BG_DIM)
        return t

    def _render_done(self):
        """Bright white box for the LLM's final answer."""
        t = Text()
        if not self.text:
            t.append("> _", style=COLOR_MUTED)
            return t
        lines = self.text.split("\\n")
        maxlen = max(len(l) for l in lines)
        # Bright white box - the user said BRIGHT WHITE for the final answer
        t.append("┌" + "═" * (maxlen + 2) + "┐", style=COLOR_WHITE)
        t.append("\\n")
        for line in lines:
            t.append("║ ", style=COLOR_WHITE)
            t.append(line, style="bold " + COLOR_WHITE)
            t.append(" ║", style=COLOR_WHITE)
            t.append("\\n")
        t.append("└" + "═" * (maxlen + 2) + "┘", style=COLOR_WHITE)
        return t

    def spinner_idx(self):
        return getattr(self, "_spinner", 0)

    def watch_mode(self, old, new):
        if new == "thinking":
            self._spinner = 0
            self.set_interval(0.15, self._tick_spinner)

    def _tick_spinner(self):
        self._spinner = (self._spinner + 1) % len(SPINNER_DOTS)
        self.refresh()


class ToolCallLine(Static):
    """Renders one tool call with a glowing indicator.

    States: 'running' (yellow ● pulsing), 'done' (green ●), 'failed' (red ✗).
    Tool calls appear in DIM text except for the indicator.
    """
    name = reactive("")
    output = reactive("")
    state = reactive("running")  # running | done | failed

    def on_mount(self):
        self._pulse = 0
        self.set_interval(0.2, self._tick_pulse)

    def _tick_pulse(self):
        if self.state == "running":
            self._pulse = (self._pulse + 1) % 4
        self.refresh()

    def render(self):
        t = Text()
        if self.state == "running":
            marks = ["●", "◉", "○", "◉"]
            t.append(marks[self._pulse], style=COLOR_YELLOW)
        elif self.state == "done":
            t.append("●", style=COLOR_GREEN)
        else:
            t.append("✗", style=COLOR_RED)
        t.append(" ", style=COLOR_MUTED)
        t.append(self.name, style="dim " + COLOR_MUTED)
        if self.output:
            out_short = self.output[:80] + ("…" if len(self.output) > 80 else "")
            t.append("  →  ", style=COLOR_MUTED)
            t.append(out_short, style="dim")
        return t


class CozyApp(App):
    """The cozy TUI v3 - product-grade per the user's spec.

    5 distinct visual states, each with its own rendering:
      0. loading    - 4-segment progress bar filling
      1. idle       - "Cozy ui" + version + [ready] + text input
      2. listening  - audio visualizer in input area, "listening" indicator
      3. thinking   - STT streaming in solid box, "working" spinner
      4. llm        - "thinking" spinner, then dim tool calls with glowing dots
      5. done       - bright white box with the LLM's final answer
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, harness, executor, voice_mode=False, threshold=0.5):
        super().__init__()
        self.h = harness
        self.exec = executor
        self.voice_mode = voice_mode
        self.threshold = threshold
        self.events = EventBus()
        self.voice_thread = None
        self.stop_flag = False
        self._last_wake_score_ts = 0.0
        self._last_wake_score_val = 0.0
        self._peak_score = 0.0
        self._text_mode = False

    def compose(self):
        """The full layout per the user's spec.

        Cat top-left, title top-right, progress bar, status panel,
        text input / visualizer at the bottom. No header, no footer,
        no panels, no timestamps.
        """
        # Top: cat on the left, title on the right
        with Vertical():
            with Horizontal(id="top_row"):
                yield Cat(id="cat")
                yield StatusPanel(id="status")
            # Progress bar in the middle
            yield ProgressBar(id="progress")
            # Tool call log (initially empty, populated as LLM fires)
            yield Static(id="tool_log")
            # Bottom: text input or visualizer
            yield TextInputOrVisualizer(id="input_area")
        # Hidden input for actual text entry
        yield Input(id="hidden_input")

    def on_mount(self):
        try:
            self.query_one("#hidden_input").display = False
        except Exception:
            pass
        # State ticker - 10Hz
        self.set_interval(0.1, self._tick)
        # Event drainer - 10Hz
        self.set_interval(0.1, self._drain_events)
        # Initial state
        try:
            status = self.query_one("#status", StatusPanel)
            status.state = "loading"
            status.state_text = "loading wake + STT + LLM + TTS…"
        except Exception:
            pass
        if self.voice_mode:
            # Pre-load all 4 plugins in parallel - same thread starts the
            # voice loop, the warmup thread does LLM + TTS.
            def on_progress(name, ok):
                try:
                    bar = self.query_one("#progress", ProgressBar)
                    bar.update_segment(name, "done" if ok else "failed")
                    status = self.query_one("#status", StatusPanel)
                    if ok:
                        status.state_text = f"loaded {name}"
                except Exception:
                    pass
            if self.h is not None:
                self.h.warmup(on_progress=on_progress)
            self._start_voice_loop()

    def on_exception(self, exc):
        try:
            status = self.query_one("#status", StatusPanel)
            status.state = "error"
            status.state_text = f"{type(exc).__name__}: {exc}"
        except Exception:
            pass

    def _tick(self):
        try:
            status = self.query_one("#status", StatusPanel)
            inp = self.query_one("#input_area", TextInputOrVisualizer)
        except Exception:
            return
        # Decay: if no event in the last 5s, transition listening ->
        # idle (only if no audio activity)
        if (self.events.last_event_ts > 0 and
                time.time() - self.events.last_event_ts > 5 and
                status.state in ("listening", "thinking")):
            status.state = "idle"
            inp.mode = "idle"
            try:
                cat = self.query_one("#cat", Cat)
                cat.state = "idle"
            except Exception:
                pass

    def _drain_events(self):
        try:
            status = self.query_one("#status", StatusPanel)
            inp = self.query_one("#input_area", TextInputOrVisualizer)
            cat = self.query_one("#cat", Cat)
            bar = self.query_one("#progress", ProgressBar)
            tool_log = self.query_one("#tool_log", Static)
        except Exception:
            return
        now = time.time()
        ev = self.events.get_nowait()
        while ev is not None:
            try:
                kind = ev.get("kind")
                if kind == "wake_score":
                    score = float(ev.get("score", 0.0))
                    if (abs(score - self._last_wake_score_val) >= 0.05 or
                            now - self._last_wake_score_ts >= 0.2):
                        self._last_wake_score_val = score
                        self._last_wake_score_ts = now
                        inp.audio_level = score
                    if score > self._peak_score:
                        self._peak_score = score
                elif kind == "wake":
                    cat.state = "listening"
                    inp.mode = "listening"
                    status.state = "listening"
                elif kind == "heard":
                    cat.state = "thinking"
                    inp.mode = "thinking"
                    inp.text = ev.get("text", "")
                    status.state = "thinking"
                    status.state_text = "thinking…"
                elif kind == "llm":
                    # Tool call: add to the tool log with running indicator
                    name = ev.get("tool", "?")
                    args = ev.get("args", "")
                    # Update tool log widget
                    self._add_tool_line(tool_log, name, args, "running")
                    cat.state = "thinking"
                    status.state = "thinking"
                    status.state_text = f"calling {name}…"
                elif kind == "tool_result":
                    # Mark the last running tool as done
                    self._mark_last_tool(tool_log, "done",
                                          ev.get("out", ""))
                    cat.state = "thinking"
                elif kind == "tool_fail":
                    self._mark_last_tool(tool_log, "failed",
                                          ev.get("out", ""))
                elif kind == "tool_error":
                    self._mark_last_tool(tool_log, "failed",
                                          f"{ev.get('name')}: {ev.get('msg', '')}")
                elif kind == "llm_text":
                    # Streaming LLM text - show in the input area
                    cat.state = "thinking"
                    inp.mode = "thinking"
                    inp.text = ev.get("text", "")
                elif kind == "tts":
                    # The LLM's final answer is in tts text. Show it
                    # in BRIGHT WHITE in the input area. No "speaking"
                    # text - the user said "dont show tts text its a
                    # duplicate just show the text in bright white".
                    cat.state = "done"
                    inp.mode = "done"
                    inp.text = strip_special_tokens(ev.get("text", ""))
                    status.state = "done"
                    status.state_text = ""
                elif kind == "rejected":
                    pass
                elif kind == "error":
                    status.state = "error"
                    status.state_text = ev.get("msg", "")
                    inp.mode = "idle"
            except Exception as exc:
                try:
                    status.state = "error"
                    status.state_text = f"event: {exc}"
                except Exception:
                    pass
            ev = self.events.get_nowait()

    def _add_tool_line(self, tool_log_widget, name, args, state):
        """Append a ToolCallLine to the tool_log Static widget."""
        try:
            existing = tool_log_widget.renderable or Text()
            new = ToolCallLine()
            new.name = name
            new.args = args
            new.state = state
            # Append to the renderable
            if isinstance(existing, Text):
                combined = Text()
                combined.append_text(existing)
                combined.append("\\n")
                combined.append_text(new.render())
                tool_log_widget.update(combined)
            else:
                tool_log_widget.update(new.render())
        except Exception:
            pass

    def _mark_last_tool(self, tool_log_widget, state, output):
        """Mark the last tool call as done/failed with its output."""
        try:
            from rich.console import Group
            from rich.text import Text
            # Rebuild the tool log from scratch (simpler than mutating)
            lines = getattr(self, "_tool_lines", [])
            if lines:
                last = lines[-1]
                last["state"] = state
                last["output"] = output
            # The above mutation doesn't update the UI; do a full refresh
            self._refresh_tool_log(tool_log_widget)
        except Exception:
            pass

    def _refresh_tool_log(self, tool_log_widget):
        try:
            lines = getattr(self, "_tool_lines", [])
            text = Text()
            for line in lines:
                t = ToolCallLine()
                t.name = line.get("name", "?")
                t.args = line.get("args", "")
                t.state = line.get("state", "done")
                t.output = line.get("output", "")
                text.append_text(t.render())
                text.append("\\n")
            tool_log_widget.update(text)
        except Exception:
            pass

    def on_key(self, event):
        key = event.key
        if key == "escape":
            try:
                self.query_one("#hidden_input").display = False
                self.query_one("#input_area", TextInputOrVisualizer).mode = "idle"
            except Exception:
                pass
            return
        if len(key) > 1 and not key.startswith("ctrl"):
            return
        if key.startswith("ctrl"):
            return
        try:
            inp = self.query_one("#hidden_input")
            inp.display = True
            inp.focus()
        except Exception:
            pass

    def on_input_submitted(self, event):
        text = event.value.strip()
        event.input.value = ""
        event.input.display = False
        if not text:
            return
        self._handle_user_text(text)

    def _handle_user_text(self, text):
        if self.h is None:
            return
        try:
            status = self.query_one("#status", StatusPanel)
            inp = self.query_one("#input_area", TextInputOrVisualizer)
            cat = self.query_one("#cat", Cat)
        except Exception:
            return
        status.state = "thinking"
        cat.state = "thinking"
        inp.mode = "thinking"
        inp.text = text
        status.state_text = "thinking…"
        t0 = time.time()
        try:
            name, args = self.h.decide(text)
        except Exception as exc:
            status.state = "error"
            status.state_text = f"decide: {exc}"
            return
        dt = time.time() - t0
        # Init tool log buffer
        if not hasattr(self, "_tool_lines"):
            self._tool_lines = []
        if name == "none" or name == "":
            for t in reversed(self.h.trace.recent):
                if t.role == "assistant" and t.content:
                    txt = strip_special_tokens(t.content)
                    self._show_done(txt)
                    from tts import is_available, speak
                    if is_available():
                        speak(txt)
                    break
        elif name:
            status.state = "thinking"
            self._tool_lines.append({
                "name": name, "args": str(args)[:60], "state": "running", "output": ""
            })
            self._refresh_tool_log(self.query_one("#tool_log", Static))
            try:
                result = self.exec(name, args or {})
                if result.get("ok"):
                    out = result.get("output", "")
                    self._tool_lines[-1]["state"] = "done"
                    self._tool_lines[-1]["output"] = out[:200]
                    self._refresh_tool_log(self.query_one("#tool_log", Static))
                    self._show_done(out)
                    from tts import is_available, speak
                    if is_available():
                        speak(out)
                else:
                    self._tool_lines[-1]["state"] = "failed"
                    self._tool_lines[-1]["output"] = result.get("output", "")[:200]
                    self._refresh_tool_log(self.query_one("#tool_log", Static))
                    status.state = "error"
                    status.state_text = result.get("output", "")
            except Exception as exc:
                self._tool_lines[-1]["state"] = "failed"
                self._tool_lines[-1]["output"] = f"error: {exc}"
                self._refresh_tool_log(self.query_one("#tool_log", Static))
                status.state = "error"
                status.state_text = str(exc)

    def _show_done(self, text):
        try:
            status = self.query_one("#status", StatusPanel)
            inp = self.query_one("#input_area", TextInputOrVisualizer)
            cat = self.query_one("#cat", Cat)
        except Exception:
            return
        cat.state = "done"
        status.state = "done"
        status.state_text = ""
        inp.mode = "done"
        inp.text = strip_special_tokens(text)

    def _start_voice_loop(self):
        from livekit.wakeword import WakeWordModel
        from stt import CozySTT
        from tts import is_available, speak as tts_speak
        import sounddevice as sd
        import numpy as np
        WW_PATH = Path("/home/saptodeepsarkar/Projects/Cozy/wakeword/output/hey_cozy/hey_cozy.onnx")
        if not WW_PATH.exists():
            return
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            wake = WakeWordModel(models=[str(WW_PATH)])
            stt = CozySTT()
        wake_name = next(iter(wake._classifiers.keys()))
        # Mark wake + STT as done
        try:
            self.query_one("#progress", ProgressBar).update_segment("wake", "done")
            self.query_one("#progress", ProgressBar).update_segment("stt", "done")
        except Exception:
            pass
        SR = 16000
        CHUNK = 1280
        WIN = SR * 2
        audio_q = queue.Queue()
        audio_buf = np.zeros(WIN, dtype=np.int16)
        audio_buf_fill = 0
        def audio_cb(indata, *_):
            audio_q.put(indata.copy())
        def voice_worker():
            nonlocal audio_buf, audio_buf_fill
            cooldown_until = 0.0
            try:
                with sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                                    blocksize=CHUNK, callback=audio_cb):
                    while not self.stop_flag:
                        try:
                            chunk = audio_q.get(timeout=0.5)
                        except queue.Empty:
                            continue
                        if time.time() < cooldown_until:
                            continue
                        chunk = chunk[:, 0].astype(np.int16)
                        n = len(chunk)
                        audio_buf = np.roll(audio_buf, -n)
                        audio_buf[-n:] = chunk
                        audio_buf_fill = min(WIN, audio_buf_fill + n)
                        if audio_buf_fill < WIN:
                            continue
                        scores = wake.predict(audio_buf.copy())
                        score = float(scores[wake_name])
                        self.events.emit("wake_score", score=score)
                        if score < self.threshold:
                            continue
                        cooldown_until = time.time() + 4.0
                        self.events.emit("wake", score=score)
                        text = self._capture_and_transcribe(stt, audio_q, audio_buf, audio_buf_fill)
                        if not text:
                            continue
                        if self._is_self_feedback(text):
                            self.events.emit("rejected", reason="self-feedback (TTS echo)")
                            cooldown_until = time.time() + 6.0
                            continue
                        self.events.emit("heard", text=text)
                        if self.h is None:
                            continue
                        t0 = time.time()
                        try:
                            name, args = self.h.decide(text)
                        except Exception as exc:
                            self.events.emit("error", msg=str(exc))
                            continue
                        dt = time.time() - t0
                        if name == "none":
                            for t in reversed(self.h.trace.recent):
                                if t.role == "assistant" and t.content:
                                    text_reply = strip_special_tokens(t.content)
                                    self.events.emit("llm_text", text=text_reply, dt=dt)
                                    if is_available():
                                        tts_speak(text_reply)
                                        self.events.emit("tts", text=text_reply)
                                    break
                        elif name:
                            self.events.emit("llm", tool=name, args=str(args)[:60], dt=dt)
                            try:
                                result = self.exec(name, args or {})
                                if result.get("ok"):
                                    out = result.get("output", "")
                                    self.events.emit("tool_result", name=name, out=out)
                                    if is_available():
                                        tts_speak(out)
                                        self.events.emit("tts", text=out)
                                else:
                                    self.events.emit("tool_fail", name=name, out=result.get("output", ""))
                            except Exception as exc:
                                self.events.emit("tool_error", name=name, msg=str(exc))
                        else:
                            for t in reversed(self.h.trace.recent):
                                if t.role == "assistant" and t.content:
                                    text_reply = strip_special_tokens(t.content)
                                    self.events.emit("llm_text", text=text_reply, dt=dt)
                                    if is_available():
                                        tts_speak(text_reply)
                                        self.events.emit("tts", text=text_reply)
                                    break
                        audio_buf[:] = 0
                        audio_buf_fill = 0
            except Exception as exc:
                self.events.emit("error", msg=f"voice worker died: {exc}")
        self.voice_thread = threading.Thread(target=voice_worker, daemon=True)
        self.voice_thread.start()

    def _is_self_feedback(self, text):
        if self.h is None:
            return False
        last = ""
        for t in reversed(self.h.trace.recent):
            if t.role == "assistant" and t.content:
                last = strip_special_tokens(t.content)
                break
        if not last:
            return False
        a = set(last.lower().split())
        b = set(text.lower().split())
        if not a or not b:
            return False
        return len(a & b) / min(len(a), len(b)) > 0.5

    def _capture_and_transcribe(self, stt, audio_q, audio_buf, audio_buf_fill):
        import numpy as np
        import soundfile as sf
        from pathlib import Path as _P
        frames = [audio_buf.copy()]
        silent_for = 0.0
        spoken = False
        t0 = time.time()
        while time.time() - t0 < 7.0:
            try:
                chunk = audio_q.get(timeout=0.05)
            except queue.Empty:
                chunk = None
            if chunk is not None:
                pcm = chunk[:, 0]
                frames.append(pcm.copy())
                level = float(np.abs(pcm).mean())
                if level > 600:
                    spoken = True
                    silent_for = 0.0
                elif spoken:
                    silent_for += len(pcm) / 16000
            if spoken and silent_for >= 1.0:
                break
        pcm = np.concatenate(frames) if len(frames) > 1 else np.zeros(16000, np.int16)
        energy = float(np.abs(pcm).mean())
        if energy < 100:
            return ""
        tmp = _P("/tmp/cozy_cmd.wav")
        sf.write(str(tmp), pcm, 16000, subtype="PCM_16")
        try:
            text = stt.transcribe_file(str(tmp))
        except Exception:
            return ""
        text = (text or "").strip()
        if len(text) < 3:
            return ""
        if not any(c.isalpha() for c in text):
            return ""
        return text

    def action_quit(self):
        self.stop_flag = True
        if self.voice_thread is not None:
            self.voice_thread.join(timeout=2.0)
        self.exit()


def run_textual(harness, executor, voice_mode=False, threshold=0.5):
    app = CozyApp(harness, executor, voice_mode=voice_mode, threshold=threshold)
    app.run()
