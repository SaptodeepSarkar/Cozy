"""Pause/resume the cozy assistant runtime to free GPU VRAM.

The cozy voice assistant (assistant/runtime.py) keeps Qwen3-0.6B
loaded on the dGPU (~1.7 GB). When the vision harness needs that
memory for the VLM+VLA, we can SIGSTOP the assistant process.

This is NOT the same as killing the process. When SIGCONT is sent,
the cozy runtime picks up right where it left off. Wake word, STT,
and LLM state are all preserved (just frozen in time).
"""
from __future__ import annotations

import os
import signal
import subprocess

COZY_MARKER = "assistant/runtime.py"


def _cozy_pids():
    r = subprocess.run(["pgrep", "-f", COZY_MARKER], capture_output=True, text=True)
    if r.returncode != 0:
        return []
    out = []
    for pid_s in r.stdout.split():
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().decode("utf-8", errors="replace")
        except FileNotFoundError:
            continue
        if COZY_MARKER not in cmd:
            continue
        if "--harness-only" in cmd:
            continue
        out.append(pid)
    return out


def pause_cozy():
    """SIGSTOP every cozy runtime. Returns the PIDs that were paused."""
    paused = []
    for pid in _cozy_pids():
        try:
            os.kill(pid, signal.SIGSTOP)
            paused.append(pid)
        except ProcessLookupError:
            pass
    return paused


def resume_cozy(pids):
    """SIGCONT the previously-paused cozies (best-effort)."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
