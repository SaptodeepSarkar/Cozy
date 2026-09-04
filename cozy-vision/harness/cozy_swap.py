"""Swap cosy-llm-v1 off the dGPU to free VRAM for the vision harness.

Strategy:
  1. Find all cozy assistant processes (skipping --harness-only).
  2. Kill them with SIGTERM. systemd (PID 1577) may respawn them.
  3. Wait up to N seconds for the GPU to be free.
  4. Run the vision task.
  5. Restart cozy by spawning a fresh instance if systemd didn't.

Usage:
  from cozy_vision.harness.cozy_swap import swap_out, swap_in
  paused = swap_out()
  try:
      ... vision task ...
  finally:
      swap_in()
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

COZY_MARKER = "assistant/runtime.py"
COZY_ROOT = Path(__file__).resolve().parents[2] / "Cozy"
COZY_PYTHON = str(COZY_ROOT / "assistant" / ".venv" / "bin" / "python")
COZY_RUNTIME = str(COZY_ROOT / "assistant" / "runtime.py")


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


def _gpu_used_mb():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


def swap_out(timeout_s: float = 10.0) -> list[int]:
    """Kill cozies and wait for the GPU to be free.

    A watchdog (e.g. the dsh-qa-box sidecar, or the prime agent) may
    respawn cozy after we kill it. We loop: kill, wait, check, kill
    any newcomers, until the GPU is actually free for `timeout_s` of
    consecutive seconds, or the global timeout hits.

    Returns the PIDs we killed (best effort).
    """
    pids_killed: list[int] = []
    deadline = time.time() + timeout_s
    quiet_until = time.time() + 5.0  # need 5s of GPU < 200 MB to be "free"
    while time.time() < deadline:
        current = _cozy_pids()
        for pid in current:
            if pid not in pids_killed:
                try:
                    os.kill(pid, signal.SIGTERM)
                    pids_killed.append(pid)
                except ProcessLookupError:
                    pass
        # SIGKILL any survivors after a short grace
        time.sleep(1.0)
        for pid in current:
            if _is_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        time.sleep(0.5)
        if _gpu_used_mb() < 200:
            if time.time() >= quiet_until:
                return pids_killed
        else:
            quiet_until = time.time() + 5.0
    return pids_killed


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def swap_in(log_file: str = "/tmp/cozy_respawn.log") -> int:
    """Respawn cozy if it's not running. Returns the new PID or 0."""
    if _cozy_pids():
        # Already running
        return 0
    if not Path(COZY_PYTHON).is_file() or not Path(COZY_RUNTIME).is_file():
        return 0
    # Spawn detached
    p = subprocess.Popen(
        [COZY_PYTHON, COZY_RUNTIME, "--text"],
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return p.pid
