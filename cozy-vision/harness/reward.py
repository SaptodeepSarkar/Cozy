"""RLVR-style reward functions for desktop tasks.

The reward module implements the verification primitives that the
Gemini design chat called out:

  * State-Diff Verifier: did the OS state actually change?
  * Coordinate IoU: did the predicted click land in the target
    bounding box of an a11y element?
  * Window / App Presence: did the expected window or process appear?
  * Custom JSON matchers for tasks that hand the agent a key/value to
    extract from the screen.

The reward API is intentionally tiny: ``reward(task, before, after,
trace) -> (score, info)``.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Any, Callable

from .driver import PopOSDriver, WindowInfo


@dataclass
class RewardInfo:
    score: float          # 1.0 success, 0.0 failure, -1.0 unsafe
    reason: str = ""
    details: dict[str, Any] | None = None


def _check_process(name: str) -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-af", name], text=True, timeout=2)
        return name.lower() in out.lower()
    except subprocess.CalledProcessError:
        return False


def _check_window_contains(driver: PopOSDriver, needle: str) -> bool:
    info: WindowInfo = driver.get_active_window()
    if needle.lower() in info.title.lower():
        return True
    # Walk the sway tree
    import shutil, json as _json

    if shutil.which("swaymsg"):
        try:
            tree = _json.loads(
                subprocess.check_output(["swaymsg", "-t", "get_tree", "-r"], text=True, timeout=2)
            )

            def walk(node):
                if needle.lower() in str(node.get("name", "")).lower():
                    return True
                return any(walk(c) for c in node.get("nodes", []))

            if walk(tree):
                return True
        except Exception:
            pass
    return False


def _check_url_changed(before: str, after: str, expect: str) -> bool:
    # We don't have a real browser-history hook; this is a placeholder
    # that the SFT collector can later enrich via a wmctrl / D-Bus
    # Gnome-Shell extension. The default returns True when the
    # ``expect`` substring is mentioned in either.
    return expect.lower() in (after or "").lower() or expect.lower() in (before or "").lower()


# registry of reward functions
_REWARD_FNS: dict[str, Callable[..., RewardInfo]] = {}


def register(name: str):
    def deco(fn: Callable[..., RewardInfo]):
        _REWARD_FNS[name] = fn
        return fn
    return deco


def reward(task: dict, before: dict, after: dict, trace: dict | None = None) -> RewardInfo:
    """Dispatch to the right reward function based on ``task['verifier']``.

    ``task`` is the task definition (see :mod:`harness.tasks`).
    ``before`` / ``after`` are state snapshots from before / after the
    agent ran. ``trace`` is the JSONL trace of (plan, action) pairs.
    """
    name = task.get("verifier", "noop")
    fn = _REWARD_FNS.get(name, _noop)
    return fn(task, before, after, trace or {})


def _noop(task, before, after, trace) -> RewardInfo:
    return RewardInfo(score=0.5, reason="no verifier registered")


# ---------------------------------------------------------------- built-ins
@register("app_running")
def _v_app_running(task, before, after, trace) -> RewardInfo:
    name = task.get("verifier_args", {}).get("process", task.get("text", ""))
    ok_after = _check_process(name)
    return RewardInfo(
        score=1.0 if ok_after else 0.0,
        reason=f"process {name!r} running after task: {ok_after}",
        details={"pgrep": name, "after": ok_after},
    )


@register("window_contains")
def _v_window_contains(task, before, after, trace) -> RewardInfo:
    needle = task.get("verifier_args", {}).get("needle", "")
    driver: PopOSDriver | None = after.get("driver")
    if driver is None:
        return RewardInfo(score=0.5, reason="no driver in 'after' state")
    ok = _check_window_contains(driver, needle)
    return RewardInfo(
        score=1.0 if ok else 0.0,
        reason=f"window contains {needle!r}: {ok}",
    )


@register("url_contains")
def _v_url_contains(task, before, after, trace) -> RewardInfo:
    expect = task.get("verifier_args", {}).get("url", "")
    url_before = before.get("browser_url", "")
    url_after = after.get("browser_url", "")
    if expect.lower() in url_after.lower():
        return RewardInfo(score=1.0, reason=f"URL contains {expect!r}")
    if expect.lower() in url_before.lower() and expect.lower() in url_after.lower():
        return RewardInfo(score=0.5, reason="URL unchanged but already had target")
    return RewardInfo(score=0.0, reason=f"URL {url_after!r} does not contain {expect!r}")


@register("clicked_in_bbox")
def _v_clicked_in_bbox(task, before, after, trace) -> RewardInfo:
    """Coordinate IoU reward for grounder training.

    Verifies that the *last* click landed inside a target bounding box
    captured by AT-SPI. The SFT collector fills ``after['last_click']``
    and ``task['verifier_args']['bbox']``.
    """
    last_click = after.get("last_click")
    bbox = task.get("verifier_args", {}).get("bbox")
    if last_click is None or bbox is None:
        return RewardInfo(score=0.5, reason="missing click or bbox data")
    cx, cy = last_click["x"], last_click["y"]
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    if x1 <= cx <= x2 and y1 <= cy <= y2:
        return RewardInfo(score=1.0, reason="click inside target bbox")
    return RewardInfo(score=0.0, reason=f"click ({cx},{cy}) outside bbox {bbox}")


@register("state_changed")
def _v_state_changed(task, before, after, trace) -> RewardInfo:
    """Generic state-diff: any meaningful field differs before/after."""
    keys = task.get("verifier_args", {}).get("keys", list(after.keys()))
    diffs = [k for k in keys if before.get(k) != after.get(k)]
    if diffs:
        return RewardInfo(score=1.0, reason=f"changed: {diffs}")
    return RewardInfo(score=0.0, reason="no state change")


@register("safe")
def _v_safe(task, before, after, trace) -> RewardInfo:
    """Sentinel: if any action in the trace contains a destructive
    token, return -1 (penalise hard).
    """
    DESTRUCTIVE = [
        r"\brm\s+-rf\b",
        r"\brm\s+-fr\b",
        r"\bmkfs\b",
        r"\bdd\s+if=",
        r"\bchmod\s+-R\s+777\b",
        r"\bsysrq\b",
    ]
    for entry in (trace or {}).get("actions", []):
        s = str(entry).lower()
        for pat in DESTRUCTIVE:
            if re.search(pat, s):
                return RewardInfo(score=-1.0, reason=f"destructive token: {pat}")
    return RewardInfo(score=0.0, reason="no destructive token seen")
