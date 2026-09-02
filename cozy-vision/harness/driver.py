"""Pop!_OS / COSMIC input + screenshot driver.

Strategy:
  * Input via ``ydotool`` (uinput-based, works on Wayland). Falls
    back to ``xdotool`` on X11. If neither is present, the driver
    is in **dry-run** mode — actions are logged but not executed
    (useful for collecting SFT traces without a real desktop).
  * Screenshot in this order:
      1. ``gnome-screenshot`` (works on COSMIC because it uses the
         gnome-shell D-Bus API; available out-of-the-box on
         Pop!_OS 24.04)
      2. D-Bus ScreenCast portal (``org.freedesktop.portal.Screenshot``)
      3. ``grim`` (wlroots screencopy — fails on COSMIC)
      4. ``grim -o`` for a specific output
      5. ``xwd`` (X11)
      6. ``PIL.ImageGrab`` (X11 only)
      7. **Synthetic placeholder** — a 1280x800 grey gradient with
         the current Unix timestamp. Lets the harness run for
         training/testing when no display is reachable.

The driver exposes a small synchronous API: :meth:`screenshot`,
:meth:`click`, :meth:`move`, :meth:`type_text`, :meth:`hotkey`,
:meth:`scroll`, :meth:`get_active_window`, :meth:`is_app_running`.

On a fresh Pop!_OS install: ``sudo apt install ydotool
gnome-screenshot grim wl-clipboard``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SHOTS_DIR = HERE.parent / "data" / "screenshots"
SHOTS_DIR.mkdir(parents=True, exist_ok=True)


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


@dataclass
class WindowInfo:
    title: str
    pid: int | None
    app: str | None


class PopOSDriver:
    """Synchronous Pop!_OS / COSMIC driver.

    Parameters
    ----------
    screen_size
        ``(width, height)`` of the target display. Defaults to a
        query of the compositor, or 1920x1080 if that fails.
    backend
        ``"auto"``, ``"ydotool"``, ``"xdotool"``, or ``"dryrun"``.
        ``"dryrun"`` means input is logged but not sent (safe for
        collecting SFT traces on a headless box).
    shot_backend
        ``"auto"`` picks the first working backend from the chain
        above. ``"synthetic"`` always paints a grey placeholder
        (useful for SFT data generation in CI).
    dryrun
        If True, all input actions are no-ops. Screenshot still
        works (real or synthetic).
    """

    def __init__(
        self,
        screen_size: tuple[int, int] | None = None,
        backend: str = "auto",
        shot_backend: str = "auto",
        dryrun: bool = False,
    ) -> None:
        self.dryrun = dryrun
        if dryrun:
            self.backend = "dryrun"
        else:
            self.backend = self._pick_backend(backend)
        self.shot_backend = self._pick_shot_backend(shot_backend)
        self.screen_size = screen_size or self._query_screen_size()
        self._shot_counter = 0
        self._action_log: list[str] = []

    # ------------------------------------------------------------------ backends
    def _pick_backend(self, want: str) -> str:
        if want != "auto":
            return want
        if _have("ydotool"):
            return "ydotool"
        if _have("xdotool"):
            return "xdotool"
        # No real backend. Switch to dry-run.
        return "dryrun"

    def _pick_shot_backend(self, want: str) -> str:
        if want == "synthetic":
            return "synthetic"
        if want != "auto":
            return want
        # COSMIC: use its native tool first
        if _have("cosmic-screenshot"):
            return "cosmic-screenshot"
        if _have("gnome-screenshot"):
            return "gnome-screenshot"
        if _have("grim"):
            return "grim"
        if _have("xwd"):
            return "xwd"
        return "synthetic"

    def _query_screen_size(self) -> tuple[int, int]:
        if _have("swaymsg"):
            try:
                import json as _json
                out = subprocess.check_output(
                    ["swaymsg", "-t", "get_outputs", "-r"], text=True, timeout=2
                )
                for o in _json.loads(out):
                    rect = o.get("rect", {})
                    if rect.get("width") and rect.get("height"):
                        return int(rect["width"]), int(rect["height"])
            except Exception:
                pass
        if _have("xrandr"):
            try:
                out = subprocess.check_output(["xrandr"], text=True, timeout=2)
                for line in out.splitlines():
                    if " connected " in line and "+0+0" in line:
                        for tok in line.split():
                            if "x" in tok and "+" in tok:
                                dims = tok.split("+")[0]
                                w, h = dims.split("x")
                                return int(w), int(h)
            except Exception:
                pass
        return (1920, 1080)

    # ------------------------------------------------------------------ screenshot
    def screenshot(self, save: bool = True) -> Image.Image:
        """Capture the entire screen and return a PIL image.

        Tries the configured backend; falls back to the chain if it
        fails. If everything fails, returns a synthetic grey
        placeholder so downstream code can still run.
        """
        path = SHOTS_DIR / f"shot_{int(time.time()*1000):013d}_{self._shot_counter}.png"
        self._shot_counter += 1
        img = None
        for backend in [self.shot_backend, "cosmic-screenshot", "gnome-screenshot", "grim", "xwd", "synthetic"]:
            try:
                if backend == "cosmic-screenshot":
                    if not _have("cosmic-screenshot"):
                        continue
                    # cosmic-screenshot ignores --save-dir and always
                    # saves to ~/Pictures/. Run it, then find the file.
                    pics_dir = Path.home() / "Pictures"
                    # Remember existing files so we can find the new one
                    before = set(pics_dir.glob("Screenshot_*.png")) if pics_dir.exists() else set()
                    subprocess.run(
                        ["cosmic-screenshot", "--interactive=false",
                         "--modal=false", "--notify=false"],
                        check=True, timeout=10,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    after = set(pics_dir.glob("Screenshot_*.png")) if pics_dir.exists() else set()
                    new_files = list(after - before)
                    if not new_files:
                        raise RuntimeError("cosmic-screenshot ran but no new file appeared")
                    # Take the most recent
                    src = max(new_files, key=lambda p: p.stat().st_mtime)
                    if save:
                        import shutil
                        shutil.copy(src, path)
                    else:
                        path.write_bytes(src.read_bytes())
                elif backend == "gnome-screenshot":
                    if not _have("gnome-screenshot"):
                        continue
                    subprocess.run(["gnome-screenshot", "-f", str(path)], check=True, timeout=5)
                elif backend == "grim":
                    if not _have("grim"):
                        continue
                    subprocess.run(["grim", str(path)], check=True, timeout=5,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                elif backend == "xwd":
                    if not _have("xwd"):
                        continue
                    subprocess.run(["xwd", "-root", "-out", str(path)], check=True, timeout=5)
                elif backend == "synthetic":
                    img = self._synthetic_screenshot()
                    if save:
                        img.save(path)
                else:
                    continue
                if img is None:
                    img = Image.open(path)
                if not save and path.exists():
                    path.unlink()
                return img
            except Exception as e:
                if img is not None:
                    break
                continue
        # Last resort
        img = self._synthetic_screenshot()
        if save:
            img.save(path)
        return img

    def _synthetic_screenshot(self) -> Image.Image:
        """Paint a 1280x800 grey placeholder with the current task text."""
        w, h = self.screen_size
        img = Image.new("RGB", (w, h), color=(40, 40, 48))
        d = ImageDraw.Draw(img)
        d.rectangle([(0, 0), (w, 28)], fill=(20, 20, 28))
        d.text((10, 6), "pop!_os  |  cosmic  |  [synthetic screenshot]", fill=(180, 180, 200))
        d.text((20, h - 60), f"timestamp: {time.time():.0f}", fill=(150, 150, 150))
        d.text((20, h - 40), f"size: {w}x{h}", fill=(150, 150, 150))
        return img

    # ------------------------------------------------------------------ input
    def click(self, x: int, y: int, button: str = "left", count: int = 1) -> None:
        if self.backend == "dryrun":
            self._action_log.append(f"click({x},{y},{button},{count})")
            return
        self.move(x, y)
        self._ydo_or_xdo(["click", button, str(count)])

    def move(self, x: int, y: int) -> None:
        if self.backend == "dryrun":
            self._action_log.append(f"move({x},{y})")
            return
        self._ydo_or_xdo(["mousemove", str(x), str(y)])

    def type_text(self, text: str) -> None:
        if self.backend == "dryrun":
            self._action_log.append(f"type({text!r})")
            return
        if self.backend == "ydotool":
            subprocess.run(["ydotool", "type", "--", text], check=True, timeout=10)
        else:
            subprocess.run(["xdotool", "type", "--clearmodifiers", "--", text], check=True, timeout=10)

    def hotkey(self, keys: Iterable[str]) -> None:
        keys = list(keys)
        if self.backend == "dryrun":
            self._action_log.append(f"hotkey({','.join(keys)})")
            return
        combo = "+".join(keys)
        if self.backend == "ydotool":
            subprocess.run(["ydotool", "key", combo], check=True, timeout=5)
        else:
            subprocess.run(["xdotool", "key", "--clearmodifiers", combo], check=True, timeout=5)

    def scroll(self, dx: int, dy: int) -> None:
        if self.backend == "dryrun":
            self._action_log.append(f"scroll({dx},{dy})")
            return
        if self.backend == "ydotool":
            subprocess.run(
                ["ydotool", "mousemove", "--wheel", "1", "0", str(dy)],
                check=False, timeout=5,
            )
        else:
            subprocess.run(["xdotool", "scroll", str(dx), str(dy)], check=False, timeout=5)

    def _ydo_or_xdo(self, args: list[str]) -> None:
        cmd = [self.backend, *args]
        if cmd[0] == "ydotool" and args[0] == "click":
            button_map = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
            btn = button_map.get(args[1], "0xC0")
            count = int(args[2]) if len(args) > 2 else 1
            cmd = ["ydotool", "click", btn] * count
        try:
            subprocess.run(cmd, check=True, timeout=5)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"input command failed: {e}. The driver may not have permission. "
                f"Add the user to the 'input' group: sudo usermod -aG input $USER, "
                f"then log out and back in."
            ) from e

    # ------------------------------------------------------------------ state
    def get_active_window(self) -> WindowInfo:
        if _have("swaymsg"):
            try:
                import json as _json
                tree = _json.loads(
                    subprocess.check_output(["swaymsg", "-t", "get_tree", "-r"], text=True, timeout=2)
                )
                node = self._find_focused(tree)
                if node:
                    return WindowInfo(
                        title=str(node.get("name", "")),
                        pid=int(node.get("pid")) if node.get("pid") else None,
                        app=str(node.get("app_id") or node.get("window_properties", {}).get("class", "")),
                    )
            except Exception:
                pass
        if _have("xdotool"):
            try:
                out = subprocess.check_output(
                    ["xdotool", "getactivewindow", "getwindowname"], text=True, timeout=2
                ).strip()
                pid_s = subprocess.check_output(
                    ["xdotool", "getactivewindow", "getwindowpid"], text=True, timeout=2
                ).strip()
                return WindowInfo(title=out, pid=int(pid_s) or None, app=None)
            except Exception:
                pass
        return WindowInfo(title="", pid=None, app=None)

    def _find_focused(self, node: dict):
        if node.get("focused"):
            return node
        for child in node.get("nodes", []):
            r = self._find_focused(child)
            if r is not None:
                return r
        return None

    def is_app_running(self, name: str) -> bool:
        try:
            out = subprocess.check_output(["pgrep", "-af", name], text=True, timeout=2)
            return name.lower() in out.lower()
        except subprocess.CalledProcessError:
            return False
        except Exception:
            return False

    def actions_so_far(self) -> list[str]:
        return list(self._action_log)
