"""Live OS context collector for Pop!_OS / COSMIC.

Gathers a structured snapshot of the desktop state at the moment the
agent is asked to plan. This snapshot is fed into the VLM system
prompt so the planner can write a plan that is grounded in *what is
actually on screen right now* (active window, focused element, open
windows, recent activity) rather than guessing.

Sources (in order of preference):
  * wlroots foreign-toplevel protocol via ``swaymsg`` (COSMIC exposes
    it). Gives the full tree of open windows with PID, app_id,
    title, geometry.
  * xdotool / xprop fallback under XWayland.
  * AT-SPI2 via ``pyatspi`` for the focused element's name, role, and
    a11y text.

The collector is read-only — it never moves the mouse, never types.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

try:
    import pyatspi  # type: ignore
    HAVE_ATSPI = True
except Exception:
    HAVE_ATSPI = False


@dataclass
class WindowState:
    title: str
    app_id: Optional[str] = None
    pid: Optional[int] = None
    geometry: Optional[dict] = None  # {x, y, width, height}
    focused: bool = False


@dataclass
class OSContext:
    active_window: Optional[WindowState] = None
    open_windows: list[WindowState] = field(default_factory=list)
    working_area: Optional[dict] = None      # {x, y, width, height}
    screen_size: tuple[int, int] = (1920, 1080)
    focused_element: Optional[dict] = None   # {name, role, text, app, pid}
    recent_apps: list[str] = field(default_factory=list)  # recently used
    clipboard: Optional[str] = None
    cursor_xy: tuple[int, int] = (0, 0)
    timestamp: float = 0.0

    def to_prompt(self) -> str:
        """Return a compact text block suitable for a system prompt."""
        lines = ["# OS CONTEXT (live snapshot)"]
        if self.active_window:
            aw = self.active_window
            lines.append(
                f"active_window: {aw.title!r}"
                + (f"  app_id={aw.app_id}" if aw.app_id else "")
                + (f"  pid={aw.pid}" if aw.pid else "")
            )
        else:
            lines.append("active_window: (none / desktop)")
        if self.open_windows:
            lines.append(f"open_windows ({len(self.open_windows)}):")
            for w in self.open_windows[:12]:
                bits = [f"  - {w.title!r}"]
                if w.app_id:
                    bits.append(f"app_id={w.app_id}")
                if w.focused:
                    bits.append("[FOCUSED]")
                lines.append(" ".join(bits))
        if self.working_area:
            wa = self.working_area
            lines.append(
                f"working_area: x={wa['x']} y={wa['y']} {wa['width']}x{wa['height']}"
            )
        lines.append(f"screen_size: {self.screen_size[0]}x{self.screen_size[1]}")
        if self.focused_element:
            fe = self.focused_element
            bits = [f"focused_element: name={fe.get('name')!r} role={fe.get('role')!r}"]
            if fe.get("text"):
                t = fe["text"][:200]
                bits.append(f"text={t!r}")
            lines.append(" ".join(bits))
        if self.recent_apps:
            lines.append(f"recent_apps: {self.recent_apps[:5]}")
        if self.cursor_xy != (0, 0):
            lines.append(f"cursor: {self.cursor_xy}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class OSContextCollector:
    """Reads the live desktop state.

    All methods are best-effort: they degrade gracefully if a backend
    is missing. ``snapshot()`` always returns an :class:`OSContext`
    populated with whatever was reachable.
    """

    def __init__(self) -> None:
        self._have_sway = shutil.which("swaymsg") is not None
        self._have_xdo = shutil.which("xdotool") is not None
        self._have_wl = shutil.which("wl-paste") is not None
        self._have_ydotool = shutil.which("ydotool") is not None
        self._have_xdg = shutil.which("xdg-open") is not None

    def snapshot(self, screen_size: tuple[int, int] = (1920, 1080)) -> OSContext:
        ctx = OSContext(timestamp=time.time(), screen_size=screen_size)
        if self._have_sway:
            self._fill_sway(ctx)
        elif self._have_xdo:
            self._fill_xdo(ctx)
        ctx.cursor_xy = self._cursor_position()
        if self._have_wl:
            ctx.clipboard = self._clipboard()
        ctx.recent_apps = self._recent_apps()
        if HAVE_ATSPI:
            ctx.focused_element = self._focused_a11y()
        return ctx

    # -------------------------------------------------- sway / cosmic
    def _fill_sway(self, ctx: OSContext) -> None:
        try:
            tree = json.loads(
                subprocess.check_output(
                    ["swaymsg", "-t", "get_tree", "-r"], text=True, timeout=2
                )
            )
        except Exception:
            return
        # walk the tree, collect windows
        def walk(node, parent_geom=None):
            ntype = node.get("type")
            if ntype in ("con", "floating_con"):
                rect = node.get("window_rect") or node.get("rect", {})
                geom = {
                    "x": int(rect.get("x", 0)),
                    "y": int(rect.get("y", 0)),
                    "width": int(rect.get("width", 0)),
                    "height": int(rect.get("height", 0)),
                }
                if node.get("focused"):
                    ctx.active_window = WindowState(
                        title=str(node.get("name", "")),
                        app_id=node.get("app_id") or node.get("window_properties", {}).get("class"),
                        pid=node.get("pid"),
                        geometry=geom,
                        focused=True,
                    )
                if node.get("name") and node.get("pid"):
                    ctx.open_windows.append(WindowState(
                        title=str(node.get("name", "")),
                        app_id=node.get("app_id") or node.get("window_properties", {}).get("class"),
                        pid=node.get("pid"),
                        geometry=geom,
                        focused=bool(node.get("focused")),
                    ))
                for c in node.get("nodes", []):
                    walk(c, geom)
            else:
                for c in node.get("nodes", []):
                    walk(c, parent_geom)

        walk(tree)
        # working area (output the focused output)
        try:
            outs = json.loads(
                subprocess.check_output(
                    ["swaymsg", "-t", "get_outputs", "-r"], text=True, timeout=2
                )
            )
            for o in outs:
                if o.get("focused") or o.get("primary"):
                    rect = o.get("rect", {})
                    wa = o.get("workspace_rect", rect)
                    ctx.working_area = {
                        "x": int(wa.get("x", 0)),
                        "y": int(wa.get("y", 0)),
                        "width": int(wa.get("width", screen_size[0])),
                        "height": int(wa.get("height", screen_size[1])),
                    }
                    ctx.screen_size = (
                        int(rect.get("width", screen_size[0])),
                        int(rect.get("height", screen_size[1])),
                    )
                    break
        except Exception:
            pass

    # -------------------------------------------------- x11 fallback
    def _fill_xdo(self, ctx: OSContext) -> None:
        try:
            out = subprocess.check_output(
                ["xdotool", "getactivewindow", "getwindowname"], text=True, timeout=2
            ).strip()
            pid_s = subprocess.check_output(
                ["xdotool", "getactivewindow", "getwindowpid"], text=True, timeout=2
            ).strip()
            ctx.active_window = WindowState(title=out, pid=int(pid_s) or None)
        except Exception:
            pass

    # -------------------------------------------------- misc
    def _cursor_position(self) -> tuple[int, int]:
        # Try ydotool first (older ydotool versions don't have this; ignore failures)
        if self._have_ydotool:
            try:
                out = subprocess.check_output(
                    ["ydotool", "getmouselocation"], text=True, timeout=2,
                    stderr=subprocess.DEVNULL,
                )
                # "x:100 y:200" or similar
                parts = out.replace(",", " ").split()
                x, y = 0, 0
                for p in parts:
                    if p.startswith("x:"):
                        x = int(p[2:])
                    elif p.startswith("y:"):
                        y = int(p[2:])
                return (x, y)
            except Exception:
                pass
        # xdotool fallback
        if self._have_xdo:
            try:
                out = subprocess.check_output(["xdotool", "getmouselocation"], text=True, timeout=2)
                parts = out.replace(",", " ").split()
                x, y = 0, 0
                for p in parts:
                    if p.startswith("x:"):
                        x = int(p[2:])
                    elif p.startswith("y:"):
                        y = int(p[2:])
                return (x, y)
            except Exception:
                pass
        return (0, 0)

    def _clipboard(self) -> Optional[str]:
        if not self._have_wl:
            return None
        try:
            out = subprocess.check_output(["wl-paste", "--no-newline"], text=True, timeout=2)
            return out[:500]
        except subprocess.CalledProcessError:
            return None
        except Exception:
            return None

    def _recent_apps(self) -> list[str]:
        # gnome / cosmic keep recently-used in dconf. kde keeps in
        # ~/.local/share/RecentApplications. As a portable fallback we
        # also list the most-recent pids from /proc by start time.
        apps = []
        try:
            # Walk Recently used apps xbel (most desktops)
            xbel = Path.home() / ".local" / "share" / "recently-used.xbel"
            if xbel.is_file():
                import xml.etree.ElementTree as ET

                root = ET.parse(xbel).getroot()
                for bk in root.iter("bookmark"):
                    mime = bk.attrib.get("mime", "")
                    if "application" in mime or "exec" in bk.attrib:
                        exec_ = bk.attrib.get("exec", "")
                        if exec_:
                            apps.append(exec_.split()[0].split("/")[-1])
        except Exception:
            pass
        return apps[:5]

    def _focused_a11y(self) -> Optional[dict]:
        if not HAVE_ATSPI:
            return None
        try:
            desk = pyatspi.Registry.getDesktop(0)
        except Exception:
            return None
        try:
            focused = None

            def visit(acc):
                nonlocal focused
                try:
                    state = acc.getState()
                    if state.contains(pyatspi.STATE_FOCUSED):
                        try:
                            iface = acc.queryText()
                            text = iface.getText(0, iface.characterCount)
                        except Exception:
                            text = None
                        try:
                            role = acc.getRoleName()
                        except Exception:
                            role = None
                        try:
                            name = acc.name
                        except Exception:
                            name = None
                        try:
                            app = acc.getApplication()
                            app_name = app.name if app else None
                        except Exception:
                            app_name = None
                        focused = {
                            "name": name,
                            "role": role,
                            "text": text,
                            "app": app_name,
                        }
                except Exception:
                    pass
                for i in range(acc.childCount):
                    try:
                        visit(acc.getChildAtIndex(i))
                    except Exception:
                        pass

            for i in range(desk.childCount):
                try:
                    visit(desk.getChildAtIndex(i))
                except Exception:
                    pass
            return focused
        except Exception:
            return None
