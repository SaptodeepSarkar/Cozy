#!/usr/bin/env python3
"""Cozy system executor - turns LLM tool calls into real actions."""
from __future__ import annotations

import datetime
import json
import shutil
import subprocess
import urllib.parse
from pathlib import Path


def _run(cmd, timeout=15):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
        ok = p.returncode == 0
        out = (p.stdout or p.stderr or "").strip()
        return ok, out[:400]
    except FileNotFoundError:
        return False, "missing binary: " + cmd[0]
    except subprocess.TimeoutExpired:
        return False, "timeout"


def _which_any(*names):
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def system_volume_set(params):
    level = max(0, min(100, int(params.get("level", 50))))
    ok, out = _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@",
                    str(level) + "%"])
    if not ok:
        ok, out = _run(["amixer", "-q", "sset", "Master", str(level) + "%"])
    return ok, ("volume " + str(level) + "%") if ok else out


def system_volume_mute(params=None):
    ok, out = _run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"])
    if not ok:
        ok, out = _run(["amixer", "-q", "sset", "Master", "toggle"])
    return ok, "muted/unmuted" if ok else out


def system_brightness_set(params):
    level = max(1, min(100, int(params.get("level", 50))))
    tool = _which_any("brightnessctl")
    if tool:
        return _run([tool, "set", str(level) + "%"])
    base = Path("/sys/class/backlight")
    if base.exists():
        for bl in base.iterdir():
            try:
                mx = int((bl / "max_brightness").read_text().strip())
                (bl / "brightness").write_text(str(int(mx * level / 100)))
                return True, "brightness " + str(level) + "%"
            except Exception:
                continue
    return False, "no backlight control found"


APP_ALIASES = {
    "browser": ["firefox", "google-chrome-stable", "chromium", "zen"],
    "chrome": ["google-chrome-stable", "chromium"],
    "files": ["nautilus"],
    "terminal": ["gnome-terminal", "kgx", "x-terminal-emulator"],
    "calculator": ["gnome-calculator"],
    "settings": ["gnome-control-center"],
    "notes": ["gnome-text-editor", "gedit"],
    "text editor": ["gnome-text-editor", "gedit"],
    "mail": ["thunderbird", "geary"],
    "calendar": ["gnome-calendar"],
    "camera": ["snapshot"],
}


def resolve_app(name):
    name_l = name.strip().lower()
    candidates = APP_ALIASES.get(name_l, [name_l])
    for cand in candidates:
        exe = shutil.which(cand)
        if exe:
            return [exe]
    flatpak = _which_any("flatpak")
    if flatpak:
        probe = subprocess.run([flatpak, "list", "--app",
                                "--columns=application"],
                               capture_output=True, text=True, timeout=10)
        for line in probe.stdout.splitlines():
            if name_l.replace(" ", "") in line.replace(".", "").lower():
                return [flatpak, "run", line.strip()]
    xdg = _which_any("xdg-open")
    return [xdg or "xdg-open", name]


def app_open(params):
    name = str(params.get("name", "")).strip()
    if not name:
        return False, "no app name"
    cmd = resolve_app(name)
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        return True, "opening " + name
    except Exception as exc:
        return False, "could not open " + name + ": " + str(exc)


def app_close(params):
    name = str(params.get("name", "")).strip().lower()
    for cand in APP_ALIASES.get(name, [name]):
        ok, out = _run(["pkill", "-f", cand])
        if ok:
            return True, "closed " + cand
    return False, "no running app matched " + name


def screenshot_take(params=None):
    out_dir = Path.home() / "Pictures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = str(out_dir / ("cozy_shot_" + str(int(__import__("time").time()))
                         + ".png"))
    tool = _which_any("gnome-screenshot")
    if tool:
        return _run([tool, "-f", out], timeout=20), out
    tool = _which_any("grim")
    if tool:
        return _run([tool, out], timeout=20), out
    tool = _which_any("scrot")
    if tool:
        return _run([tool, out], timeout=20), out
    tool = _which_any("import")
    if tool:
        return _run([tool, "-window", "root", out], timeout=20), out
    return False, "no screenshot tool"


def media_control(action, params=None):
    pc = _which_any("playerctl")
    if not pc:
        return False, "playerctl not installed"
    return _run([pc, action])


def window_minimize_all(params=None):
    wm = _which_any("wmctrl")
    if not wm:
        return False, "wmctrl not installed"
    return _run([wm, "-k", "on"])


def settings_open(params):
    page = str(params.get("page", "")).strip().lower()
    valid = {"wifi": "wifi", "bluetooth": "bluetooth", "display": "display",
             "sound": "sound", "notifications": "notifications",
             "power": "power"}
    gcc = _which_any("gnome-control-center")
    if not gcc:
        return False, "gnome-control-center missing"
    if page in valid:
        return _run([gcc, valid[page]])
    return _run([gcc])


def browser_search(params):
    q = urllib.parse.quote(str(params.get("query", "")))
    url = "https://www.google.com/search?q=" + q
    xdg = _which_any("xdg-open")
    if not xdg:
        return False, "xdg-open missing"
    subprocess.Popen([xdg, url], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
    return True, "searching: " + str(params.get("query", ""))


def browser_open_url(params):
    url = str(params.get("url", "")).strip()
    if not url.startswith("http"):
        url = "https://" + url
    xdg = _which_any("xdg-open")
    if not xdg:
        return False, "xdg-open missing"
    subprocess.Popen([xdg, url], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
    return True, url


def time_now(params=None):
    now = datetime.datetime.now()
    return True, now.strftime("%H:%M, %A %d %B %Y")


HANDLERS = {
    "system.volume.set": lambda p: system_volume_set(p),
    "system.volume.mute": system_volume_mute,
    "system.brightness.set": system_brightness_set,
    "app.open": app_open,
    "app.close": app_close,
    "screenshot.take": screenshot_take,
    "media.play": lambda p: media_control("play"),
    "media.pause": lambda p: media_control("pause"),
    "media.next": lambda p: media_control("next"),
    "media.previous": lambda p: media_control("previous"),
    "window.minimize_all": window_minimize_all,
    "settings.open": settings_open,
    "browser.search": browser_search,
    "browser.open_url": browser_open_url,
    "time.now": time_now,
}


def execute(tool_name, params=None):
    handler = HANDLERS.get(tool_name)
    if handler is None:
        return {"ok": False, "output": "unknown tool " + str(tool_name)}
    try:
        ok, output = handler(params or {})
        return {"ok": bool(ok), "output": output}
    except Exception as exc:
        return {"ok": False, "output": "error: " + str(exc)}
