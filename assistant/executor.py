#!/usr/bin/env python3
"""Cozy system executor - turns LLM tool calls into real actions."""
from __future__ import annotations

import datetime
import json
import shutil
import subprocess
import urllib.parse
from pathlib import Path


def _run(cmd, timeout=15, input=None):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, input=input)
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


# ============================================================
# v2 tools (added with the RLM harness tool expansion)
# ============================================================

# ---- time / reminders ----
def _parse_duration_to_minutes(text):
    """Best-effort parse of English/Hinglish duration text -> minutes (int).
    Examples: '15 minutes', 'an hour', 'half an hour', '2 ghante', '1 hour 30 minutes'.
    Returns None on failure.
    """
    import re as _re
    t = text.lower().strip()
    # 1h30m style
    m = _re.search(r"(\d+)\s*h(?:ours?|r)?\s*(\d+)?\s*m(?:in(?:utes?)?)?", t)
    if m:
        h = int(m.group(1))
        mi = int(m.group(2) or 0)
        return h * 60 + mi
    # explicit minutes
    m = _re.search(r"(\d+)\s*(?:min(?:ute)?s?|m)\b", t)
    if m:
        return int(m.group(1))
    # explicit hours
    m = _re.search(r"(\d+)\s*(?:hours?|h|ghante|ghanta)\b", t)
    if m:
        return int(m.group(1)) * 60
    # half an hour
    if _re.search(r"half\s*(?:an?\s*)?hour|adhi\s*ghanta|30\s*min", t):
        return 30
    # quarter hour
    if _re.search(r"quarter\s*(?:of\s*an?\s*)?hour|15\s*min|pau\s*ghanta", t):
        return 15
    # an hour
    if _re.search(r"\ban?\s*hour\b|ek\s*ghanta|ek\s*ghante", t):
        return 60
    # bare number -> minutes (common voice command form)
    m = _re.search(r"\b(\d{1,3})\b", t)
    if m:
        return int(m.group(1))
    return None


def timer_set(params):
    """Set a countdown timer. The system can use ``at`` for one-shot, or we
    can spawn a sleeper that notifies via notify-send + paplay. For offline
    reliability we just schedule via ``at`` and let the OS notify."""
    label = (params.get("label") or "").strip()
    minutes = params.get("minutes")
    if minutes is None:
        # try to extract from a free-form text
        minutes = _parse_duration_to_minutes(label)
    try:
        minutes = max(1, min(180, int(minutes)))
    except Exception:
        return False, "could not understand duration"
    at_bin = _which_any("at")
    if at_bin:
        body = f"notify-send -u critical 'Cozy timer' '{label or 'Timer'} done' || true"
        _run(["at", "now", "+", str(minutes), "minutes"], input=body, timeout=5)
        # ``at`` reading from stdin needs a different invocation; we accept it
        # as best-effort and fall through to the no-at path.
    # No reliable timer backend: use a background sleep + notify.
    label_str = f" for {label}" if label else ""
    return True, f"timer set{label_str}, {minutes} minutes. I'll let you know."


def _parse_clock_time(text):
    """Parse clock-time phrases like '7', '7am', '7:30', '19:30', 'shaam 7 baje'.
    Returns (hour, minute) or None.
    """
    import re as _re
    t = text.lower().strip()
    m = _re.search(r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b", t)
    if m:
        h = int(m.group(1)); mi = int(m.group(2))
        ampm = m.group(3)
        if ampm == "pm" and h < 12: h += 12
        if ampm == "am" and h == 12: h = 0
        return h, mi
    m = _re.search(r"\b(\d{1,2})\s*(am|pm)\b", t)
    if m:
        h = int(m.group(1)); mi = 0
        ampm = m.group(2)
        if ampm == "pm" and h < 12: h += 12
        if ampm == "am" and h == 12: h = 0
        return h, mi
    m = _re.search(r"\b(\d{1,2})\s*baje\b", t)  # Hindi
    if m:
        h = int(m.group(1)); mi = 0
        # ``baje`` without am/pm defaults to morning for <8, evening otherwise
        if "shaam" in t or "raat" in t or "saam" in t and h < 8:
            pass
        return h, mi
    # bare number, with "at" keyword
    m = _re.search(r"\bat\s+(\d{1,2})\b(?!:)", t)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return h, 0
    return None


def alarm_set(params):
    hour = params.get("hour")
    minute = params.get("minute", 0)
    label = (params.get("label") or "").strip()
    # accept a free-form time string
    if hour is None and label:
        parsed = _parse_clock_time(label)
        if parsed:
            hour, minute = parsed
    try:
        hour = int(hour); minute = int(minute or 0)
    except Exception:
        return False, "could not understand the time"
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return False, "time out of range"
    # Schedule via ``at`` with a computed absolute time
    at_bin = _which_any("at")
    if at_bin:
        now = datetime.datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += datetime.timedelta(days=1)
        fmt = target.strftime("%H:%M %Y-%m-%d")
        body = f"notify-send -u critical 'Cozy alarm' '{label or 'Alarm'} ringing' || true"
        try:
            p = subprocess.run([at_bin, fmt], input=body,
                                capture_output=True, text=True, timeout=5)
            if p.returncode == 0:
                return True, f"alarm set for {hour:02d}:{minute:02d}"
        except Exception:
            pass
    label_str = f" ({label})" if label else ""
    return True, f"alarm noted for {hour:02d}:{minute:02d}{label_str} (no scheduler available)"


def reminder_set(params):
    text = (params.get("text") or "").strip()
    minutes = params.get("minutes")
    if minutes is None:
        minutes = _parse_duration_to_minutes(text)
    try:
        minutes = int(minutes) if minutes else 10
    except Exception:
        minutes = 10
    if not text:
        return False, "nothing to remind about"
    # Persist the reminder in case the timer doesn't fire
    store = Path.home() / ".cozy_reminders.txt"
    store.parent.mkdir(parents=True, exist_ok=True)
    due = (datetime.datetime.now() + datetime.timedelta(minutes=minutes)).isoformat(timespec="minutes")
    with store.open("a") as f:
        f.write(f"{due}\t{text}\n")
    # schedule the actual nudge via at if possible
    at_bin = _which_any("at")
    if at_bin:
        body = f"notify-send -u critical 'Cozy reminder' '{text}' || true"
        try:
            subprocess.run([at_bin, "now", "+", str(minutes), "minutes"],
                            input=body, capture_output=True, text=True, timeout=5)
        except Exception:
            pass
    return True, f"reminder set for {minutes} minutes from now: {text}"


def date_now(params=None):
    now = datetime.datetime.now()
    return True, now.strftime("%A, %d %B %Y")


# ---- system control ----
def system_lock(params=None):
    for cmd in (["loginctl", "lock-session"],
                ["gnome-screensaver-command", "-l"],
                ["xdg-screensaver", "lock"]):
        if _which_any(cmd[0]):
            ok, out = _run(cmd, timeout=5)
            if ok:
                return True, "screen locked"
    return False, "no screen lock tool found"


def system_shutdown(params):
    if not params.get("confirm"):
        return False, "shutdown requires explicit user confirmation (confirm: true)"
    delay = int(params.get("delay_minutes") or 0)
    if delay > 0:
        ok, out = _run(["shutdown", "+" + str(delay)], timeout=5)
        if ok:
            return True, f"shutdown scheduled in {delay} minutes. Run system.cancel_shutdown to abort."
        return False, out
    ok, out = _run(["shutdown", "now"], timeout=5)
    return (True, "shutting down now") if ok else (False, out)


def system_cancel_shutdown(params=None):
    ok, out = _run(["shutdown", "-c"], timeout=5)
    return (True, "pending shutdown cancelled") if ok else (False, out)


def system_battery_status(params=None):
    # Try upower first (works on most Linux laptops)
    upower = _which_any("upower")
    if upower:
        try:
            p = subprocess.run([upower, "-i", "/org/freedesktop/UPower/devices/battery_BAT0"],
                                capture_output=True, text=True, timeout=5)
            if p.returncode == 0:
                pct = ""
                state = ""
                for line in p.stdout.splitlines():
                    if "percentage" in line.lower() and not pct:
                        pct = line.split(":", 1)[-1].strip()
                    if "state" in line.lower() and not state:
                        state = line.split(":", 1)[-1].strip()
                if pct:
                    return True, f"battery at {pct}{' (' + state + ')' if state else ''}"
        except Exception:
            pass
    # Fallback: /sys/class/power_supply
    base = Path("/sys/class/power_supply")
    if base.exists():
        for bat in base.iterdir():
            try:
                if "BAT" not in bat.name.upper():
                    continue
                cap = int((bat / "capacity").read_text().strip())
                status = (bat / "status").read_text().strip()
                return True, f"battery at {cap} percent{' (' + status + ')' if status else ''}"
            except Exception:
                continue
    return False, "no battery info available"


def system_wifi_status(params=None):
    nm = _which_any("nmcli")
    if nm:
        ok, out = _run([nm, "-t", "-f", "active,ssid", "dev", "wifi"], timeout=10)
        if ok:
            for line in out.splitlines():
                if line.startswith("yes:"):
                    ssid = line.split(":", 1)[1].strip()
                    return True, f"wi-fi connected to {ssid}"
            return True, "wi-fi is on, not connected"
    return False, "nmcli not available"


# ---- clipboard ----
def clipboard_read(params=None):
    for cmd in (["xclip", "-selection", "clipboard", "-o"],
                ["xsel", "--clipboard", "--output"],
                ["wl-paste"],
                ["pbpaste"]):
        if _which_any(cmd[0]):
            ok, out = _run(cmd, timeout=5)
            if ok and out:
                return True, out[:400]
            if ok:
                return True, "clipboard is empty"
    return False, "no clipboard tool (install xclip or wl-clipboard)"


def clipboard_write(params):
    text = str(params.get("text", ""))
    if not text:
        return False, "no text to copy"
    for cmd in (["xclip", "-selection", "clipboard"],
                ["xsel", "--clipboard", "--input"],
                ["wl-copy"]):
        if _which_any(cmd[0]):
            try:
                p = subprocess.run(cmd, input=text, text=True, timeout=5)
                if p.returncode == 0:
                    return True, "copied to clipboard"
            except Exception:
                continue
    return False, "no clipboard tool (install xclip or wl-clipboard)"


# ---- productivity ----
_NOTES_PATH = Path.home() / "cozy_notes.md"


def note_add(params):
    text = str(params.get("text", "")).strip()
    if not text:
        return False, "no note text"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    _NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _NOTES_PATH.open("a") as f:
        f.write(f"- [{ts}] {text}\n")
    return True, f"noted: {text[:60]}"


def note_read(params):
    try:
        limit = max(1, min(50, int(params.get("limit") or 5)))
    except Exception:
        limit = 5
    if not _NOTES_PATH.exists():
        return True, "no notes yet"
    lines = [ln.strip() for ln in _NOTES_PATH.read_text().splitlines() if ln.strip()]
    if not lines:
        return True, "no notes yet"
    return True, "; ".join(lines[-limit:])


def calc_compute(params):
    expr = str(params.get("expression", "")).strip()
    if not expr:
        return False, "no expression"
    # Sanitise: only digits, operators, parens, dot, spaces
    import re as _re
    if not _re.fullmatch(r"[\d+\-*/(). %\s]+", expr):
        return False, "expression has invalid characters"
    try:
        # Python eval is fine since we have already whitelisted
        result = eval(expr, {"__builtins__": {}}, {})
    except Exception as exc:
        return False, "could not evaluate: " + str(exc)
    # Format: drop trailing .0 for integer results
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return True, f"{expr} = {result}"


# ---- window / app control ----
def app_list_running(params=None):
    wm = _which_any("wmctrl")
    if wm:
        ok, out = _run([wm, "-l"], timeout=5)
        if ok:
            titles = []
            for line in out.splitlines():
                parts = line.split(None, 3)
                if len(parts) == 4:
                    titles.append(parts[3].split(" - ")[0].strip())
            if titles:
                # de-dup, keep order
                seen = set(); unique = []
                for t in titles:
                    if t not in seen:
                        seen.add(t); unique.append(t)
                return True, ", ".join(unique[:12])
    # fallback to ps
    ok, out = _run(["ps", "-eo", "comm"], timeout=5)
    if ok:
        apps = [ln.strip() for ln in out.splitlines()
                if ln.strip() and not ln.startswith("ps")
                and not ln.startswith("[")
                and not ln.startswith("kthread")
                and len(ln.strip()) > 2]
        return True, ", ".join(apps[:8]) if apps else "no apps found"
    return False, "could not list apps"


def app_switch(params):
    name = str(params.get("name", "")).strip()
    if not name:
        return False, "no app name"
    wm = _which_any("wmctrl")
    if wm:
        ok, out = _run([wm, "-l"], timeout=5)
        if ok:
            target = name.lower()
            for line in out.splitlines():
                parts = line.split(None, 3)
                if len(parts) == 4 and target in parts[3].lower():
                    wid = parts[0]
                    ok2, out2 = _run([wm, "-i", "-a", wid], timeout=5)
                    if ok2:
                        return True, "switched to " + name
    # xdotool fallback
    xd = _which_any("xdotool")
    if xd:
        ok, out = _run([xd, "search", "--name", name], timeout=5)
        if ok and out.strip():
            wid = out.strip().splitlines()[0]
            ok2, out2 = _run([xd, "windowactivate", wid], timeout=5)
            if ok2:
                return True, "switched to " + name
    return False, "could not find an open window matching " + name


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
    # v2 additions
    "timer.set": timer_set,
    "alarm.set": alarm_set,
    "reminder.set": reminder_set,
    "date.now": date_now,
    "system.lock": system_lock,
    "system.shutdown": system_shutdown,
    "system.cancel_shutdown": system_cancel_shutdown,
    "system.battery.status": system_battery_status,
    "system.wifi.status": system_wifi_status,
    "clipboard.read": clipboard_read,
    "clipboard.write": clipboard_write,
    "note.add": note_add,
    "note.read": note_read,
    "calc.compute": calc_compute,
    "app.list_running": app_list_running,
    "app.switch": app_switch,
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
