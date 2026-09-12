#!/usr/bin/env python3
"""Cozy system executor - turns LLM tool calls into real actions."""
from __future__ import annotations

import ast
import math
import operator
import shlex
import datetime
import json
import os
import signal
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
    if "level" not in params:
        return False, "volume level is required"
    level = max(0, min(100, int(params["level"])))
    f = level / 100.0
    # PipeWire (Arch default), then PulseAudio compat, then ALSA fallback.
    ok, out = _run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{f:.2f}"])
    if not ok:
        ok, out = _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@",
                        str(level) + "%"])
    if not ok:
        ok, out = _run(["amixer", "-q", "sset", "Master", str(level) + "%"])
    return ok, ("volume " + str(level) + "%") if ok else out


def system_volume_mute(params=None):
    ok, out = _run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"])
    if not ok:
        ok, out = _run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"])
    if not ok:
        ok, out = _run(["amixer", "-q", "sset", "Master", "toggle"])
    return ok, "muted/unmuted" if ok else out


def system_brightness_set(params):
    if "level" not in params:
        return False, "brightness level is required"
    level = max(1, min(100, int(params["level"])))
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
    "browser": ["firefox", "google-chrome-stable", "chromium", "zen", "brave"],
    "chrome": ["google-chrome-stable", "chromium", "brave"],
    "files": ["nautilus", "thunar", "dolphin", "nemo", "pcmanfm"],
    "terminal": ["gnome-terminal", "kgx", "x-terminal-emulator",
                 "konsole", "xfce4-terminal", "alacritty", "kitty", "foot"],
    "calculator": ["gnome-calculator", "kcalc", "qalculate-gtk"],
    "settings": ["gnome-control-center", "systemsettings",
                 "xfce4-settings-manager"],
    "notes": ["gnome-text-editor", "gedit", "kate", "mousepad", "xed"],
    "text editor": ["gnome-text-editor", "gedit", "kate", "mousepad", "xed"],
    "mail": ["thunderbird", "geary", "kmail", "claws-mail", "evolution"],
    "calendar": ["gnome-calendar", "korganizer", "calcurse"],
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
    return None


def app_open(params):
    name = str(params.get("name", "")).strip()
    if not name:
        return False, "no app name"
    cmd = resolve_app(name)
    if cmd is None:
        return False, "application not found: " + name
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        return True, "opening " + name
    except Exception as exc:
        return False, "could not open " + name + ": " + str(exc)


def app_close(params):
    name = str(params.get("name", "")).strip().lower()
    protected = {"python", "python3", "node", "bash", "sh", "zsh",
                 "systemd", "pipewire", "wireplumber", "hyprland",
                 "xwayland", "cozy"}
    for cand in APP_ALIASES.get(name, [name]):
        executable = Path(cand).name.lower()
        if not executable or executable in protected:
            continue
        ok, out = _run(["pgrep", "-x", executable])
        if not ok:
            continue
        pids = []
        for value in out.splitlines():
            try:
                pid = int(value)
            except ValueError:
                continue
            if pid not in (os.getpid(), os.getppid()):
                pids.append(pid)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                continue
        if pids:
            return True, "closed " + cand
    return False, "no running app matched " + name


def screenshot_take(params=None):
    out_dir = Path.home() / "Pictures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = str(out_dir / ("cozy_shot_" + str(int(__import__("time").time()))
                         + ".png"))
    # Try Wayland-friendly tools first, then GNOME, then X11.
    for cand in ("grim", "gnome-screenshot", "scrot", "import"):
        tool = _which_any(cand)
        if tool:
            if cand == "grim":
                ok, error = _run([tool, out], timeout=20)
                return (True, out) if ok else (False, error)
            if cand == "gnome-screenshot":
                ok, error = _run([tool, "-f", out], timeout=20)
                return (True, out) if ok else (False, error)
            if cand == "scrot":
                ok, error = _run([tool, out], timeout=20)
                return (True, out) if ok else (False, error)
            if cand == "import":
                ok, error = _run([tool, "-window", "root", out], timeout=20)
                return (True, out) if ok else (False, error)
    return False, "no screenshot tool (install grim)"


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
    # Try GNOME first, then KDE, then XFCE. Most distros ship one of these.
    candidates = [
        ("gnome-control-center", list(valid.values())),
        ("systemsettings", []),  # KDE: pages are KDE-specific, fall through to no-arg
        ("xfce4-settings-manager", []),
    ]
    for bin_name, pages in candidates:
        bin_path = _which_any(bin_name)
        if not bin_path:
            continue
        if bin_name == "gnome-control-center" and page in valid:
            return _run([bin_path, valid[page]])
        return _run([bin_path])
    return False, "no settings app found (install gnome-control-center, systemsettings, or xfce4-settings-manager)"


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


def _schedule_notification(when, title, message):
    """Only acknowledge a job after the OS scheduler accepts it."""
    at_bin = _which_any("at")
    if not at_bin:
        return False, "no scheduler available; install and start atd"
    notify = _which_any("notify-send")
    if not notify:
        return False, "notify-send is unavailable; install libnotify"
    body = shlex.join([notify, "-u", "critical", "--", title, message]) + "\n"
    ok, output = _run([at_bin, *when], input=body, timeout=5)
    return ok, output or ("notification scheduled" if ok else "scheduler rejected the job")


def timer_set(params):
    label = str(params.get("label") or "").strip()
    minutes = params.get("minutes")
    if minutes is None:
        minutes = _parse_duration_to_minutes(label)
    try:
        minutes = int(minutes)
        if not 1 <= minutes <= 180:
            return False, "timer duration must be between 1 and 180 minutes"
    except (TypeError, ValueError):
        return False, "could not understand duration"
    ok, output = _schedule_notification(
        ["now", "+", str(minutes), "minutes"], "Cozy timer", f"{label or 'Timer'} done")
    if not ok:
        return False, output
    return True, f"timer set{f' for {label}' if label else ''}, {minutes} minutes"


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
        if any(word in t for word in ("shaam", "raat", "saam")) and 1 <= h < 12:
            h += 12
        elif any(word in t for word in ("subah", "subha")) and h == 12:
            h = 0
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
    now = datetime.datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += datetime.timedelta(days=1)
    ok, output = _schedule_notification(
        ["-t", target.strftime("%Y%m%d%H%M")], "Cozy alarm", f"{label or 'Alarm'} ringing")
    return (True, f"alarm set for {hour:02d}:{minute:02d}") if ok else (False, output)


def reminder_set(params):
    text = str(params.get("text") or "").strip()
    if not text:
        return False, "nothing to remind about"
    minutes = params.get("minutes")
    if minutes is None:
        minutes = _parse_duration_to_minutes(text)
    try:
        minutes = int(minutes)
        if not 1 <= minutes <= 525600:
            return False, "reminder duration must be between 1 minute and 1 year"
    except (TypeError, ValueError):
        return False, "could not understand duration"
    ok, output = _schedule_notification(
        ["now", "+", str(minutes), "minutes"], "Cozy reminder", text)
    if not ok:
        return False, output
    return True, f"reminder set for {minutes} minutes from now: {text}"


def date_now(params=None):
    now = datetime.datetime.now()
    return True, now.strftime("%A, %d %B %Y")


# ---- system control ----
def system_lock(params=None):
    # loginctl works on every systemd login session (Arch + Debian).
    # Then Wayland compositors, then GNOME, then X11 fallback.
    for cmd in (["loginctl", "lock-session"],
                ["swaylock", "-f"],
                ["hyprlock"],
                ["waylock"],
                ["gnome-screensaver-command", "-l"],
                ["xdg-screensaver", "lock"]):
        if _which_any(cmd[0]):
            ok, out = _run(cmd, timeout=5)
            if ok:
                return True, "screen locked"
    return False, "no screen lock tool found (install swaylock, gnome-screensaver, or xdg-utils)"


def system_shutdown(params):
    # A model-generated parameter is not user authorization. Power actions are
    # unavailable unless the user explicitly opts in before launching Cozy.
    if os.environ.get("COZY_ALLOW_POWER_ACTIONS") != "1":
        return False, "shutdown is disabled; set COZY_ALLOW_POWER_ACTIONS=1 before launch to opt in"
    if params.get("confirm") is not True:
        return False, "shutdown requires the boolean confirm=true"
    delay = max(0, min(24 * 60, int(params.get("delay_minutes") or 0)))
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
    # Prefer /sys/class/power_supply first - upower on desktops/headless
    # boxes reports a phantom "0% (should be ignored)" stub that lies to users.
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
    # Fallback: upower (only on real batteries, skip the phantom stub)
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
                if pct and "should be ignored" not in pct.lower() \
                        and "should be ignored" not in state.lower():
                    return True, f"battery at {pct}{' (' + state + ')' if state else ''}"
        except Exception:
            pass
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
    if len(expr) > 256:
        return False, "expression is too long (maximum 256 characters)"
    operations = {ast.Add: operator.add, ast.Sub: operator.sub,
                  ast.Mult: operator.mul, ast.Div: operator.truediv,
                  ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
                  ast.Pow: operator.pow}

    def bounded(value):
        if type(value) not in (int, float) or abs(value) > 10 ** 100:
            raise ValueError("result exceeds calculator limits")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("result must be finite")
        return value

    def evaluate(node):
        if isinstance(node, ast.Constant):
            return bounded(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return bounded(-value if isinstance(node.op, ast.USub) else value)
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent exceeds calculator limit (100)")
            return bounded(operations[type(node.op)](left, right))
        raise ValueError("only numeric arithmetic is supported")

    try:
        tree = ast.parse(expr, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            raise ValueError("expression is too complex")
        result = evaluate(tree.body)
    except (ValueError, TypeError, SyntaxError, ArithmeticError, RecursionError) as exc:
        return False, "could not evaluate: " + str(exc)
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


from system_tools import HANDLERS as SYSTEM_HANDLERS
HANDLERS.update(SYSTEM_HANDLERS)

def execute(tool_name, params=None):
    handler = HANDLERS.get(tool_name)
    if handler is None:
        return {"ok": False, "output": "unknown tool " + str(tool_name)}
    try:
        ok, output = handler(params or {})
        return {"ok": bool(ok), "output": output}
    except Exception as exc:
        return {"ok": False, "output": "error: " + str(exc)}
