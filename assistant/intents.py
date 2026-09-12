"""Cozy intent router: rule-based fast path + LLM hook for everything else.

Rules map common commands straight to tool calls (zero LLM latency).
Anything unmatched goes to the function-calling LLM once llm-agent's model
lands in assistant/llm_model/ (see llm.py stub).

Tool-call schema (shared with llm-agent — see team/channel.jsonl):
    {"tool": "open_app|set_volume|system_setting|browser_search|
              media_control|timer|query", "args": {...}, "_source": ...}
"""
import json
import sys
import re
from pathlib import Path

LLM_MODEL_DIR = Path(__file__).resolve().parent / "llm_model"


def route(text: str) -> dict:
    t = text.strip().lower()

    # wake-only / noise
    if re.fullmatch(r"(hey |hay )?cozy[.!?]?", t):
        return {"tool": "none", "args": {"reason": "wake_only"}, "_source": text}

    m = re.search(r"\b(?:what(?:'s| is)? the )?time\b|\btime (?:kya|kitna)\b", t)
    if m:
        return {"tool": "query_time", "args": {}, "_source": text}

    if re.search(r"\b(open|launch|start|khol|bring up)\b.*", t) and len(t.split()) <= 8:
        app = _app_name(t)
        if app:
            return {"tool": "open_app", "args": {"app": app}, "_source": text}

    if re.search(r"\b(close|quit|exit)\b.*", t) and len(t.split()) <= 8:
        app = _app_name(t)
        if app:
            return {"tool": "close_app", "args": {"app": app}, "_source": text}

    m = re.search(r"(?:set )?volume (?:to )?(\d{1,3})\s*(?:%|percent)?", t)
    if m:
        return {"tool": "set_volume", "args": {"level": min(100, int(m.group(1)))},
                "_source": text}
    m = re.search(r"\b(?:volume|speakers?)\b(?:\s+(?:to|at))?\s+([a-z -]+?)\s*(?:%|percent)\b", t)
    if m:
        level = _number_value(m.group(1))
        if level is not None:
            return {"tool": "set_volume", "args": {"level": min(100, level)},
                    "_source": text}
    if re.search(r"\b(volume (?:up|badha)|louder)\b", t):
        return {"tool": "set_volume", "args": {"delta": 10}, "_source": text}
    if re.search(r"\b(volume down|quieter|kam karo)\b", t):
        return {"tool": "set_volume", "args": {"delta": -10}, "_source": text}
    if re.search(r"\b(mute|silence)\b", t) and re.search(r"\b(volume|sound|audio|speaker|laptop|silence)\b", t):
        return {"tool": "mute_volume", "args": {}, "_source": text}

    m = re.search(r"\bbrightness\b(?:\s+(?:to|at))?\s+([a-z\d -]+?)(?:\s*(?:%|percent))?[.!?]?$", t)
    if m:
        level = _number_value(m.group(1))
        if level is not None:
            return {"tool": "set_brightness", "args": {"level": min(100, level)}, "_source": text}

    if re.search(r"\b(resume|play|continue)\b.*\b(music|song|media)\b|\bresume my music\b", t):
        return {"tool": "media_play", "args": {}, "_source": text}
    if re.search(r"\b(pause|hold|stop)\b.*\b(music|song|media)\b|\bhold the song\b", t):
        return {"tool": "media_pause", "args": {}, "_source": text}

    m = re.search(r"(?:search (?:for )?|google )(.+?)(?:[.?!]?$)", t)
    if m and re.search(r"\b(search|google|dhundo)\b", t):
        return {"tool": "browser_search", "args": {"q": m.group(1).strip()},
                "_source": text}

    if re.search(r"\b(screenshot)\b|\b(capture|grab)\b.*\b(screen|display)\b", t):
        return {"tool": "screenshot", "args": {}, "_source": text}

    if re.search(r"\b(today'?s date|date today|what(?:'s| is) the date)\b", t):
        return {"tool": "query_date", "args": {}, "_source": text}
    if re.search(r"\b(battery|charge)\b.*\b(status|level|left|remaining|percent)\b|\bbattery status\b", t):
        return {"tool": "battery_status", "args": {}, "_source": text}
    if re.search(r"\b(system )?uptime\b|\bhow long\b.*\b(running|on)\b", t):
        return {"tool": "system_uptime", "args": {}, "_source": text}

    # ---- LLM fallback hook -------------------------------------------
    if LLM_MODEL_DIR.exists():
        from llm import llm_route  # provided by llm-agent integration
        return llm_route(text)
    return {"tool": "unhandled", "args": {"text": text}, "_source": text}


_APPS = {
    "terminal": "terminal", "chrome": "chrome", "chromium": "chromium",
    "firefox": "firefox", "code": "code", "vs code": "code",
    "vscode": "code", "files": "nautilus", "file manager": "nautilus",
    "calculator": "gnome-calculator", "settings": "gnome-control-center",
    "spotify": "spotify", "vlc": "vlc", "telegram": "telegram",
    "whatsapp": "whatsapp", "calendar": "gnome-calendar",
}


def _app_name(t: str):
    best = None
    for phrase, app in _APPS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", t):
            if best is None or len(phrase) > len(best[0]):
                best = (phrase, app)
    return best[1] if best else None


def _number_value(value: str):
    """Parse a small spoken integer used by volume/brightness commands."""
    match = re.search(r"\d{1,3}", value)
    if match:
        return int(match.group())
    ones = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
            "fourteen": 14, "fifteen": 15, "sixteen": 16,
            "seventeen": 17, "eighteen": 18, "nineteen": 19}
    tens = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
            "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
    words = re.findall(r"[a-z]+", value)
    if len(words) == 1:
        return ones.get(words[0], tens.get(words[0]))
    if len(words) == 2 and words[0] in tens and words[1] in ones:
        return tens[words[0]] + ones[words[1]]
    return None


if __name__ == "__main__":
    print(json.dumps(route(" ".join(sys.argv[1:]) or input("> ")), indent=1))
