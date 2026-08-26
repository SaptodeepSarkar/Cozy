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

    if re.search(r"\b(open|launch|start|khol)\b.*", t) and len(t.split()) <= 6:
        app = _app_name(t)
        if app:
            return {"tool": "open_app", "args": {"app": app}, "_source": text}

    m = re.search(r"(?:set )?volume (?:to )?(\d{1,3})\s*(?:%|percent)?", t)
    if m:
        return {"tool": "set_volume", "args": {"level": min(100, int(m.group(1)))},
                "_source": text}
    if re.search(r"\b(volume (?:up|badha)|louder)\b", t):
        return {"tool": "set_volume", "args": {"delta": 10}, "_source": text}
    if re.search(r"\b(volume down|quieter|kam karo)\b", t):
        return {"tool": "set_volume", "args": {"delta": -10}, "_source": text}

    m = re.search(r"(?:search (?:for )?|google )(.+?)(?:[.?!]?$)", t)
    if m and re.search(r"\b(search|google|dhundo)\b", t):
        return {"tool": "browser_search", "args": {"q": m.group(1).strip()},
                "_source": text}

    if re.search(r"\b(screenshot)\b", t):
        return {"tool": "screenshot", "args": {}, "_source": text}

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


if __name__ == "__main__":
    print(json.dumps(route(" ".join(sys.argv[1:]) or input("> ")), indent=1))
