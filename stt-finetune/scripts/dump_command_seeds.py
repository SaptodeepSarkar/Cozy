#!/usr/bin/env python3
"""Dump real user utterances (recordings/*/.txt) as function-calling seed
examples in the TEAM tool schema -> team/data/stt_command_seeds.jsonl
Format per line: {"text": ..., "tool": {"name": ..., "parameters": {...}}}
Run: .venv/bin/python scripts/dump_command_seeds.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RECORDINGS_DIR, ROOT  # noqa: E402

OUT = ROOT.parent / "team" / "data" / "stt_command_seeds.jsonl"

RULES = [
    (r"\btime\b(?! table)", lambda t: ("time.now", {})),
    (r"weather|temperature outside", lambda t: ("browser.search",
        {"query": re.sub(r".*?(?:like in|in|for)\s+", "", t).strip() or "weather"})),
    (r"^play (?:some )?(?:soft |chill )?music", lambda t: ("media.play", {})),
    (r"stop the music", lambda t: ("media.pause", {})),
    (r"volume down|quieter|kam karo", lambda t: ("system.volume.set", {"level": 25})),
    (r"volume up|louder|badha do", lambda t: ("system.volume.set", {"level": 75})),
    (r"volume (?:to )?(\d+)", lambda m_t: ("system.volume.set",
        {"level": int(re.search(r"volume (?:to )?(\d+)", m_t).group(1))})),
    (r"screenshot", lambda t: ("screenshot.take", {})),
    (r"\bsearch (?:for )?|google |dhundo", None),  # handled specially below
]


def classify(text: str):
    t = text.strip().lower()
    if re.search(r"\b(?:search|google)\b", t):
        m = re.search(r"(?:search for|search|google)\s+(.*)", t)
        return ("browser.search",
                {"query": (m.group(1) if m else t).strip(" ?.")})
    if re.search(r"\bweather\b|\btemperature\b|\bmausam\b", t):
        return ("browser.search", {"query": text.strip(" ?.")})
    if "time" in t and "timer" not in t and "times" not in t:
        return ("time.now", {})
    if re.search(r"volume", t):
        m = re.search(r"(\d{1,3})", t)
        if "down" in t or "kam" in t or "quieter" in t:
            return ("system.volume.set", {"level": 25})
        if "up" in t or "badha" in t or "louder" in t:
            return ("system.volume.set", {"level": 75})
        if m:
            return ("system.volume.set", {"level": min(100, int(m.group(1)))})
        return ("media.play", {})
    if t.startswith(("play ",)):
        return ("media.play", {})
    if "screenshot" in t:
        return ("screenshot.take", {})
    if re.search(r"\b(open|launch|khol)\b", t):
        words = re.sub(r"\b(open|launch|khol|the|and|please|cozy|hey)\b", " ", t)
        cand = words.split()[0] if words.split() else ""
        known = {"terminal", "chrome", "firefox", "code", "calculator",
                 "settings", "spotify", "vlc", "telegram", "files"}
        if cand in known:
            return ("app.open", {"name": cand})
        if "calendar" in t:
            return ("app.open", {"name": "calendar"})
    # everything else = plain conversation / no tool available yet
    return ("none", {})


def main():
    rows = []
    for sdir in sorted(RECORDINGS_DIR.glob("session_*")):
        for txt in sorted(sdir.glob("*.txt")):
            text = txt.read_text().strip()
            if not text:
                continue
            name, params = classify(text)
            rows.append({"text": text,
                         "tool": {"name": name, "parameters": params},
                         "session": sdir.name})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    c = Counter(r["tool"]["name"] for r in rows)
    print(f"wrote {len(rows)} seeds -> {OUT}")
    print("tool distribution:", dict(c))


if __name__ == "__main__":
    main()
