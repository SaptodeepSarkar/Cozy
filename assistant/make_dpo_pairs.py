#!/usr/bin/env python3
"""Generate DPO preference pairs for Cozy.

Each pair has:
- prompt: list of {role, content} messages ending with a user turn
- chosen: the GOOD response (correct tool call or text reply)
- rejected: the BAD response (hallucinated tool, wrong args, wrong tool name, etc.)

Categories of negatives:
1. Wrong tool name (similar but invalid)
2. Hallucinated tool name (not in schema)
3. Empty arguments {} when params are required
4. Garbage parameter values (string when int expected, etc.)
5. Tool call when should be plain text (greetings)
6. Plain text when should be tool call
"""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

random.seed(1337)

SYSTEM = (
    "You are Cozy, a voice assistant running fully offline on the user's "
    "laptop. Respond fast and short. When the user wants an action, call "
    "exactly one tool with compact JSON. For plain chat, answer briefly "
    "and warmly without tools."
)

TOOLS = json.loads((ROOT / "team" / "tool_schema.json").read_text())["tools"]
TOOL_NAMES = [t["name"] for t in TOOLS]

# Things the SFT-trained model commonly hallucinates
HALLUCINATED_TOOLS = [
    "generate_random_fact", "get_random_fact", "send_email", "get_quote",
    "generate_quote", "generate_random_quote", "get_joke", "tell_joke",
    "calculate_bmi", "create_todo", "create_user_account", "translate_text",
    "generate_random_password", "get_news", "get_news_headlines",
    "get_weather", "play_music", "open_app", "close_app", "set_timer",
    "set_reminder", "set_alarm", "get_time", "get_date", "tell_joke",
    "search_web", "web_search", "google_search", "ask_ai", "chat",
    "compute_math", "do_math", "math", "calculate", "generate_id",
    "generate_uuid", "random_number", "roll_dice", "flip_coin",
    "get_battery", "battery_level", "wifi_status", "check_wifi",
    "mute_volume", "unmute_volume", "set_volume", "change_volume",
    "take_screenshot", "screenshot", "minimize_windows", "show_desktop",
    "system_shutdown", "shutdown", "reboot", "restart",
]


def make_tool_call(name: str, args: dict) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, separators=(", ", ": "))}
    }


def assistant_tool_call(name: str, args: dict) -> list[dict]:
    return [{"role": "assistant", "content": "", "tool_calls": [make_tool_call(name, args)]}]


def assistant_text(text: str) -> list[dict]:
    return [{"role": "assistant", "content": text}]


# ---------------------------------------------------------------------------
# Pair generators
# ---------------------------------------------------------------------------
def pair_correct_volume(prompt_text: str, level: int) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_tool_call("system.volume.set", {"level": level}),
        "rejected": assistant_tool_call("system.volume.set", {"level": "int 0-100"}),  # bad: string desc
    }


def pair_hallucinated_volume(prompt_text: str, level: int) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_tool_call("system.volume.set", {"level": level}),
        "rejected": assistant_tool_call("set_volume", {"level": level}),  # hallucinated name
    }


def pair_brightness(prompt_text: str, level: int) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_tool_call("system.brightness.set", {"level": level}),
        "rejected": assistant_tool_call("system.brightness.set", {"level": "0-100"}),
    }


def pair_app_open(prompt_text: str, app: str) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_tool_call("app.open", {"name": app}),
        "rejected": assistant_tool_call(random.choice(HALLUCINATED_TOOLS), {"name": app}),
    }


def pair_browser_search(prompt_text: str, query: str) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_tool_call("browser.search", {"query": query}),
        "rejected": assistant_tool_call("google_search", {"query": query}),
    }


def pair_browser_url(prompt_text: str, url: str) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_tool_call("browser.open_url", {"url": url}),
        "rejected": assistant_tool_call("browser.search", {"query": url}),  # wrong tool
    }


def pair_time(prompt_text: str) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_tool_call("time.now", {}),
        "rejected": assistant_tool_call("system.battery.status", {}),  # common confusion
    }


def pair_greeting(prompt_text: str, reply: str) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_text(reply),
        "rejected": assistant_tool_call("system.battery.status", {}),  # what the model currently does
    }


def pair_calc(prompt_text: str, expr: str) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_tool_call("calc.compute", {"expression": expr}),
        "rejected": assistant_text(f"the answer is {random.randint(1, 100)}"),
    }


def pair_timer(prompt_text: str, minutes: int) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_tool_call("timer.set", {"minutes": minutes}),
        "rejected": assistant_tool_call("timer.set", {}),  # missing required arg
    }


def pair_alarm(prompt_text: str, hour: int, minute: int) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_tool_call("alarm.set", {"hour": hour, "minute": minute}),
        "rejected": assistant_tool_call("alarm.set", {"hour": hour}),  # missing minute
    }


def pair_reminder(prompt_text: str, text: str, minutes: int) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_tool_call("reminder.set", {"text": text, "minutes": minutes}),
        "rejected": assistant_tool_call("reminder.set", {"text": text}),  # missing minutes
    }


def pair_note(prompt_text: str, text: str) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_tool_call("note.add", {"text": text}),
        "rejected": assistant_tool_call("note.read", {}),  # wrong direction
    }


def pair_clipboard_write(prompt_text: str, text: str) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_tool_call("clipboard.write", {"text": text}),
        "rejected": assistant_tool_call("clipboard.read", {}),  # wrong direction
    }


def pair_app_close(prompt_text: str, app: str) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_tool_call("app.close", {"name": app}),
        "rejected": assistant_tool_call("app.open", {"name": app}),  # wrong direction
    }


def pair_app_switch(prompt_text: str, app: str) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_tool_call("app.switch", {"name": app}),
        "rejected": assistant_tool_call("app.open", {"name": app}),  # different tool
    }


def pair_media(prompt_text: str, tool: str) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_tool_call(tool, {}),
        "rejected": assistant_tool_call(random.choice([t for t in
            ["media.play", "media.pause", "media.next", "media.previous"]
            if t != tool]), {}),
    }


def pair_battery(prompt_text: str) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_tool_call("system.battery.status", {}),
        "rejected": assistant_tool_call("system.battery.status", {"level": 10}),  # bad fake arg
    }


def pair_how_are_you(prompt_text: str) -> dict:
    replies = ["doing well, you?", "all good, thanks for asking", "great, you?"]
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt_text}],
        "chosen": assistant_text(random.choice(replies)),
        "rejected": assistant_tool_call("system.battery.status", {}),  # the actual bug
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pairs = []

    # Volume set - 80 pairs, mix of good vs bad-args
    for _ in range(60):
        level = random.randint(0, 100)
        phrasings = [f"set volume to {level}", f"volume {level}", f"change volume to {level}",
                     f"audio level {level}", f"speaker volume {level}", f"volume {level} karo"]
        pairs.append(pair_correct_volume(random.choice(phrasings), level))
    for _ in range(40):
        level = random.randint(0, 100)
        phrasings = [f"set volume to {level}", f"volume {level}"]
        pairs.append(pair_hallucinated_volume(random.choice(phrasings), level))

    # Brightness - 40
    for _ in range(40):
        level = random.randint(20, 100)
        phrasings = [f"set brightness to {level}", f"brightness {level}",
                     f"screen brightness {level}", f"brightness ko {level} karo"]
        pairs.append(pair_brightness(random.choice(phrasings), level))

    # App open - 80
    apps = ["firefox", "chrome", "terminal", "spotify", "vlc", "code", "settings",
            "calculator", "files", "discord", "telegram", "steam"]
    for _ in range(80):
        app = random.choice(apps)
        phrasings = [f"open {app}", f"launch {app}", f"start {app}", f"fire up {app}"]
        pairs.append(pair_app_open(random.choice(phrasings), app))

    # App close - 40
    for _ in range(40):
        app = random.choice(apps)
        phrasings = [f"close {app}", f"exit {app}", f"kill {app}"]
        pairs.append(pair_app_close(random.choice(phrasings), app))

    # App switch - 40
    for _ in range(40):
        app = random.choice(apps)
        phrasings = [f"switch to {app}", f"go to {app}", f"focus {app}"]
        pairs.append(pair_app_switch(random.choice(phrasings), app))

    # Browser search - 60
    queries = ["python tutorial", "weather today", "restaurants near me", "how to boil egg",
               "stock price NVDA", "kafka vs rabbitmq", "next cricket match",
               "cheap flights goa", "best laptop 50000", "manchester united score"]
    for _ in range(60):
        q = random.choice(queries)
        phrasings = [f"search {q}", f"google {q}", f"look up {q}", f"find {q}"]
        pairs.append(pair_browser_search(random.choice(phrasings), q))

    # Browser open url - 40
    urls = ["https://github.com", "https://youtube.com", "https://reddit.com",
            "https://news.ycombinator.com", "https://gmail.com"]
    for _ in range(40):
        url = random.choice(urls)
        phrasings = [f"open {url}", f"go to {url}", f"visit {url}"]
        pairs.append(pair_browser_url(random.choice(phrasings), url))

    # Time - 60
    for _ in range(60):
        phrasings = ["what time is it", "what's the time", "tell me the time",
                     "current time", "time please"]
        pairs.append(pair_time(random.choice(phrasings)))

    # Greetings - 80
    greetings = [
        ("hey cozy", ["hey!", "hi there", "hello!"]),
        ("hi", ["hi!", "hello", "hey"]),
        ("hello", ["hello!", "hi", "hey there"]),
        ("good morning", ["morning!", "good morning"]),
        ("how are you", ["doing well, you?", "great, thanks for asking"]),
        ("how are you doing", ["doing great, you?", "all good here"]),
        ("what's up", ["not much, you?", "hey, all good"]),
        ("thanks", ["you're welcome", "anytime"]),
        ("bye", ["bye!", "see you", "later"]),
        ("who are you", ["i'm cozy, your offline voice assistant", "cozy"]),
    ]
    for prompt_text, replies in greetings:
        for _ in range(8):
            pairs.append(pair_greeting(prompt_text, random.choice(replies)))

    # "how are you" specifically - the actual bug
    for _ in range(30):
        pairs.append(pair_how_are_you(random.choice([
            "how are you", "how are you doing", "how are you today", "how r u",
        ])))

    # Calc - 40
    for _ in range(40):
        expr = random.choice(["2+2", "10*5", "(2+3)*4", "100/4", "sqrt(16)", "2^10", "7*8"])
        phrasings = [f"what is {expr}", f"calculate {expr}", f"compute {expr}", f"{expr} = ?"]
        pairs.append(pair_calc(random.choice(phrasings), expr))

    # Timer - 40
    for _ in range(40):
        m = random.randint(1, 60)
        phrasings = [f"set a timer for {m} minutes", f"timer {m} minutes", f"{m} minute timer"]
        pairs.append(pair_timer(random.choice(phrasings), m))

    # Alarm - 30
    for _ in range(30):
        h, m = random.randint(0, 23), random.choice([0, 15, 30, 45])
        phrasings = [f"set alarm for {h}:{m:02d}", f"alarm at {h}:{m:02d}", f"wake me up at {h}:{m:02d}"]
        pairs.append(pair_alarm(random.choice(phrasings), h, m))

    # Reminder - 40
    notes = ["buy milk", "call mom", "finish PR by friday", "doctor appointment",
             "renew passport", "pay bill", "water the plants"]
    for _ in range(40):
        text = random.choice(notes)
        mins = random.randint(5, 60)
        phrasings = [f"remind me to {text} in {mins} minutes", f"set a reminder for {text}"]
        pairs.append(pair_reminder(random.choice(phrasings), text, mins))

    # Note - 30
    for _ in range(30):
        text = random.choice(notes)
        phrasings = [f"note: {text}", f"add note {text}", f"jot down {text}"]
        pairs.append(pair_note(random.choice(phrasings), text))

    # Clipboard write - 30
    for _ in range(30):
        text = random.choice(notes)
        phrasings = [f"copy {text} to clipboard", f"clipboard copy {text}"]
        pairs.append(pair_clipboard_write(random.choice(phrasings), text))

    # Media - 40
    for _ in range(40):
        tool = random.choice(["media.play", "media.pause", "media.next", "media.previous"])
        phrasings = {
            "media.play": ["play music", "play", "resume", "start the music"],
            "media.pause": ["pause", "pause music", "stop the music"],
            "media.next": ["next", "next song", "skip", "next track"],
            "media.previous": ["previous", "prev song", "go back"],
        }[tool]
        pairs.append(pair_media(random.choice(phrasings), tool))

    # Battery - 20
    for _ in range(20):
        phrasings = ["battery status", "battery percentage", "how much battery"]
        pairs.append(pair_battery(random.choice(phrasings)))

    random.shuffle(pairs)

    out_path = DATA / "dpo_pairs.jsonl"
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} DPO pairs -> {out_path}")


if __name__ == "__main__":
    main()
