#!/usr/bin/env python3
"""Generate a clean, large SFT dataset for Cozy from the tool schema.

Strategy:
- Use only the 30+ tools declared in team/tool_schema.json.
- For each tool, generate 50-200 prompt/assistant pairs with varied phrasing.
- Arguments are filled in with realistic values (not empty {}).
- Add a large social/greeting block so the model learns when to NOT call a tool.
- Add "none" examples (plain chat → no tool call, short text reply).
- Save to data/sft_train.jsonl and data/sft_val.jsonl (90/10 split).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

random.seed(42)

SYSTEM = (
    "You are Cozy, a voice assistant running fully offline on the user's "
    "laptop. Respond fast and short. When the user wants an action, call "
    "exactly one tool with compact JSON. For plain chat, answer briefly "
    "and warmly without tools."
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
TOOLS = json.loads((ROOT / "team" / "tool_schema.json").read_text())["tools"]
TOOL_INDEX = {t["name"]: t for t in TOOLS}


def tools_for(only: list[str] | None = None) -> list[dict]:
    if only is None:
        return TOOLS
    return [t for t in TOOLS if t["name"] in only]


# ---------------------------------------------------------------------------
# Phrase banks
# ---------------------------------------------------------------------------
APPS = [
    "firefox", "chrome", "chromium", "terminal", "konsole", "files", "nautilus",
    "spotify", "vlc", "code", "vscode", "thunderbird", "gimp", "obs",
    "calculator", "kcalc", "settings", "telegram", "discord", "steam",
    "libreoffice", "writer", "impress", "pomodoro", "keepassxc", "signal",
]
SETTINGS_PAGES = ["wifi", "bluetooth", "display", "sound", "network", "power", "privacy", "keyboard", "mouse", "appearance"]
BROWSER_QUERIES = [
    "best italian restaurants near me", "weather today", "python tutorial",
    "github status", "openai news", "how to boil an egg", "stock price of NVDA",
    "next cricket match", "traffic to airport", "cheap flights to goa",
    "kafka vs rabbitmq", "linux kernel latest", "elon musk latest tweet",
    "manchester united score", "what is rsi", "best laptop under 50000",
    "translate hello to french", "define serendipity",
]
BROWSER_URLS = [
    "https://news.ycombinator.com", "https://github.com", "https://reddit.com",
    "https://youtube.com", "https://gmail.com", "https://calendar.google.com",
    "https://wikipedia.org", "https://twitter.com", "https://linkedin.com",
    "https://stackoverflow.com", "https://archlinux.org", "https://duckduckgo.com",
]
NOTES = [
    "buy milk", "call mom", "finish the cozy PR by friday", "doctor appointment tuesday",
    "renew passport", "pay electricity bill", "fix kitchen sink", "water the plants",
    "send birthday card to arjun", "review the design doc", "book train tickets",
    "pickup laundry", "schedule dentist", "groceries: tomato, paneer, rice",
    "read chapter 4 of the book", "pay credit card", "renew domain",
    "send invoice to client", "follow up with hr", "water the garden",
]
LABELS = ["pasta", "workout", "laundry", "study", "meeting", "meds", "pomodoro", "yoga", "break"]
TITLES = ["team standup", "doctor", "design review", "1:1 with manager", "call mom", "gym", "yoga", "interview"]
MATH = [
    ("2 + 2", "4"), ("10 * 5", "50"), ("(2+3)*4", "20"), ("100/4", "25"),
    ("sqrt(16)", "4"), ("2^10", "1024"), ("7 * 8", "56"), ("3.14 * 10", "31.4"),
    ("(100-32)*5/9", "37.777..."), ("sin(0)", "0"),
]
GREETINGS = [
    ("hey cozy", ["hey!", "hi there", "hello!", "hey, what's up"]),
    ("hi", ["hi!", "hello", "hey", "hey there"]),
    ("hello", ["hello!", "hi", "hey there", "hi!"]),
    ("good morning", ["morning!", "good morning", "hey, morning", "morning!"]),
    ("good afternoon", ["afternoon!", "hey, good afternoon", "hi"]),
    ("good evening", ["evening!", "good evening", "hey"]),
    ("good night", ["night!", "good night", "sleep well", "night"]),
    ("how are you", ["doing well, you?", "great, thanks for asking", "all good, you?", "pretty good"]),
    ("how are you doing", ["doing great, you?", "all good here", "fine, you?"]),
    ("how's it going", ["going well, you?", "good, you?", "all good"]),
    ("what's up", ["not much, you?", "hey, all good", "chilling"]),
    ("yo", ["yo", "hey", "hiya"]),
    ("thanks", ["you're welcome", "anytime", "no problem", "sure thing"]),
    ("thank you", ["you're welcome", "anytime", "happy to help"]),
    ("thanks a lot", ["you're welcome", "anytime", "no worries"]),
    ("bye", ["bye!", "see you", "later", "bye bye"]),
    ("goodbye", ["goodbye!", "see you", "later", "bye"]),
    ("see you", ["see you!", "later", "bye", "take care"]),
    ("who are you", ["i'm cozy, your offline voice assistant", "cozy, running on your laptop"]),
    ("what's your name", ["cozy", "i'm cozy"]),
    ("what can you do", ["i can open apps, set timers, take notes, search the web, control volume and brightness, and chat with you"]),
    ("help", ["sure - try 'open firefox', 'set a 10 minute timer', 'volume 50', 'what time is it', or 'how are you'"]),
    ("i love you", ["aww, thanks", "that's sweet", "love you too"]),
    ("i'm sad", ["sorry to hear that, want to talk?", "i'm here if you need anything"]),
    ("i'm happy", ["nice! glad to hear it", "love that"]),
    ("i'm bored", ["want me to open spotify, or set a pomodoro?", "i can search the web or tell you a joke"]),
    ("i'm tired", ["maybe take a break? want me to set a reminder?", "rest up"]),
    ("i'm hungry", ["want me to open doordash or search for recipes?", "sounds like a snack break"]),
    ("good job", ["thanks!", "appreciate it", "cheers"]),
    ("nice", ["thanks", "cheers"]),
    ("cool", ["glad you like it", "yep"]),
    ("you're awesome", ["thanks!", "you're not bad yourself", "cheers"]),
]

# ---------------------------------------------------------------------------
# Tool-call generators
# ---------------------------------------------------------------------------
def make_tool_call(name: str, args: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, separators=(", ", ": ")),
        },
    }


def example(user: str, tool_name: str, args: dict, reply: str = "On it.", tool_reply: str = "OK: OK") -> dict:
    """Build a single training row."""
    tools = [t for t in TOOLS if t["name"] == tool_name] or [{"name": tool_name, "params": args, "desc": ""}]
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": "", "tool_calls": [make_tool_call(tool_name, args)]},
    ]
    if tool_reply:
        msgs.append({"role": "tool", "name": tool_name, "content": tool_reply})
    msgs.append({"role": "assistant", "content": reply})
    return {"messages": msgs, "tools": tools}


def none_example(user: str, reply: str) -> dict:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": reply},
    ]
    return {"messages": msgs, "tools": TOOLS}


# ---------------------------------------------------------------------------
# Phrase templates
# ---------------------------------------------------------------------------
def varied(phrasings: list[str], *filler) -> str:
    return random.choice(phrasings).format(*filler)


def num(n: int) -> str:
    """Render n as a number, with occasional word form."""
    if random.random() < 0.7:
        return str(n)
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
             "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred"]
    if 0 <= n <= 100 and random.random() < 0.5:
        return words[n] if n < len(words) else str(n)
    return str(n)


# ---------------------------------------------------------------------------
# Generators per tool
# ---------------------------------------------------------------------------
def gen_volume_set(n: int) -> dict:
    level = random.randint(0, 100)
    user = varied([
        f"set volume to {level}", f"volume {level}", f"change volume to {level}",
        f"make it louder, set volume to {level}", f"lower volume to {level}",
        f"set sound to {level}", f"volume {level} please", f"master volume {level}",
        f"can you set volume to {level}", f"audio level {level}", f"speaker volume {level}",
        f"volume ko {level} karo", f"volume {level} kar do", f"awaaz {level} karo",
    ])
    return example(user, "system.volume.set", {"level": level})


def gen_volume_mute(n: int) -> dict:
    user = varied([
        "mute", "mute the volume", "unmute", "toggle mute", "silence the audio",
        "mute please", "can you mute", "sab mute karo", "awaaz band karo",
    ])
    return example(user, "system.volume.mute", {})


def gen_brightness_set(n: int) -> dict:
    level = random.randint(10, 100)
    user = varied([
        f"set brightness to {level}", f"brightness {level}", f"screen brightness {level}",
        f"make the screen {level}", f"dim to {level}", f"display brightness {level}",
        f"can you set brightness to {level}", f"brightness ko {level} karo",
    ])
    return example(user, "system.brightness.set", {"level": level})


def gen_app_open(n: int) -> dict:
    app = random.choice(APPS)
    user = varied([
        f"open {app}", f"launch {app}", f"start {app}", f"can you open {app}",
        f"i need {app}", f"open up {app}", f"run {app}", f"fire up {app}",
        f"{app} kholo", f"{app} open karo",
    ])
    return example(user, "app.open", {"name": app})


def gen_app_close(n: int) -> dict:
    app = random.choice(APPS)
    user = varied([
        f"close {app}", f"kill {app}", f"shut {app}", f"exit {app}",
        f"can you close {app}", f"close the {app} window", f"band karo {app}",
    ])
    return example(user, "app.close", {"name": app})


def gen_screenshot(n: int) -> dict:
    user = varied([
        "take a screenshot", "screenshot", "snap the screen",
        "capture the screen", "screenshot please", "take screenshot",
        "screen capture", "can you screenshot this",
    ])
    return example(user, "screenshot.take", {})


def gen_media_play(n: int) -> dict:
    return example(varied(["play music", "resume", "play", "start the music", "play song", "chalao"]), "media.play", {})


def gen_media_pause(n: int) -> dict:
    return example(varied(["pause", "pause music", "stop the music", "hold on", "ruk jao"]), "media.pause", {})


def gen_media_next(n: int) -> dict:
    return example(varied(["next", "next song", "skip", "next track", "agle pe jao"]), "media.next", {})


def gen_media_previous(n: int) -> dict:
    return example(varied(["previous", "prev song", "go back", "previous track", "pichla gaana"]), "media.previous", {})


def gen_minimize_all(n: int) -> dict:
    return example(varied([
        "minimize everything", "show desktop", "minimize all windows",
        "clear the screen", "minimize all", "desktop dikhao",
    ]), "window.minimize_all", {})


def gen_settings_open(n: int) -> dict:
    page = random.choice(SETTINGS_PAGES)
    user = varied([
        f"open {page} settings", f"{page} settings", f"settings {page}",
        f"open the {page} page", f"go to {page} settings", f"show {page} settings",
    ])
    return example(user, "settings.open", {"page": page})


def gen_browser_search(n: int) -> dict:
    q = random.choice(BROWSER_QUERIES)
    user = varied([
        f"search {q}", f"google {q}", f"look up {q}", f"find {q}",
        f"web search {q}", f"can you search for {q}", f"{q} search karo",
    ])
    return example(user, "browser.search", {"query": q})


def gen_browser_open_url(n: int) -> dict:
    url = random.choice(BROWSER_URLS)
    user = varied([
        f"open {url}", f"go to {url}", f"visit {url}", f"navigate to {url}",
        f"launch {url}", f"open url {url}", f"browser me {url} kholo",
    ])
    return example(user, "browser.open_url", {"url": url})


def gen_time_now(n: int) -> dict:
    user = varied([
        "what time is it", "what's the time", "tell me the time", "current time",
        "time please", "time kya hai", "kitne baj gaye",
    ])
    return example(user, "time.now", {}, tool_reply="OK: 3:45 PM")


def gen_date_now(n: int) -> dict:
    user = varied([
        "what's today's date", "what's the date", "today's date",
        "what day is it", "date please", "aaj ki date", "aaj kya tarikh hai",
    ])
    return example(user, "date.now", {}, tool_reply="OK: Friday, September 5, 2026")


def gen_timer_set(n: int) -> dict:
    minutes = random.randint(1, 90)
    label = random.choice(LABELS)
    user = varied([
        f"set a timer for {minutes} minutes", f"timer {minutes} minutes",
        f"{minutes} minute timer", f"set timer {minutes} min",
        f"{minutes} min ka timer lagao", f"timer {minutes}",
    ])
    if random.random() < 0.5:
        return example(user, "timer.set", {"minutes": minutes})
    user = varied([
        f"set a {minutes} minute timer for {label}", f"timer {minutes} min {label}",
        f"{label} timer {minutes} minutes", f"set {minutes} min timer, label {label}",
    ])
    return example(user, "timer.set", {"minutes": minutes, "label": label})


def gen_alarm_set(n: int) -> dict:
    hour = random.randint(0, 23)
    minute = random.choice([0, 5, 10, 15, 20, 25, 30, 45])
    user = varied([
        f"set alarm for {hour}:{minute:02d}", f"alarm at {hour}:{minute:02d}",
        f"wake me up at {hour}:{minute:02d}", f"set alarm {hour} {minute}",
    ])
    return example(user, "alarm.set", {"hour": hour, "minute": minute})


def gen_reminder_set(n: int) -> dict:
    text = random.choice(NOTES)
    minutes = random.randint(5, 120)
    user = varied([
        f"remind me to {text} in {minutes} minutes",
        f"reminder {text} {minutes} min",
        f"set a reminder for {text}",
        f"{minutes} min me {text} yaad dilana",
    ])
    if random.random() < 0.5:
        return example(user, "reminder.set", {"text": text})
    return example(user, "reminder.set", {"text": text, "minutes": minutes})


def gen_battery(n: int) -> dict:
    return example(varied([
        "battery status", "battery percentage", "how much battery",
        "battery kitna hai", "battery check karo",
    ]), "system.battery.status", {})


def gen_wifi(n: int) -> dict:
    return example(varied([
        "wifi status", "am i connected to wifi", "what wifi am i on",
        "wifi check", "internet status", "wifi batao",
    ]), "system.wifi.status", {})


def gen_clipboard_read(n: int) -> dict:
    return example(varied([
        "read my clipboard", "what's in my clipboard", "clipboard read",
        "paste from clipboard", "clipboard kya hai",
    ]), "clipboard.read", {})


def gen_clipboard_write(n: int) -> dict:
    text = random.choice(NOTES)
    user = varied([
        f"copy {text} to clipboard", f"clipboard copy {text}",
        f"put {text} in clipboard", f"clipboard me {text} rakho",
    ])
    return example(user, "clipboard.write", {"text": text})


def gen_note_add(n: int) -> dict:
    text = random.choice(NOTES)
    user = varied([
        f"note: {text}", f"add note {text}", f"make a note: {text}",
        f"remember {text}", f"jot down {text}", f"note likho {text}",
    ])
    return example(user, "note.add", {"text": text})


def gen_note_read(n: int) -> dict:
    n = random.randint(3, 20)
    return example(varied([
        "read my notes", "what are my notes", "show notes",
        f"read last {n} notes", f"last {n} notes", "notes dikhao",
    ]), "note.read", {"limit": n})


def gen_calc(n: int) -> dict:
    expr, _ = random.choice(MATH)
    user = varied([
        f"what is {expr}", f"calculate {expr}", f"compute {expr}",
        f"what's {expr}", f"{expr} = ?", f"solve {expr}",
    ])
    return example(user, "calc.compute", {"expression": expr})


def gen_app_list(n: int) -> dict:
    return example(varied([
        "what apps are open", "list running apps", "show open windows",
        "what's running", "open apps", "kaunse apps khule hain",
    ]), "app.list_running", {})


def gen_app_switch(n: int) -> dict:
    app = random.choice(APPS)
    user = varied([
        f"switch to {app}", f"go to {app}", f"focus {app}",
        f"bring {app} to front", f"alt tab to {app}", f"{app} pe jao",
    ])
    return example(user, "app.switch", {"name": app})


def gen_lock(n: int) -> dict:
    return example(varied([
        "lock the screen", "lock screen", "lock my laptop",
        "lock please", "screen lock karo",
    ]), "system.lock", {})


# ---------------------------------------------------------------------------
# Phrasings for "no tool" - user says something conversational
# ---------------------------------------------------------------------------
CHIT_CHAT_REPLIES = [
    "haha", "lol", "lmao", "heh", "yeah", "yep", "nope",
    "i don't know", "maybe later", "ok cool", "got it", "sure",
    "no", "yes", "why", "really", "interesting", "tell me more",
    "i see", "fair enough", "makes sense", "agreed", "wait what",
]

CONVERSATIONAL_PROMPTS = [
    "what do you think about AI", "tell me a joke", "are you real",
    "do you dream", "what's the meaning of life", "why is the sky blue",
    "tell me about yourself", "what's your favorite color", "are you sentient",
    "do you have feelings", "what do you like to do", "are you human",
    "what's 2+2",  # avoid: handled by tool
    "do you sleep", "are you smart", "what year is it",
    "do you like music", "what's your favorite food", "do you have a body",
    "sing a song", "tell me a story", "say something funny",
    "are you alive", "do you get tired", "do you eat",
    "what languages do you speak", "can you learn", "how old are you",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
GENERATORS = [
    ("system.volume.set", 200, gen_volume_set),
    ("system.volume.mute", 80, gen_volume_mute),
    ("system.brightness.set", 120, gen_brightness_set),
    ("app.open", 250, gen_app_open),
    ("app.close", 100, gen_app_close),
    ("screenshot.take", 80, gen_screenshot),
    ("media.play", 80, gen_media_play),
    ("media.pause", 80, gen_media_pause),
    ("media.next", 60, gen_media_next),
    ("media.previous", 60, gen_media_previous),
    ("window.minimize_all", 80, gen_minimize_all),
    ("settings.open", 120, gen_settings_open),
    ("browser.search", 200, gen_browser_search),
    ("browser.open_url", 150, gen_browser_open_url),
    ("time.now", 200, gen_time_now),
    ("date.now", 80, gen_date_now),
    ("timer.set", 150, gen_timer_set),
    ("alarm.set", 80, gen_alarm_set),
    ("reminder.set", 120, gen_reminder_set),
    ("system.battery.status", 80, gen_battery),
    ("system.wifi.status", 80, gen_wifi),
    ("clipboard.read", 60, gen_clipboard_read),
    ("clipboard.write", 80, gen_clipboard_write),
    ("note.add", 120, gen_note_add),
    ("note.read", 80, gen_note_read),
    ("calc.compute", 100, gen_calc),
    ("app.list_running", 60, gen_app_list),
    ("app.switch", 80, gen_app_switch),
    ("system.lock", 50, gen_lock),
]


def main() -> None:
    rows: list[dict] = []

    # Tool examples
    for tool_name, count, gen in GENERATORS:
        for _ in range(count):
            rows.append(gen(0))
        print(f"  {tool_name:30s}  +{count}")

    # Greeting / social (NO tool call)
    for user, replies in GREETINGS:
        # 3x for variety
        for _ in range(3):
            rows.append(none_example(user, random.choice(replies)))
    print(f"  {'greetings':30s}  +{len(GREETINGS) * 3}")

    # Chit-chat
    for prompt in CONVERSATIONAL_PROMPTS:
        for _ in range(8):
            rows.append(none_example(prompt, random.choice(CHIT_CHAT_REPLIES + [
                "good question", "hmm, not sure", "i'll think about it",
                "beats me", "let me think", "honestly i don't know",
            ])))
    print(f"  {'chit-chat':30s}  +{len(CONVERSATIONAL_PROMPTS) * 8}")

    # Garbled STT examples (typos, filler words) so the model is robust
    fillers = ["um", "uh", "like", "you know", "actually", "basically", "so"]
    for tool_name, count, gen in GENERATORS[:10]:
        for _ in range(15):
            ex = gen(0)
            user_msg = ex["messages"][1]["content"]
            if random.random() < 0.5:
                user_msg = random.choice(fillers) + " " + user_msg
            if random.random() < 0.3:
                user_msg = user_msg + " " + random.choice(["please", "thanks", "bhai", "yaar"])
            ex["messages"][1] = {"role": "user", "content": user_msg}
            rows.append(ex)

    random.shuffle(rows)

    # 90/10 split
    val_n = max(80, len(rows) // 10)
    val_rows = rows[:val_n]
    train_rows = rows[val_n:]

    train_path = DATA / "sft_train.jsonl"
    val_path = DATA / "sft_val.jsonl"

    # Backup the old data
    if (train_path).exists():
        backup = train_path.with_suffix(".jsonl.bak")
        if not backup.exists():
            train_path.rename(backup)
            print(f"backed up old train to {backup}")
    if (val_path).exists():
        backup_v = val_path.with_suffix(".jsonl.bak")
        if not backup_v.exists():
            val_path.rename(backup_v)
            print(f"backed up old val to {backup_v}")

    with open(train_path, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_path, "w") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(train_rows)} train + {len(val_rows)} val -> {train_path}, {val_path}")


if __name__ == "__main__":
    main()
