#!/usr/bin/env python3
"""Generates the function-calling + conversation SFT dataset for Cozy's
small local LLM (Qwen3-0.6B class).

Output: assistant/data/sft_train.jsonl (+ sft_val.jsonl)
Each line: {"messages": [...], "tools": [...]} - rendered by the tokenizer's
chat template during training.

Merge note: if team/data/stt_command_seeds.jsonl exists (from the STT
agent's real user utterances), those are folded in automatically.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
SEEDS = HERE.parent / "team" / "data" / "stt_command_seeds.jsonl"

rng = random.Random(42)

TOOLS = json.loads((HERE.parent / "team" / "tool_schema.json").read_text())["tools"]

SYSTEM = (
    "You are Cozy, a voice assistant running fully offline on the user's "
    "laptop. Respond fast and short. When the user wants an action, call "
    "exactly one tool with compact JSON. For plain chat, answer briefly "
    "and warmly without tools."
)

APPS = ["browser", "chrome", "firefox", "terminal", "files", "spotify",
        "calculator", "settings", "vs code", "text editor", "camera",
        "mail", "calendar", "notes"]
PAGES = ["wifi", "bluetooth", "display", "sound", "notifications", "power"]
QUERIES = ["weather in kolkata", "best laptop under 50000", "python tutorial",
           "india cricket score", "recipe for biryani", "llvm vs gcc",
           "nearest petrol pump", "who wrote harry potter"]
URLS = ["github.com", "youtube.com", "mail.google.com", "wikipedia.org"]

VOLUME_CMDS = [
    ("set volume to {n}", {"level": "n"}),
    ("volume {n} percent", {"level": "n"}),
    ("volume {n}", {"level": "n"}),
    ("sound level {n}", {"level": "n"}),
]
HING_VOLUME = [
    ("volume {n} kar do", {"level": "n"}),
    ("awaj {n} percent kar", {"level": "n"}),
    ("volume {n} karo bhai", {"level": "n"}),
]


def sample_tools():
    return TOOLS


def msg(role, content):
    return {"role": role, "content": content}


def tool_call(name, params=None):
    return json.dumps({"name": name, "parameters": params or {}},
                      separators=(",", ":"))


samples = []


def add(user_text, assistant_content):
    samples.append({
        "messages": [
            msg("system", SYSTEM),
            msg("user", user_text),
            msg("assistant", assistant_content),
        ],
        "tools": sample_tools(),
    })


def render(tpl, n_map):
    text = tpl
    params = {}
    for key, val in n_map.items():
        if isinstance(val, str) and "{" + key + "}" in tpl:
            text = text.replace("{" + key + "}", str(val))
        else:
            params[key] = val
    return text, params


def emit_tool(templates, tool, fixed_params=None, count=30):
    for _ in range(count):
        tpl = rng.choice(templates)
        subs = dict(fixed_params or {})
        if "{n}" in tpl:
            subs["n"] = rng.choice([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        text, params = render(tpl, subs)
        if fixed_params:
            params.update(fixed_params)
        add(text, tool_call(tool, params))


def emit_conv(pairs):
    for u, a in pairs:
        add(u, a)


CONVERSATION = [
    ("hey cozy", "Hey! Listening."),
    ("hi cozy", "Hi! What do you need?"),
    ("who are you", "I'm Cozy - your offline voice assistant."),
    ("what can you do",
     "I can open apps, control volume and brightness, take screenshots, "
     "search the web and change settings. Just ask."),
    ("how are you", "Running smooth and fast. What can I do for you?"),
    ("thank you cozy", "Anytime!"),
    ("thanks", "You got it."),
    ("tell me a joke",
     "Why do programmers prefer dark mode? Because light attracts bugs."),
    ("good morning cozy", "Good morning! System is ready."),
    ("good night cozy", "Good night! Shutting down listening mode."),
    ("are you there", "Always. What do you need?"),
    ("what is your name", "Cozy. Your offline assistant."),
]

HING_CONV = [
    ("kaise ho cozy", "Badhiya! Batao kya karna hai."),
    ("tum kaun ho", "Main Cozy hoon - tumhara offline assistant."),
    ("theek hai thank you", " anytime."),
    ("so jao cozy", "Theek hai, listening mode off. Bol dena jab zaroorat ho."),
]

EDGE = [
    ("open the pod bay doors",
     tool_call("app.open", {"name": "pod bay doors"})),
    ("cozy", "Listening..."),
    ("hey", "Hey! What do you need?"),
    ("ummm just checking", "All good. I'm here if you need me."),
    ("make it louder",
     tool_call("system.volume.set", {"level": 90})),
    ("too bright",
     tool_call("system.brightness.set", {"level": 30})),
]


def main() -> None:
    # volume / brightness / media / apps / settings / browser
    emit_tool(VOLUME_CMDS, "system.volume.set")
    emit_tool(HING_VOLUME, "system.volume.set")
    emit_tool(["mute", "mute the sound", "sound band karo",
               "muute kardo"], "system.volume.mute", {}, 15)
    emit_tool(["set brightness to {n}", "brightness {n}",
               "screen {n} percent", "roshni {n} kar do"],
              "system.brightness.set", {}, 35)
    for app in APPS:
        emit_tool(["open " + app, "launch " + app, "start " + app,
                   app + " kholo", app + " open karo"],
                  "app.open", {"name": app}, 6)
    for app in APPS[:8]:
        emit_tool(["close " + app, "band karo " + app],
                  "app.close", {"name": app}, 4)
    emit_tool(["take a screenshot", "screenshot lo",
               "screen shot capture karo"], "screenshot.take", {}, 18)
    emit_tool(["play music", "play", "gaana chalu karo"],
              "media.play", {}, 10)
    emit_tool(["pause", "pause the music", "music roko"],
              "media.pause", {}, 10)
    emit_tool(["next track", "next song", "agla gaana"],
              "media.next", {}, 10)
    emit_tool(["previous track", "pichla gaana"], "media.previous", {}, 8)
    emit_tool(["minimize everything", "minimize all windows",
               "desktop dikhao"], "window.minimize_all", {}, 12)
    for page in PAGES:
        emit_tool(["open " + page + " settings",
                   page + " settings kholo",
                   "go to " + page + " settings"],
                  "settings.open", {"page": page}, 6)
    for q in QUERIES:
        for tpl in ["search for {q}", "search {q}", "{q} search karo",
                    "google {q}"]:
            text = tpl.replace("{q}", q)
            add(text, tool_call("browser.search", {"query": q}))
    for u in URLS:
        for tpl in ["open {u}", "{u} kholo"]:
            text = tpl.replace("{u}", u)
            add(text, tool_call("browser.open_url", {"url": "https://" + u}))
    emit_tool(["what time is it", "time batao", "current time",
               "aaj ki date"], "time.now", {}, 16)

    emit_conv(CONVERSATION)
    emit_conv(HING_CONV)
    for u, a in EDGE:
        add(u, a)

    # expansion: jittered user-text clones teach phrasing robustness
    SUFFIXES = ["", " please", " now", " yaar", " jaldi", " abhi", " dude"]
    extra = []
    for s in samples:
        extra.append(s)
        for _ in range(2):
            clone = json.loads(json.dumps(s))
            u0 = clone["messages"][1]["content"]
            if not isinstance(u0, str):
                continue
            suffix = rng.choice(SUFFIXES)
            if suffix and not u0.endswith(suffix):
                clone["messages"][1]["content"] = u0 + suffix
                extra.append(clone)
    samples.extend(extra)

    # fold in real seeds from the STT agent when present
    if SEEDS.exists():
        kept = 0
        for line in SEEDS.read_text().splitlines():
            try:
                row = json.loads(line)
                add(row["text"], json.dumps(row["tool"],
                                            separators=(",", ":")))
                kept += 1
            except Exception:
                continue
        print("merged", kept, "stt seeds")

    rng.shuffle(samples)
    OUT.mkdir(parents=True, exist_ok=True)
    val_n = max(1, int(len(samples) * 0.02))
    with open(OUT / "sft_val.jsonl", "w") as f:
        for s in samples[:val_n]:
            f.write(json.dumps(s, ensure_ascii=False) + chr(10))
    with open(OUT / "sft_train.jsonl", "w") as f:
        for s in samples[val_n:]:
            f.write(json.dumps(s, ensure_ascii=False) + chr(10))
    print("dataset:", len(samples) - val_n, "train /", val_n, "val")


if __name__ == "__main__":
    main()
