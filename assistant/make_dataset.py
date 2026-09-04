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
import argparse
import hashlib
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
    """Return an assistant message dict with a proper tool_calls list
    (Qwen3 / OpenAI shape). The chat template renders this as the
    <tool_call>...</tool_call> block; the runtime regex extracts it."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(params or {}, ensure_ascii=False),
            },
        }],
    }


def tool_result(name, ok=True, output="OK"):
    return {
        "role": "tool",
        "name": name,
        "content": ("OK: " if ok else "ERR: ") + str(output),
    }


AFFIRMATIONS = {
    "system.volume.set":    ["Done. Volume set.", "Volume changed.", "On it."],
    "system.volume.mute":   ["Muted.", "Done."],
    "system.brightness.set":["Brightness adjusted.", "Done."],
    "app.open":             ["Opening.", "Launching.", "On it."],
    "app.close":            ["Closed.", "Done."],
    "screenshot.take":      ["Screenshot taken.", "Captured."],
    "media.play":           ["Playing.", "On it."],
    "media.pause":          ["Paused.", "Done."],
    "media.next":           ["Next track.", "Done."],
    "media.previous":       ["Previous track.", "Done."],
    "window.minimize_all":  ["Desktop is clear.", "Done."],
    "settings.open":        ["Opening settings.", "Done."],
    "browser.search":       ["Searching.", "Looking it up."],
    "browser.open_url":     ["Opening browser.", "Done."],
    "time.now":             ["Done."],
    "date.now":             ["Done."],
}


def affirmation(name):
    opts = AFFIRMATIONS.get(name, ["Done."])
    return rng.choice(opts)


samples = []


def add(user_text, assistant_content):
    if isinstance(assistant_content, dict) and assistant_content.get("tool_calls"):
        # Tool call: emit the tool-call turn, a tool result, and an affirmation
        tc_msg = assistant_content
        name = tc_msg["tool_calls"][0]["function"]["name"]
        messages = [
            msg("system", SYSTEM),
            msg("user", user_text),
            tc_msg,
            tool_result(name),
            msg("assistant", affirmation(name)),
        ]
    else:
        messages = [
            msg("system", SYSTEM),
            msg("user", user_text),
            msg("assistant", str(assistant_content)),
        ]
    samples.append({
        "messages": messages,
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
        template_params = {}
        if isinstance(tpl, tuple):
            tpl, template_params = tpl
        subs = dict(fixed_params or {})
        if "{n}" in tpl:
            subs["n"] = rng.choice([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        for key, value in template_params.items():
            if value == "n" and "n" in subs:
                subs[key] = subs["n"]
            else:
                subs[key] = value
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


def _write_jsonl(path: Path, rows) -> None:
    """Atomically replace a JSONL dataset so interrupted builds stay usable."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _prompt(row) -> str:
    return " ".join(row["messages"][1]["content"].lower().split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--augment-copies", type=int, default=2,
                        help="jittered copies per training prompt")
    args = parser.parse_args()
    if not 0.01 <= args.val_fraction < 0.5:
        parser.error("--val-fraction must be in [0.01, 0.5)")
    if args.augment_copies < 0:
        parser.error("--augment-copies cannot be negative")
    rng.seed(args.seed)
    samples.clear()

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

    # fold in real seeds from the STT agent when present
    if SEEDS.exists():
        kept = 0
        for line in SEEDS.read_text().splitlines():
            try:
                row = json.loads(line)
                tool = row["tool"]
                name = tool.get("name")
                if name == "none":
                    add(row["text"], "I can't perform that action yet.")
                elif name in {item["name"] for item in TOOLS}:
                    add(row["text"], tool_call(name, tool.get("parameters") or {}))
                else:
                    continue
                kept += 1
            except Exception:
                continue
        print("merged", kept, "stt seeds")

    # Split by normalized prompt *before* augmentation. The old implementation
    # split after cloning, leaking near-identical prompts into validation and
    # making eval_loss look much better than real held-out performance.
    unique = list({_prompt(row): row for row in samples}.values())
    rng.shuffle(unique)
    val_n = max(1, int(len(unique) * args.val_fraction))
    val_rows = unique[:val_n]
    base_train = unique[val_n:]

    suffixes = [" please", " now", " yaar", " jaldi", " abhi", " dude"]
    train_rows = list(base_train)
    for sample in base_train:
        for _ in range(args.augment_copies):
            clone = json.loads(json.dumps(sample))
            user_text = clone["messages"][1]["content"]
            suffix = rng.choice(suffixes)
            clone["messages"][1]["content"] = user_text + suffix
            train_rows.append(clone)
    rng.shuffle(train_rows)

    overlap = {_prompt(row) for row in train_rows} & {_prompt(row) for row in val_rows}
    if overlap:
        raise RuntimeError(f"train/validation prompt leakage: {sorted(overlap)[:3]}")

    OUT.mkdir(parents=True, exist_ok=True)
    train_path = OUT / "sft_train.jsonl"
    val_path = OUT / "sft_val.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    digest = hashlib.sha256(train_path.read_bytes() + val_path.read_bytes()).hexdigest()
    report = {
        "seed": args.seed, "train_rows": len(train_rows), "validation_rows": len(val_rows),
        "unique_base_prompts": len(unique), "validation_fraction": args.val_fraction,
        "augmentation_copies": args.augment_copies, "sha256": digest,
    }
    (OUT / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("dataset:", len(train_rows), "train /", len(val_rows), "val", f"sha256={digest[:12]}")


if __name__ == "__main__":
    main()
