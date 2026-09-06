#!/usr/bin/env python3
"""Smoke test the trained Cozy LLM with edge cases that used to fail.

Run after sft_qwen.py (or after dpo_light.py) finishes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCHEMA = json.loads((ROOT / "team" / "tool_schema.json").read_text())
TOOLS = SCHEMA["tools"]


def load_model(model_dir: str = "model/cozy-llm-v1") -> tuple:
    p = HERE / model_dir
    tok = AutoTokenizer.from_pretrained(str(p))
    m = AutoModelForCausalLM.from_pretrained(str(p), dtype=torch.bfloat16,
                                              attn_implementation="sdpa").to("cuda")
    m.eval()
    return tok, m


def query(tok, m, prompt: str) -> str:
    msgs = [
        {"role": "system", "content":
            "You are Cozy, a voice assistant running fully offline on the user's "
            "laptop. Respond fast and short. When the user wants an action, call "
            "exactly one tool with compact JSON. For plain chat, answer briefly "
            "and warmly without tools."},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(**ids, max_new_tokens=200, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)


# Test cases: (prompt, expected_tool_or_None)
TESTS = [
    ("set volume to 50", "system.volume.set"),
    ("set volume to 75", "system.volume.set"),
    ("volume 30 karo", "system.volume.set"),
    ("open firefox", "app.open"),
    ("launch chrome", "app.open"),
    ("what time is it", "time.now"),
    ("what's today's date", "date.now"),
    ("set a 10 minute timer", "timer.set"),
    ("set alarm for 7:30", "alarm.set"),
    ("remind me to call mom in 10 minutes", "reminder.set"),
    ("how are you", None),  # plain text
    ("hi", None),
    ("thanks", None),
    ("take a screenshot", "screenshot.take"),
    ("search python tutorial", "browser.search"),
    ("open https://github.com", "browser.open_url"),
    ("mute", "system.volume.mute"),
    ("next song", "media.next"),
    ("pause music", "media.pause"),
    ("brightness 80", "system.brightness.set"),
    ("close firefox", "app.close"),
    ("switch to terminal", "app.switch"),
    ("note: buy milk", "note.add"),
    ("copy hello to clipboard", "clipboard.write"),
    ("what is 2+2", "calc.compute"),
    ("battery status", "system.battery.status"),
]


def main() -> None:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "model/cozy-llm-v1"
    print(f"loading {model_dir}...")
    tok, m = load_model(model_dir)

    correct = 0
    for prompt, expected in TESTS:
        try:
            out = query(tok, m, prompt)
        except Exception as e:
            print(f"ERROR on '{prompt}': {e}")
            continue
        # Parse tool call (match what extract_tool_call in harness_fast does)
        import re
        tool_name = None
        tag_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", out, re.S)
        if tag_match:
            try:
                tc = json.loads(tag_match.group(1))
                tool_name = tc.get("name")
            except Exception:
                pass
        if tool_name is None:
            # balanced-brace fallback
            cleaned = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
            depth = 0; start = None
            for i, ch in enumerate(cleaned):
                if ch == "{":
                    if depth == 0: start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start is not None:
                        try:
                            c = json.loads(cleaned[start:i+1])
                            if isinstance(c, dict) and isinstance(c.get("name"), str):
                                tool_name = c.get("name")
                        except: pass
                        start = None
                        if tool_name: break
        ok = "✓" if tool_name == expected else "✗"
        if ok == "✓":
            correct += 1
        print(f"{ok}  '{prompt:42s}'  expected={expected!s:24s} got={tool_name!s}")
        if ok == "✗":
            print(f"      raw: {out.strip()[:200]}")
    print(f"\n{correct}/{len(TESTS)} correct")


if __name__ == "__main__":
    main()
