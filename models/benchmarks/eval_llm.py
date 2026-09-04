#!/usr/bin/env python3
"""Tool-call accuracy benchmark for cozy-llm-v1.0 vs cozy-llm-v1.1.

Uses a held-out 30-probe set derived from assistant/data/sft_val.jsonl. We
expect each row to contain a user turn + assistant turn, and we check
whether the model emits a tool call when one is expected and a plain text
answer when one is not.

Run from repo root:
    python models/benchmarks/eval_llm.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "models" / "benchmarks" / "llm_eval.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

# A small, hand-curated probe set covering the 6 most-used tools in the
# executor plus 5 chitchat (should-not-call-tool) probes. We avoid the SFT
# train/val splits so this is genuinely held-out.
PROBES = [
    # Tool calls
    {"user": "set volume to 25",        "expect_tool": "system_volume_set",      "kind": "tool"},
    {"user": "turn the volume down to 10", "expect_tool": "system_volume_set",    "kind": "tool"},
    {"user": "what time is it",          "expect_tool": "date_now",               "kind": "tool"},
    {"user": "what's today's date",      "expect_tool": "date_now",               "kind": "tool"},
    {"user": "open firefox",             "expect_tool": "app_open",               "kind": "tool"},
    {"user": "launch the terminal",      "expect_tool": "app_open",               "kind": "tool"},
    {"user": "search for cozy reviews",  "expect_tool": "browser_search",         "kind": "tool"},
    {"user": "google best python ide",   "expect_tool": "browser_search",         "kind": "tool"},
    {"user": "take a screenshot",        "expect_tool": "screenshot_take",        "kind": "tool"},
    {"user": "snapshot my screen",       "expect_tool": "screenshot_take",        "kind": "tool"},
    {"user": "lock the screen",          "expect_tool": "system_lock",            "kind": "tool"},
    {"user": "lock my laptop",           "expect_tool": "system_lock",            "kind": "tool"},
    {"user": "shut down the computer",   "expect_tool": "system_shutdown",        "kind": "tool"},
    {"user": "power off",                "expect_tool": "system_shutdown",        "kind": "tool"},
    {"user": "set brightness to 80",     "expect_tool": "system_brightness_set",  "kind": "tool"},
    {"user": "dim the screen to 20 percent","expect_tool": "system_brightness_set","kind": "tool"},
    # Chitchat (no tool)
    {"user": "hey cozy",                 "expect_tool": None,                      "kind": "chitchat"},
    {"user": "hi",                       "expect_tool": None,                      "kind": "chitchat"},
    {"user": "thanks",                   "expect_tool": None,                      "kind": "chitchat"},
    {"user": "what's your name",         "expect_tool": None,                      "kind": "chitchat"},
    {"user": "tell me a joke",           "expect_tool": None,                      "kind": "chitchat"},
    # Edge cases
    {"user": "what's 2 + 2",             "expect_tool": None,                      "kind": "chitchat"},
    {"user": "why is the sky blue",      "expect_tool": None,                      "kind": "chitchat"},
    {"user": "good morning",             "expect_tool": None,                      "kind": "chitchat"},
    {"user": "are you there",            "expect_tool": None,                      "kind": "chitchat"},
    {"user": "how's the weather",        "expect_tool": None,                      "kind": "chitchat"},
    # Mixed
    {"user": "set volume to 50 and open firefox", "expect_tool": "system_volume_set","kind": "tool"},
    {"user": "what time is it and then lock the screen","expect_tool":"date_now",  "kind": "tool"},
    {"user": "open chrome and search for cats", "expect_tool": "app_open",         "kind": "tool"},
    {"user": "you there?",               "expect_tool": None,                      "kind": "chitchat"},
    {"user": "yo",                       "expect_tool": None,                      "kind": "chitchat"},
]


SYSTEM = (
    "You are Cozy, a voice assistant running fully offline on the user's "
    "laptop. Respond fast and short. When the user wants an action, call "
    "exactly one tool with compact JSON. For plain chat, answer briefly "
    "and warmly without tools."
)


def looks_like_tool_call(text: str) -> bool:
    """Detect a tool call. Three valid emission formats observed in the wild:
    (1) Qwen3 with chat_template:  <tool_call>{"name": ..., "arguments": ...}</tool_call>
    (2) Raw JSON object with 'name' + 'arguments' keys
    (3) Bare function name like 'time.now'
    """
    # Qwen3 styled tag
    if "<tool_call>" in text or "<|tool_call|>" in text:
        return True
    # Raw JSON tool call
    if '"name"' in text and '"arguments"' in text:
        return True
    # Bare tool name (the v1.0 model sometimes emits just "time.now")
    if re.search(r'\b(system\.volume\.set|date\.now|app\.open|browser\.search|'
                 r'screenshot\.take|system\.lock|system\.shutdown|'
                 r'system\.brightness\.set|system\.volume\.mute|window\.minimize\.all|'
                 r'settings\.open|app\.close|media\.control)\b', text):
        return True
    return False


def evaluate(name: str, model_path: Path) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  loading {model_path} ...")
    tok = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()

    tool_schema = json.loads((REPO / "team" / "tool_schema.json").read_text())
    correct, correct_tool, correct_chat = 0, 0, 0
    n_tool = sum(1 for p in PROBES if p["kind"] == "tool")
    n_chat = sum(1 for p in PROBES if p["kind"] == "chitchat")

    for p in PROBES:
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": p["user"]}],
            tools=tool_schema["tools"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=200, do_sample=False)
        text = tok.decode(out[0][ids["input_ids"].shape[1]:],
                          skip_special_tokens=False)
        is_tool = looks_like_tool_call(text)
        if p["kind"] == "tool":
            if is_tool:
                correct_tool += 1
                correct += 1
        else:  # chitchat
            if not is_tool:
                correct_chat += 1
                correct += 1
    out = {
        "model": name,
        "accuracy": correct / len(PROBES),
        "tool_accuracy": correct_tool / n_tool,
        "chat_accuracy": correct_chat / n_chat,
        "n_total": len(PROBES),
        "n_tool": n_tool,
        "n_chat": n_chat,
    }
    print(f"  overall: {out['accuracy']*100:.1f}%  tool: {out['tool_accuracy']*100:.1f}%  chat: {out['chat_accuracy']*100:.1f}%")
    # free GPU
    del model
    torch.cuda.empty_cache()
    return out


def main() -> None:
    v10 = evaluate("v1.0 (SFT 0.1)", REPO / "models" / "cozy-llm-v1.0" / "base")
    v11 = evaluate("v1.1 (SFT 1.0)", REPO / "models" / "cozy-llm-v1.1" / "base")
    summary = {"v1.0": v10, "v1.1": v11, "probes": PROBES}
    OUT.write_text(json.dumps(summary, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
