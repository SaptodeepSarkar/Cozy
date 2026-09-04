"""Cozy intent -> executor bridge: use rule-based router, fall back to LLM for chat only."""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from intents import route as intent_route
from executor import execute as exec_tool


def _try_route(text: str):
    """Run intent router. If it returns a known tool, execute it.
    Otherwise return None so caller can fall through to LLM chat."""
    res = intent_route(text)
    tool = res.get("tool")
    args = res.get("args", {})

    if tool == "set_volume":
        if "level" in args:
            return exec_tool("system.volume.set", {"level": args["level"]})
        if "delta" in args:
            # get current then set
            r = subprocess.run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                                capture_output=True, text=True, timeout=5)
            m = re.search(r"(d+)%", r.stdout)
            cur = int(m.group(1)) if m else 50
            new = max(0, min(100, cur + args["delta"]))
            return exec_tool("system.volume.set", {"level": new})
        return {"ok": False, "output": "no volume arg"}
    if tool == "open_app":
        return exec_tool("app.open", {"name": args.get("app", "")})
    if tool == "browser_search":
        return exec_tool("browser.search", {"query": args.get("q", "")})
    if tool == "screenshot":
        return exec_tool("screenshot.take", {})
    if tool == "query_time":
        return exec_tool("time.now", {})
    if tool == "none":
        return {"ok": True, "output": "How can I help?"}
    if tool == "unhandled":
        return None  # let LLM handle
    return None


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or input("> ")
    result = _try_route(text)
    if result is not None:
        print(json.dumps(result, indent=1))
    else:
        # LLM chat fallback
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(HERE / "model" / "cozy-llm-v1"))
        model = AutoModelForCausalLM.from_pretrained(
            str(HERE / "model" / "cozy-llm-v1"), torch_dtype=torch.bfloat16)
        model.to("cuda")
        model.eval()
        prompt = tok.apply_chat_template(
            [{"role": "system", "content":
              "You are Cozy, a friendly voice assistant. Reply briefly and warmly."},
             {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=80, do_sample=False,
                            pad_token_id=tok.eos_token_id)
        reply = tok.decode(out[0][ids["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
        print(json.dumps({"chat": reply or "..."}, indent=1))
