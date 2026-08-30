"""Core harness: load a tool-calling LLM, render a prompt, parse a call.

Two backends ship in the box:

* ``ModelBackend``  - load the local ``cozy-llm-v1`` (or a base+adapter pair)
                       and produce tool calls via HuggingFace transformers.
* ``RuleBackend``   - call the existing intent router from
                       ``assistant.intents`` (no LLM at all). Useful for
                       fast smoke tests and for collecting deterministic
                       traces.

Both expose the same interface::

    backend.decide(messages, tools) -> {"text": "...", "tool": {"name": .., "parameters": ...} | None}
"""
from __future__ import annotations

import json
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

ASSISTANT = Path(__file__).resolve().parent.parent
if str(ASSISTANT) not in sys.path:
    sys.path.insert(0, str(ASSISTANT))


SYSTEM_PROMPT = (
    "You are Cozy, a voice assistant running fully offline on the user's "
    "laptop. Respond fast and short. When the user wants an action, call "
    "exactly one tool with compact JSON. For plain chat, answer briefly "
    "and warmly without tools."
)


class Backend(ABC):
    name: str = "abstract"

    @abstractmethod
    def decide(self, messages: list[dict], tools: list[dict]) -> dict:
        """Return ``{"text": str, "tool": dict|None}``."""


# ------------------------------------------------------------ rule backend
class RuleBackend(Backend):
    """Pure rule-based backend - no LLM. Calls into assistant.intents.

    The intent router returns a dict shaped like
    ``{"tool": "open_app", "args": {"app": "chrome"}, "_source": ...}``.
    We adapt that to the same shape the model backend returns.
    """

    name = "rule"

    def __init__(self) -> None:
        # Late import so the package doesn't pull in everything on load.
        from intents import route as intent_route
        from executor import execute as exec_tool
        self._route = intent_route
        self._exec = exec_tool

    def decide(self, messages: list[dict], tools: list[dict]) -> dict:
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                user_text = m["content"]
                break
        if not user_text:
            return {"text": "", "tool": None}

        res = self._route(user_text)
        tool_name = res.get("tool")
        args = res.get("args", {})

        # Map router result to LLM-style tool call
        mapping = {
            "set_volume": ("system.volume.set", {"level": args.get("level")}),
            "open_app": ("app.open", {"name": args.get("app", "")}),
            "browser_search": ("browser.search",
                                {"query": args.get("q", args.get("query", ""))}),
            "screenshot": ("screenshot.take", {}),
            "query_time": ("time.now", {}),
            "none": (None, None),
        }
        if tool_name in mapping:
            m_name, m_args = mapping[tool_name]
            if m_name is None:
                reply = "Listening..."
                return {"text": reply, "tool": None}
            # actually execute and feed the result back as text
            result = self._exec(m_name, m_args or {})
            out = ("Done. " if result.get("ok") else "Failed: ") +                   str(result.get("output", ""))
            return {"text": out, "tool": {"name": m_name, "parameters": m_args or {}},
                    "executed": result}
        return {"text": "How can I help?", "tool": None}


# ---------------------------------------------------------- model backend
class ModelBackend(Backend):
    """Run the local Cozy LLM (Qwen3-0.6B + LoRA adapter).

    Loads from ``assistant/model/cozy-llm-v1`` (merged) by default.
    Pass ``adapter_dir`` to use the LoRA adapter with a base model.
    """

    name = "model"

    def __init__(self, model_dir: str | Path | None = None,
                 adapter_dir: str | Path | None = None,
                 device: str = "cuda",
                 dtype: str = "bfloat16",
                 max_new_tokens: int = 96) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch

        base = Path(model_dir) if model_dir else ASSISTANT / "model" / "cozy-llm-v1"
        if not base.exists():
            raise FileNotFoundError(
                f"LLM model not found at {base}. Run `python sft_qwen.py` first "
                f"or pass --model-dir.")

        self._tok = AutoTokenizer.from_pretrained(str(base))

        torch_dtype = {"bfloat16": torch.bfloat16,
                       "float16": torch.float16,
                       "float32": torch.float32}[dtype]

        if adapter_dir:
            # base + LoRA adapter
            from peft import PeftModel
            self._model = AutoModelForCausalLM.from_pretrained(
                str(base), dtype=torch_dtype,
                attn_implementation="sdpa")
            self._model = PeftModel.from_pretrained(
                self._model, str(Path(adapter_dir)))
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                str(base), dtype=torch_dtype,
                attn_implementation="sdpa")

        self._model = self._model.to(device)
        self._model.eval()
        self._max_new_tokens = max_new_tokens
        self._device = device
        print(f"[rlm-harness] loaded {base.name} on {device}")

    def decide(self, messages: list[dict], tools: list[dict]) -> dict:
        # Inject system prompt if missing
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)
        prompt = self._tok.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = self._tok(prompt, return_tensors="pt").to(self._device)
        with self._torch.inference_mode():
            out = self._model.generate(
                **ids, max_new_tokens=self._max_new_tokens, do_sample=False,
                pad_token_id=self._tok.eos_token_id)
        text = self._tok.decode(
            out[0][ids["input_ids"].shape[1]:],
            skip_special_tokens=True).strip()
        # strip any residual thinking block (Qwen3)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        return self._parse(text)

    @staticmethod
    def _parse(text: str) -> dict:
        # Try explicit <tool_call>...</tool_call> first.
        m_tag = re.search(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.S)
        if m_tag:
            try:
                call = json.loads(m_tag.group(1))
                if isinstance(call, dict) and isinstance(call.get("name"), str):
                    params = call.get("parameters") or call.get("arguments") or {}
                    if isinstance(params, str):
                        try:
                            params = json.loads(params)
                        except Exception:
                            params = {}
                    return {"text": "", "tool": {"name": call["name"],
                                                  "parameters": params}}
            except json.JSONDecodeError:
                pass
        # Fall back to a balanced-brace scan so we do not match inner
        # sub-objects (e.g. the "parameters": {"name": "chrome"} value).
        depth = 0
        start = None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    chunk = text[start:i + 1]
                    try:
                        call = json.loads(chunk)
                    except json.JSONDecodeError:
                        start = None
                        continue
                    if isinstance(call, dict) and isinstance(call.get("name"), str):
                        params = (call.get("parameters")
                                  or call.get("arguments") or {})
                        if isinstance(params, str):
                            try:
                                params = json.loads(params)
                            except Exception:
                                params = {}
                        return {"text": "", "tool": {"name": call["name"],
                                                      "parameters": params}}
                    start = None
        return {"text": text or "...", "tool": None}


# ------------------------------------------------------------- the harness
class Harness:
    """Owns a backend and a tool schema. Drives one decision at a time."""

    def __init__(self, backend: Backend,
                 tools: list[dict],
                 system_prompt: str = SYSTEM_PROMPT) -> None:
        self.backend = backend
        self.tools = tools
        self.system_prompt = system_prompt

    def decide(self, trace) -> dict:
        """Run one decision on the current trace state."""
        # Build the messages list from the trace (system + turns)
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        for t in trace.turns:
            d = t.to_dict()
            messages.append(d)
        return self.backend.decide(messages, self.tools)


# -------------------------------------------------------------- tool loader
def load_tools_schema() -> list[dict]:
    """Read ``team/tool_schema.json`` and return the tool list."""
    schema_path = ASSISTANT.parent / "team" / "tool_schema.json"
    return json.loads(schema_path.read_text())["tools"]
