"""High-level VLM: Qwen2.5-VL-3B-Instruct (NF4 4-bit).

Two modes:

  1. **plan** — break a user goal into a precise, ordered, *checkable*
     todo list, grounded in the live OS context (active window, open
     windows, focused element, working area).
  2. **answer** — answer a free-form visual question about the screen
     ("is the terminal focused?", "what's in the title bar?", "is
     firefox open?").

Both modes feed the VLM a single screenshot + the live OS context
collected by :mod:`harness.context` + the appropriate system prompt.

The model is loaded with bitsandbytes NF4 4-bit quantization so it
fits in ~2 GB on the dGPU. This leaves enough room for the VLA,
STT, and wake model to share the same GPU.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from .context import OSContext

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen2.5-vl-3b"


PLAN_SYSTEM_PROMPT = """You are the PLANNER for a local desktop agent on Pop!_OS (COSMIC desktop, Wayland).

You see the current screenshot of the user's screen, the live OS context (active window, open
windows, focused element, working area), and a high-level user goal. You must break the goal
into a precise, ordered, *checkable* todo list that a separate fast visual-grounding model will
execute one item at a time.

Each todo item must be:
  * ATOMIC: one user-facing action, e.g. "open Firefox", "type 'github.com' in the URL bar",
    "click the search button", "press Ctrl+L", "scroll down 5 times", "close the current window".
  * GROUNDED: include the target element / window / file path / text so the executor does not
    have to guess.
  * VERIFIABLE: include a `check` predicate that the executor can run after the action to
    confirm success (e.g. window title contains "Firefox", URL bar shows "github.com",
    process list contains "firefox").
  * SAFE: never propose destructive actions (rm -rf, chmod 777, dd, mkfs, sysrq).

Output ONLY a JSON array of objects, one per todo item. Schema:
[{{"action": "<short verb phrase>", "target": "<element / window / file / text>",
  "check": "<how to verify success>", "params": {{...optional kwargs...}}}}]

Do NOT output coordinates; the executor handles those. No markdown, no commentary.
"""

QA_SYSTEM_PROMPT = """You are a visual question-answering model for a local desktop agent on Pop!_OS.

You see the current screenshot of the user's screen and a question. Answer the question
concisely (1-3 sentences) based ONLY on what is visible. If the answer is not visible, say
"I cannot tell from the current screen" rather than guessing. Prefer exact text from the
screen when transcribing (titles, button labels, URLs, error messages). Do not editorialize."""


@dataclass
class TodoItem:
    action: str
    target: str = ""
    check: str = ""
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"action": self.action, "target": self.target,
                "check": self.check, "params": self.params}


@dataclass
class Plan:
    todo: list[TodoItem]
    raw: str
    context_used: dict


@dataclass
class Answer:
    text: str
    raw: str


@dataclass
class VerifyResult:
    done: bool
    reason: str
    raw: str


class Planner:
    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        device: str = "cuda:0",
        max_memory: dict | None = None,
        dtype: torch.dtype = torch.bfloat16,
        gpu_mem_cap: str = "2.5GiB",
    ) -> None:
        self.model_dir = Path(model_dir)
        # Honour the COZY_VISION_DEVICE env var for lite mode.
        env_dev = os.environ.get("COZY_VISION_DEVICE", "auto")
        if env_dev == "cpu" and device == "cuda:0":
            self.device = "cpu"
        elif env_dev == "cuda" and device == "cuda:0":
            self.device = "cuda:0"
        else:
            self.device = device
        # Cap GPU memory tightly. Cozy stack + STT + wake + VLA must coexist.
        # VLM gets ~2 GB max; remaining 4 GB is for cozy-llm-v1 (1.2) + STT (0.5)
        # + VLA-UI-TARS-NF4 (2.0) + wake + headroom.
        self.max_memory = max_memory or (
            {"cpu": "20GiB"} if self.device == "cpu"
            else {0: gpu_mem_cap, "cpu": "20GiB"}
        )
        self.dtype = dtype
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        # On CPU, bnb 4-bit doesn't help (it requires CUDA kernels).
        # Just load in bf16/fp16 and let accelerate handle placement.
        if self.device == "cpu":
            from transformers import BitsAndBytesConfig as _BNB
            bnb = None  # no quantization on CPU
        else:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype,
            )
        self._processor = AutoProcessor.from_pretrained(
            str(self.model_dir),
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        # device_map="auto" + max_memory works for bnb 4-bit when the
        # GPU cap is large enough to hold the whole quantized model.
        # The cap is the *whole* dGPU budget for this model. When the
        # cozy voice stack is also on the dGPU, lower the cap to
        # 2.0GiB via the constructor.
        kwargs = dict(
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        if bnb is not None:
            kwargs["quantization_config"] = bnb
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = self.max_memory
        else:
            # CPU mode: load normally, then move to CPU
            kwargs["device_map"] = {"": "cpu"}
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(self.model_dir),
            **kwargs,
        )
        self._model.eval()

    def free(self) -> None:
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---------------------------------------------------------- plan
    def plan(
        self,
        goal: str,
        screenshot: Image.Image,
        context: OSContext,
        history: list[TodoItem] | None = None,
    ) -> Plan:
        assert self._model is not None, "call .load() first"
        history = history or []
        history_block = ""
        if history:
            history_block = (
                "Todo items already completed (in order):\n"
                + "\n".join(f"- [{t.action}] target={t.target} check={t.check}" for t in history)
                + "\n\n"
            )
        user_text = (
            f"{context.to_prompt()}\n\n"
            f"{history_block}"
            f"User goal: {goal!r}\n\n"
            f"Return the remaining todo items as a JSON array (empty list if the goal is already "
            f"complete or unsafe)."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": PLAN_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        raw = self._generate(messages, max_new_tokens=512)
        todo = self._parse_plan(raw)
        return Plan(todo=todo, raw=raw, context_used=context.to_dict())

    # ---------------------------------------------------------- answer
    def answer(
        self,
        question: str,
        screenshot: Image.Image,
        context: OSContext,
    ) -> Answer:
        assert self._model is not None, "call .load() first"
        user_text = (
            f"{context.to_prompt()}\n\n"
            f"Question: {question!r}\n\n"
            f"Answer concisely based on the screen."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": QA_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        raw = self._generate(messages, max_new_tokens=256)
        return Answer(text=raw.strip(), raw=raw)

    # ---------------------------------------------------------- shared
    def _generate(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs = [m for m in messages[1]["content"] if m.get("type") == "image"]
        # Build image list for processor
        imgs = [screenshot for m in image_inputs for screenshot in [m.get("image")] if screenshot]
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, _ = process_vision_info(messages)
        except Exception:
            image_inputs = imgs
        inputs = self._processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(self._model.device)
        with torch.inference_mode():
            ids = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0
            )
        out_ids = ids[:, inputs.input_ids.shape[1]:]
        raw = self._processor.batch_decode(
            out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return raw

    # ---------------------------------------------------------- verify
    def verify(
        self,
        todo,
        screenshot: Image.Image,
        context: OSContext,
    ) -> "VerifyResult":
        """Ask the VLM if ``todo`` is complete based on the new screen.

        Returns a :class:`VerifyResult` with ``done`` (bool), ``reason``
        (str), and ``raw`` (the model output). Uses a small focused
        prompt so the VLM answers in <1s.
        """
        assert self._model is not None, "call .load() first"
        user_text = (
            f"{context.to_prompt()}\n\n"
            f"Todo that was just executed:\n"
            f"  action: {todo.action}\n"
            f"  target: {todo.target}\n"
            f"  check:  {todo.check}\n\n"
            f"Based on the current screen, is this todo complete?\n"
            f"Reply with ONLY a JSON object on one line:\n"
            f'{{"done": true, "reason": "<why>"}} or {{"done": false, "reason": "<why>"}}'
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        raw = self._generate(messages, max_new_tokens=80)
        done, reason = self._parse_verify(raw)
        return VerifyResult(done=done, reason=reason, raw=raw)

    @staticmethod
    def _parse_verify(raw: str) -> tuple[bool, str]:
        # Look for {"done": ...}
        m = re.search(r"\{[^{}]*\"done\"[^{}]*\}", raw, re.DOTALL)
        candidate = m.group(0) if m else raw
        try:
            d = json.loads(candidate)
            return bool(d.get("done", False)), str(d.get("reason", ""))
        except Exception:
            # Fallback: look for "done": true / false
            low = raw.lower()
            if "not done" in low or "false" in low or "failed" in low:
                return False, raw.strip()[:200]
            return True, raw.strip()[:200]

    @staticmethod
    def _parse_plan(raw: str) -> list[TodoItem]:
        # Strip code fences if any
        m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
        candidate = m.group(1) if m else raw
        # Also find a JSON array in the text
        m2 = re.search(r"\[\s*\{.*?\}\s*\]", candidate, re.DOTALL)
        if m2:
            candidate = m2.group(0)
        try:
            arr = json.loads(candidate)
        except Exception:
            return []
        if not isinstance(arr, list):
            return []
        out: list[TodoItem] = []
        for item in arr:
            if isinstance(item, str):
                out.append(TodoItem(action=item))
            elif isinstance(item, dict):
                out.append(TodoItem(
                    action=str(item.get("action", "")).strip(),
                    target=str(item.get("target", "")).strip(),
                    check=str(item.get("check", "")).strip(),
                    params=item.get("params", {}) or {},
                ))
        return [t for t in out if t.action]
