"""Fast-loop visual grounder VLA: UI-TARS-2B-SFT.

The grounder is given ONE todo item at a time (the head of the VLM's
plan) plus the current screen, and produces the next concrete action
(click, type, hotkey, scroll, wait, or finished).

UI-TARS is a *native* GUI agent: it directly maps a screenshot + a
free-form instruction to a structured action string like
``click(120, 340)`` or ``type('hello world')`` or ``hotkey('ctrl+c')``.
No JSON schema, no system-prompt overhead, no chain-of-thought. That
is why it is the right model for the 200-500 ms per-step loop.

The model is fp16 (~10 GB on disk) and is loaded with bitsandbytes
NF4 4-bit so it fits in ~2.5 GB on the dGPU. Per the user's spec
"STT + VLA + VLM + wake all on GPU, minimal CPU offload", we keep
most of the model on GPU and only spill the largest layers to CPU
if needed.

We re-implement the official UI-TARS action grammar so we can parse
the model output into typed dataclasses and dispatch to the driver.
Reference: https://github.com/bytedance/UI-TARS
"""
from __future__ import annotations
import os

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "ui-tars-2b-sft"


@dataclass
class ClickAction:
    x: int
    y: int
    button: str = "left"
    count: int = 1

    def __str__(self) -> str:
        return f"click({self.x}, {self.y})"


@dataclass
class TypeAction:
    text: str

    def __str__(self) -> str:
        return f"type({self.text!r})"


@dataclass
class HotkeyAction:
    keys: list[str]

    def __str__(self) -> str:
        return f"hotkey({','.join(self.keys)})"


@dataclass
class ScrollAction:
    dx: int
    dy: int

    def __str__(self) -> str:
        return f"scroll({self.dx}, {self.dy})"


@dataclass
class WaitAction:
    seconds: float = 1.0

    def __str__(self) -> str:
        return f"wait({self.seconds})"


@dataclass
class FinishedAction:
    summary: str = ""

    def __str__(self) -> str:
        return f"finished({self.summary!r})"


Action = ClickAction | TypeAction | HotkeyAction | ScrollAction | WaitAction | FinishedAction


SYSTEM_PROMPT = """You are a GUI agent executing ONE todo item at a time on Pop!_OS / COSMIC.

You will be given a single, atomic todo item and the current screenshot. Output ONE action.

Allowed actions (one per turn):
  click(x, y)        Click at the given pixel coordinates. Use 'left' (default) | 'right' | 'middle'.
  type(text)         Type the given text into the focused element.
  hotkey(k1, k2,..)  Press a hotkey combination. Example: hotkey('ctrl', 'l').
  scroll(dx, dy)     Scroll. dy positive = scroll down.
  wait()             Wait briefly for the screen to settle.
  finished()         The todo item is complete and the check has passed.

Output ONLY the action call. No commentary, no coordinates outside the screen, no
destructive actions (rm -rf, chmod 777, sysrq). If the target is not visible, use scroll()
or wait() and try again next turn."""


class Grounder:
    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        device: str = "cuda:0",
        max_memory: dict | None = None,
        dtype: torch.dtype = torch.bfloat16,
        gpu_mem_cap: str = "1.8GiB",
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
        # UI-TARS-2B fp16 = 10 GB. NF4 = 2.5 GB. We cap GPU to 2 GB,
        # letting ~500 MB spill to CPU for the largest tensors. This
        # keeps the model "mostly on GPU" per the user's request.
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
        # UI-TARS-2B in NF4 is ~1.5 GB on disk. The bnb compute
        # overhead pushes the in-memory footprint to ~1.8 GB. We cap
        # GPU at 2.0 GiB so the whole model fits there when the
        # cozy voice stack isn't loaded. When it IS loaded, lower
        # the cap via the constructor and accelerate will spill the
        # tail to CPU (with the standard bnb meta-tensor caveat).
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(self.model_dir),
            quantization_config=bnb,
            torch_dtype=self.dtype,
            device_map="auto",
            max_memory=self.max_memory,
            low_cpu_mem_usage=True,
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

    def step(
        self,
        todo,
        screenshot: Image.Image,
        history: Optional[list[str]] = None,
        target_hint: str = "",
    ) -> Action:
        """Predict the next action for a single :class:`TodoItem`."""
        assert self._model is not None, "call .load() first"
        history = history or []
        todo_str = f"action={todo.action}, target={todo.target}"
        if target_hint:
            todo_str += f", hint={target_hint}"
        history_block = ""
        if history:
            history_block = "Previous actions in this todo: " + " | ".join(history) + "\n"
        user_text = f"Todo: {todo_str}\n{history_block}Next action:"
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, _ = process_vision_info(messages)
        except Exception:
            image_inputs = [screenshot]
        inputs = self._processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(self._model.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            ids = self._model.generate(
                **inputs, max_new_tokens=64, do_sample=False, temperature=0.0
            )
        dt_ms = (time.perf_counter() - t0) * 1000
        out_ids = ids[:, inputs.input_ids.shape[1]:]
        raw = self._processor.batch_decode(
            out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        action = self._parse(raw)
        action.latency_ms = getattr(action, "latency_ms", dt_ms)
        return action

    @staticmethod
    def _parse(raw: str) -> Action:
        m = re.search(r"([a-zA-Z_]+)\s*\((.*?)\)", raw, re.DOTALL)
        if not m:
            return WaitAction(seconds=1.0)
        name, args = m.group(1).lower(), m.group(2)
        try:
            if name == "click":
                nums = re.findall(r"-?\d+", args)
                if len(nums) < 2:
                    return WaitAction()
                x, y = int(nums[0]), int(nums[1])
                button = "left"
                if "right" in args.lower():
                    button = "right"
                elif "middle" in args.lower():
                    button = "middle"
                return ClickAction(x=x, y=y, button=button)
            if name == "type":
                m2 = re.search(r"""['"](.*?)['"]""", args, re.DOTALL)
                text = m2.group(1) if m2 else args.strip()
                return TypeAction(text=text)
            if name == "hotkey":
                parts = re.findall(r"""['"]([^'"]+)['"]""", args)
                if not parts:
                    parts = [p.strip() for p in args.split(",") if p.strip()]
                return HotkeyAction(keys=parts)
            if name == "scroll":
                nums = re.findall(r"-?\d+", args)
                dx, dy = (int(nums[0]), int(nums[1])) if len(nums) >= 2 else (0, 100)
                return ScrollAction(dx=dx, dy=dy)
            if name in ("wait", "sleep"):
                nums = re.findall(r"[\d.]+", args)
                sec = float(nums[0]) if nums else 1.0
                return WaitAction(seconds=sec)
            if name in ("finished", "done", "stop", "complete"):
                m2 = re.search(r"""['"](.*?)['"]""", args, re.DOTALL)
                return FinishedAction(summary=m2.group(1) if m2 else "")
        except Exception:
            return WaitAction()
        return WaitAction(seconds=1.0)
