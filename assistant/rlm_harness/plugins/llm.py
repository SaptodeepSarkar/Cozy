"""LLM plugin: Qwen3-0.6B + DPO adapter, loaded on first use.

Memory budget:
  - cozy-llm-v1 base: 1.2 GB VRAM (bf16)
  - cozy-llm-v1-dpo: 40 MB + small merge overhead
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

from ..harness_fast import Plugin, ASSISTANT
from ..fasttool import extract_tool_call


def json_emit_safe(kind, **fields):
    """Emit a JSON event if --json-events is set. Falls back silently otherwise."""
    if os.environ.get("COZY_TUI_MODE") != "node":
        return
    ev = {"kind": kind, "ts": time.time(), **fields}
    try:
        sys.stdout.write(json.dumps(ev, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


class LLMPlugin(Plugin):
    name = "llm"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._model = None
        self._tok = None
        self._device = "cuda"
        self._model_dir = ASSISTANT / "model" / "cozy-llm-v1"
        self._dpo_dir = ASSISTANT / "model" / "cozy-llm-v1-dpo"
        self._use_dpo = (self._dpo_dir / "adapter_model.safetensors").exists()
        self._tool_schema = None

    def _do_load(self):
        if not (self._model_dir / "model.safetensors").exists():
            raise FileNotFoundError(
                f"LLM weights missing: {self._model_dir}/model.safetensors. "
                f"Run assistant/sft_qwen.py first."
            )
        import torch
        import contextlib, io
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from ._base import json_emit_safe as _emit
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            extra = {}
            cfg_path = self._model_dir / "tokenizer_config.json"
            if cfg_path.exists():
                try:
                    raw = json.loads(cfg_path.read_text()).get("extra_special_tokens", [])
                    extra = {str(token): str(token) for token in raw} if isinstance(raw, list) else {}
                except (OSError, ValueError, TypeError):
                    pass
            self._tok = AutoTokenizer.from_pretrained(str(self._model_dir), extra_special_tokens=extra)
            base = AutoModelForCausalLM.from_pretrained(
                str(self._model_dir), torch_dtype=torch.bfloat16,
                attn_implementation="sdpa")
            if self._use_dpo:
                base = PeftModel.from_pretrained(base, str(self._dpo_dir))
                base = base.merge_and_unload()
        self._model = base.to(self._device).eval()
        self._torch = torch
        _emit("warmup", model="llm", state="done")

    def _do_free(self):
        if self._model is not None:
            del self._model
        if hasattr(self, "_torch") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._model = None
        self._tok = None
        import gc; gc.collect()

    def generate(self, messages, max_new_tokens=200, on_token=None):
        assert self._loaded, "call .load() first"
        import re
        # Tool definitions are static across turns; parse once so generation
        # spends its time on model tokens rather than filesystem/JSON work.
        if self._tool_schema is None:
            self._tool_schema = json.loads(
                (ASSISTANT.parent / "team" / "tool_schema.json").read_text())["tools"]
        schema = self._tool_schema
        prompt = self._tok.apply_chat_template(
            messages, tools=schema, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = self._tok(prompt, return_tensors="pt").to(self._device)
        if on_token is None:
            with self._torch.inference_mode():
                out = self._model.generate(
                    **ids, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self._tok.eos_token_id, use_cache=True)
            text = self._tok.decode(
                out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False).strip()
        else:
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(
                self._tok, skip_prompt=True, skip_special_tokens=False)
            gen_kwargs = dict(
                input_ids=ids["input_ids"],
                attention_mask=ids["attention_mask"],
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tok.eos_token_id,
                use_cache=True,
            )
            import threading
            t = threading.Thread(
                target=self._model.generate, kwargs=gen_kwargs)
            t.start()
            chunks = []
            for piece in streamer:
                on_token(piece)
                chunks.append(piece)
            t.join()
            text = "".join(chunks).strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        self.touch()
        return text
