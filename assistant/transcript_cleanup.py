"""ArchFlow-compatible transcript cleanup for Cozy's voice input.

The deterministic pass is always available.  On the owner's workstation the
same Qwen3-0.6B + cleanup LoRA used by ArchFlow is loaded once and reused for
longer utterances; missing external artifacts safely degrade to local polish.
"""
from __future__ import annotations

import os
import re
import threading
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHFLOW_CLEANUP = HERE.parent.parent / "ArchFlow" / "training" / "cleanup-llm" / "output"

FILLERS = {"uh", "uhh", "uhhh", "um", "umm", "ummm", "uhm", "er", "erm", "ah", "mmm"}
INTENTIONAL_DOUBLES = {
    "no", "yes", "yeah", "yep", "nope", "oh", "ha", "hey", "hi", "hello",
    "well", "so", "very", "really", "quite", "far", "long", "many", "much",
    "more", "most", "again", "over", "bye", "please", "thanks", "sorry",
}
NEGATIONS = {"no", "not", "never", "none", "nothing", "don't", "doesn't", "didn't", "can't", "cannot", "won't"}

SYSTEM = (
    "You are a source-grounded transcript editor. Fix grammar, punctuation, "
    "capitalization, sentence boundaries, filler words, false starts, duplicate "
    "phrases, and obvious spelling mistakes. Preserve content words, names, "
    "numbers, dates, quantities, units, code, paths, negation, profanity, and the "
    "original language. Do not add, remove, reorder, translate, summarize, or "
    "reinterpret content. If unsure, return the input unchanged."
)


def _core(word: str) -> str:
    return re.sub(r"^\W+|\W+$", "", word, flags=re.UNICODE).lower()


def polish_transcript(text: str) -> str:
    """Remove fillers, false starts and accidental 1–3 word repetitions."""
    words = [word for word in text.split() if not (len(_core(word)) >= 2 and _core(word) in FILLERS)]
    i = 0
    while i + 1 < len(words):
        first, second = _core(words[i]), _core(words[i + 1])
        if len(first) >= 4 and len(second) > len(first) and second.startswith(first):
            words.pop(i)
        else:
            i += 1
    for size in (3, 2, 1):
        i = 0
        while i + 2 * size <= len(words):
            left = [_core(word) for word in words[i:i + size]]
            right = [_core(word) for word in words[i + size:i + 2 * size]]
            exempt = size == 1 and left[0] in INTENTIONAL_DOUBLES
            if left == right and not exempt:
                del words[i + size:i + 2 * size]
            else:
                i += 1
    out = " ".join(words)
    out = re.sub(r"\s+([,.!?;:)\]])", r"\1", out)
    out = re.sub(r"([\[(])\s+", r"\1", out)
    return re.sub(r"\s+", " ", out).strip()


def _safe_rewrite(raw: str, cleaned: str) -> bool:
    """Reject likely cleanup hallucinations before they reach the command LLM."""
    if not cleaned or "<|" in cleaned or len(cleaned) > max(80, int(len(raw) * 1.8)):
        return False
    if len(cleaned) < int(len(raw) * 0.45):
        return False
    numbers = lambda value: Counter(re.findall(r"\d+(?:\.\d+)?", value))
    if numbers(raw) != numbers(cleaned):
        return False
    negations = lambda value: Counter(_core(word) for word in value.split() if _core(word) in NEGATIONS)
    if negations(raw) != negations(cleaned):
        return False
    normalize = lambda value: " ".join(re.findall(r"[\w']+", value.lower()))
    return SequenceMatcher(None, normalize(raw), normalize(cleaned)).ratio() >= 0.55


class TranscriptCleaner:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.lock = threading.Lock()
        self.threshold = max(1, int(os.environ.get("COZY_CLEANUP_WORD_THRESHOLD", "10")))
        self.model_dir = Path(os.environ.get("COZY_CLEANUP_MODEL", ARCHFLOW_CLEANUP / "base-model"))
        self.adapter_dir = Path(os.environ.get("COZY_CLEANUP_ADAPTER", ARCHFLOW_CLEANUP / "dpo-sft"))
        self.status = "deterministic"
        self.load_error = ""

    def load(self) -> None:
        """Load the ArchFlow cleanup LoRA once; rules remain the fallback."""
        if not (self.model_dir / "config.json").exists() or not (self.adapter_dir / "adapter_model.safetensors").exists():
            return
        try:
            import torch
            if not torch.cuda.is_available():
                self.status = "deterministic (CUDA unavailable)"
                return
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), trust_remote_code=True)
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            base = AutoModelForCausalLM.from_pretrained(
                str(self.model_dir), dtype=torch.bfloat16, trust_remote_code=True,
                attn_implementation="sdpa",
            )
            self.model = PeftModel.from_pretrained(base, str(self.adapter_dir)).to("cuda").eval()
            self.model.config.use_cache = True
            self.torch = torch
            self.status = f"Qwen3-0.6B + {self.adapter_dir.name}"
        except Exception as exc:
            self.model = None
            self.tokenizer = None
            self.load_error = str(exc)
            self.status = "deterministic (LLM unavailable)"

    def clean(self, text: str) -> str:
        polished = polish_transcript(text)
        if self.model is None or len(polished.split()) < self.threshold:
            return polished
        try:
            with self.lock, self.torch.inference_mode():
                prompt = (
                    f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                    f"<|im_start|>user\n{polished}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=min(192, max(48, len(polished.split()) * 3)),
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
                cleaned = self.tokenizer.decode(
                    output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                ).strip()
        except Exception:
            return polished
        return cleaned if _safe_rewrite(polished, cleaned) else polished

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
