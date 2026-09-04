"""Cozy STT wrapper: fast CT2 engine with automatic HF fallback.

Usage:
    from stt import CozySTT
    stt = CozySTT()
    text = stt.transcribe_array(samples_16k_float32)   # or .transcribe_file(path)
"""
import sys
from pathlib import Path

STT_ROOT = Path(__file__).resolve().parent.parent / "stt-finetune"
CT2_DIR = STT_ROOT / "output" / "cozy_stt_v1_ct2_int8"
HF_DIR = STT_ROOT / "output" / "hf_finetuned"

sys.path.insert(0, str(STT_ROOT / "scripts"))


class CozySTT:
    def __init__(self, prefer_engine="auto"):
        self._ct2 = None
        self._ct2_device = None
        self._hf = None
        self.prefer = prefer_engine
        self.last_engine = None

    # ---- engines -------------------------------------------------------
    def _get_ct2(self):
        if self._ct2 is None:
            from faster_whisper import WhisperModel
            import torch
            if torch.cuda.is_available():
                self._ct2 = WhisperModel(str(CT2_DIR), device="cuda",
                                         device_index=0, compute_type="int8_float16")
                self._ct2_device = "cuda"
            else:
                # CTranslate2 is still much faster than loading the HF
                # Whisper fallback and works reliably on CPU-only hosts.
                self._ct2 = WhisperModel(str(CT2_DIR), device="cpu",
                                         compute_type="int8")
                self._ct2_device = "cpu"
        return self._ct2

    def _get_hf(self):
        if self._hf is None:
            import librosa
            import torch
            from transformers import (WhisperForConditionalGeneration,
                                      WhisperProcessor)
            self._hf_proc = WhisperProcessor.from_pretrained(str(HF_DIR))
            self._hf = WhisperForConditionalGeneration.from_pretrained(
                str(HF_DIR), torch_dtype=torch.float16).to("cuda")
            self._librosa = librosa
        return self._hf

    # ---- public --------------------------------------------------------
    def transcribe_file(self, path, hinglish_hint=False):
        import librosa
        audio, _ = librosa.load(str(path), sr=16000, mono=True)
        return self.transcribe_array(audio)

    def transcribe_array(self, audio, hinglish_hint=False):
        """audio: float32 16 kHz mono."""
        if self.prefer in ("auto", "ct2") and CT2_DIR.exists():
            try:
                text = self._run_ct2(audio)
                if text:
                    self.last_engine = "ct2"
                    return text
            except (OSError, RuntimeError, ImportError) as exc:
                # CTranslate2 wheels are commonly built for CUDA 12 while a
                # host may have CUDA 13 (libcublas.so.12 missing). Do not
                # strand the voice loop: fall through to the HF CUDA engine.
                self.ct2_error = str(exc)
                # CUDA wheels may be installed without the matching cuBLAS
                # runtime (for example libcublas.so.12). Retry this compact
                # int8 model on CPU instead of falling back to a 16-second
                # HF model load for every first utterance.
                try:
                    from faster_whisper import WhisperModel
                    self._ct2 = WhisperModel(str(CT2_DIR), device="cpu", compute_type="int8")
                    text = self._run_ct2(audio)
                    if text:
                        self.last_engine = "ct2-cpu"
                        return text
                except (OSError, RuntimeError, ImportError):
                    self._ct2 = None
        if HF_DIR.exists():
            text = self._run_hf(audio)
            self.last_engine = "hf"
            return text
        raise RuntimeError("No finetuned STT model found under stt-finetune/output")

    # ---- internals -----------------------------------------------------
    def _run_ct2(self, audio):
        model = self._get_ct2()
        segs, _ = model.transcribe(audio, language="en", beam_size=1)
        return " ".join(s.text.strip() for s in segs).strip()

    def _run_hf(self, audio):
        import numpy as np
        import torch
        model = self._get_hf()
        feats = self._hf_proc(np.asarray(audio, dtype=np.float32),
                              sampling_rate=16000,
                              return_tensors="pt").input_features
        feats = feats.to("cuda", torch.float16)
        ids = model.generate(feats, language="english", task="transcribe",
                             max_new_tokens=224)
        return self._hf_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


if __name__ == "__main__":
    stt = CozySTT()
    print(stt.transcribe_file(sys.argv[1] if len(sys.argv) > 1
                              else "/tmp/cozy_mic.wav"))
