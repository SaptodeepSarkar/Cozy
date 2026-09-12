"""Cozy STT wrapper: fast CT2 engine with automatic HF fallback.

Usage:
    from stt import CozySTT
    stt = CozySTT()
    text = stt.transcribe_array(samples_16k_float32)   # or .transcribe_file(path)
"""
import ctypes
import os
import sys
from pathlib import Path

STT_ROOT = Path(__file__).resolve().parent.parent / "stt-finetune"
_CT2_V12 = STT_ROOT / "output" / "cozy_stt_v1.2_ct2_int8"
_HF_V12 = STT_ROOT / "output" / "hf_finetuned_v1.2"
CT2_DIR = _CT2_V12 if (_CT2_V12 / "model.bin").exists() else STT_ROOT / "output" / "cozy_stt_v1_ct2_int8"
HF_DIR = _HF_V12 if (_HF_V12 / "model.safetensors").exists() else STT_ROOT / "output" / "hf_finetuned"

sys.path.insert(0, str(STT_ROOT / "scripts"))


def _preload_ct2_cuda12():
    """Load CTranslate2's CUDA 12 BLAS beside a CUDA 13 system install."""
    try:
        import nvidia.cublas
    except ImportError:
        return
    # NVIDIA's wheels expose ``nvidia.cublas`` as a namespace package, so
    # ``__file__`` is normally None.  Its package path points at the directory
    # that owns the bundled ``lib/`` tree.
    package_paths = list(nvidia.cublas.__path__)
    if not package_paths:
        return
    lib_dir = Path(package_paths[0]).resolve() / "lib"
    for name in ("libcublasLt.so.12", "libcublas.so.12"):
        path = lib_dir / name
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


class CozySTT:
    def __init__(self, prefer_engine="auto"):
        self._ct2 = None
        self._ct2_device = None
        self._hf = None
        if prefer_engine not in {"auto", "ct2", "hf"}:
            raise ValueError("prefer_engine must be auto, ct2, or hf")
        self.prefer = prefer_engine
        self.last_engine = None
        self.ct2_error = None

    # ---- engines -------------------------------------------------------
    def _get_ct2(self):
        if self._ct2 is None:
            from faster_whisper import WhisperModel
            import torch
            if torch.cuda.is_available():
                _preload_ct2_cuda12()
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
                str(HF_DIR), torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
            self._librosa = librosa
        return self._hf

    # ---- public --------------------------------------------------------
    def transcribe_file(self, path, hinglish_hint=False):
        import librosa
        audio, _ = librosa.load(str(path), sr=16000, mono=True)
        return self.transcribe_array(audio, hinglish_hint=hinglish_hint)

    def transcribe_array(self, audio, hinglish_hint=False):
        """audio: float32 16 kHz mono."""
        import numpy as np
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("audio must be finite mono samples at 16 kHz")
        if not audio.size or not np.any(audio):
            return ""
        configured_language = os.environ.get("COZY_STT_LANGUAGE", "en").strip()
        self._language = None if hinglish_hint or configured_language.lower() == "auto" else configured_language
        if self.prefer in ("auto", "ct2") and CT2_DIR.exists():
            try:
                text = self._run_ct2(audio)
                self.last_engine = "ct2-cpu" if self._ct2_device == "cpu" else "ct2"
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
                    self._ct2_device = "cpu"
                    text = self._run_ct2(audio)
                    self.last_engine = "ct2-cpu"
                    return text
                except (OSError, RuntimeError, ImportError):
                    self._ct2 = None
        if HF_DIR.exists():
            text = self._run_hf(audio)
            self.last_engine = "hf"
            return text
        raise RuntimeError("No finetuned STT model found under stt-finetune/output")

    def warmup(self):
        """Load the engine and execute one small inference during startup."""
        import numpy as np
        if self.prefer == "hf" or not CT2_DIR.exists():
            self._get_hf()
            return
        silence = np.zeros(16000, dtype=np.float32)
        try:
            self._run_ct2(silence)
        except (OSError, RuntimeError, ImportError) as exc:
            self.ct2_error = str(exc)
            try:
                from faster_whisper import WhisperModel
                self._ct2 = WhisperModel(str(CT2_DIR), device="cpu", compute_type="int8")
                self._ct2_device = "cpu"
                self._run_ct2(silence)
            except (OSError, RuntimeError, ImportError):
                self._ct2 = None
                if not HF_DIR.exists():
                    raise
                self._get_hf()

    # ---- internals -----------------------------------------------------
    def _run_ct2(self, audio):
        model = self._get_ct2()
        # Capture already performs endpointing. A second VAD pass here can
        # remove quiet first/last words, and beam=3 adds avoidable latency for
        # short desktop commands.
        segs, _ = model.transcribe(audio, language=getattr(self, "_language", "en"), beam_size=1,
                                   condition_on_previous_text=False, vad_filter=False)
        return " ".join(s.text.strip() for s in segs).strip()

    def _run_hf(self, audio):
        import numpy as np
        import torch
        model = self._get_hf()
        feats = self._hf_proc(np.asarray(audio, dtype=np.float32),
                              sampling_rate=16000,
                              return_tensors="pt").input_features
        feats = feats.to(device=model.device, dtype=model.dtype)
        ids = model.generate(feats, language=getattr(self, "_language", "en"), task="transcribe",
                             max_new_tokens=224)
        return self._hf_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def close(self):
        """Release CT2/CUDA objects while the Python runtime is intact."""
        self._ct2 = None
        self._hf = None
        self._hf_proc = None
        self._librosa = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


if __name__ == "__main__":
    stt = CozySTT()
    print(stt.transcribe_file(sys.argv[1] if len(sys.argv) > 1
                              else "/tmp/cozy_mic.wav"))
