"""Shared helpers for the Cozy STT finetuning pipeline."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Everything lives inside stt-finetune/ — nothing leaks to ~/.cache
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_MODEL = os.environ.get("COZY_BASE_MODEL", "openai/whisper-small")
FLEURS_DIR = ROOT / "data" / "fleurs" / "en_in"
MANIFEST_DIR = ROOT / "data" / "manifests"
RECORDINGS_DIR = ROOT / "recordings"
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "output"

SAMPLE_RATE = 16000


def english_normalizer():
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    # Basic spell-check dictionary is optional; the plain normalizer is fine here.
    return EnglishTextNormalizer({})


def wer(pred_texts, ref_texts):
    """Word Error Rate via jiwer, robust to empty strings."""
    import jiwer
    refs = [t.strip() for t in ref_texts]
    hyps = [h.strip() if h.strip() else "<empty>" for h in pred_texts]
    return 100.0 * jiwer.wer(refs, hyps)


def read_manifest(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
