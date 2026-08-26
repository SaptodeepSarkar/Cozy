#!/usr/bin/env python
"""Control experiment: convert STOCK whisper-small to GGML and verify
whisper.cpp decodes it correctly on this machine. Isolates build/runtime
issues from finetuned-weight issues.
Run from stt-finetune/: .venv/bin/python scripts/ggml_base_control.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402

WCLI = ROOT / "third_party" / "whisper.cpp" / "build" / "bin" / "whisper-cli"
CONVERT = ROOT / "third_party" / "whisper.cpp" / "models" / "convert-h5-to-ggml.py"


def main():
    from huggingface_hub import snapshot_download
    base = Path(snapshot_download(
        "openai/whisper-small",
        allow_patterns=["*.json", "*.txt", "*.model", "pytorch_model.bin"]))

    work = Path("/tmp/base_hf")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    skip = {"flax_model.msgpack", "tf_model.h5", ".gitattributes", "README.md"}
    for f in base.iterdir():
        if f.name not in skip:
            shutil.copy2(f, work / f.name)

    outdir = Path("/tmp/base_ggml")
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    r = subprocess.run([sys.executable, str(CONVERT), str(work),
                        str(ROOT / "third_party" / "openai-whisper"),
                        str(outdir)], capture_output=True, text=True)
    print(r.stdout[-400:], r.stderr[-400:])
    assert (outdir / "ggml-model.bin").exists(), "conversion failed"

    wav = str(ROOT / "recordings" / "session_1" / "000.wav")
    r = subprocess.run([str(WCLI), "-m", str(outdir / "ggml-model.bin"),
                        "-f", wav, "-l", "en", "-np"],
                       capture_output=True, text=True)
    print("BASE-GGML DECODE:", r.stdout.strip()[-200:] or r.stderr.strip()[-200:])


if __name__ == "__main__":
    main()
