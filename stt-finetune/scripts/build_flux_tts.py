#!/usr/bin/env python3
"""Synthesize the hard-case sentence set with Deepgram Flux TTS via OpenRouter.

Generates 125 sentences x 4 voices = 500 clips under data/flux_tts/:
  data/flux_tts/wav/clip_XXXXX.wav   (16 kHz mono PCM16)
  data/flux_tts/manifest.jsonl       (rows: audio_path, text, source=flux_tts, ...)

Voices: flux-priya-en (IN-F), flux-naveen-en (IN-M),
        flux-alexis-en (US-F), flux-marcus-en (US-M)

Eval split: every 5th sentence (idx % 5 == 0) -> all 4 voices to eval,
so eval tests lexical generalisation. 400 train / 100 eval.

Usage:
  OPENROUTER_API=... .venv/bin/python scripts/build_flux_tts.py
  .venv/bin/python scripts/build_flux_tts.py --limit 8   # smoke test (8 clips)
  .venv/bin/python scripts/build_flux_tts.py --resume    # default: skip existing

Rate limits: free tier -> 3 workers, exponential backoff on 429/5xx.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR  # noqa: E402

OUT_DIR = DATA_DIR / "flux_tts"
WAV_DIR = OUT_DIR / "wav"
MP3_DIR = OUT_DIR / "mp3"
SENT_FILE = OUT_DIR / "sentences.json"
MANIFEST = OUT_DIR / "manifest.jsonl"

MODEL = "deepgram/flux-tts:free"
VOICES = ["flux-priya-en", "flux-naveen-en", "flux-alexis-en", "flux-marcus-en"]
API_URL = "https://openrouter.ai/api/v1/audio/speech"


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API") or os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    # fall back to repo .env
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("OPENROUTER_API"):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENROUTER_API key not found (env or repo .env)")


def tts_request(session, text: str, voice: str, api_key: str, retries: int = 6) -> bytes:
    import urllib.request
    payload = json.dumps({
        "model": MODEL, "input": text, "voice": voice, "response_format": "mp3",
    }).encode()
    last_err = ""
    for attempt in range(retries):
        req = urllib.request.Request(
            API_URL, data=payload,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json",
                     "HTTP-Referer": "https://github.com/cozy",
                     "X-Title": "Cozy STT finetune"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
                if len(data) < 1000:
                    raise ValueError(f"suspiciously small response ({len(data)}B)")
                return data
        except Exception as e:  # noqa: BLE001 - network flakes, 429s, 5xx
            last_err = f"{type(e).__name__}: {e}"
            # honour Retry-After if present (HTTPError carries headers)
            wait = 2 ** attempt * 2 + random.uniform(0, 2)
            try:
                ra = getattr(e, "headers", {}).get("Retry-After")
                if ra:
                    wait = max(wait, float(ra))
            except Exception:
                pass
            print(f"    [retry {attempt+1}/{retries}] {voice}: {last_err} (wait {wait:.0f}s)",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError(f"TTS failed after {retries} attempts for voice={voice}: {last_err}")


def mp3_to_wav16k(mp3_path: Path, wav_path: Path) -> float:
    """Convert to 16kHz mono PCM16 wav. Returns duration in seconds."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(mp3_path),
         "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(wav_path)],
        capture_output=True, text=True)
    if r.returncode != 0 or not wav_path.exists():
        raise RuntimeError(f"ffmpeg failed for {mp3_path.name}: {r.stderr[:300]}")
    r2 = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)],
        capture_output=True, text=True)
    try:
        return float(r2.stdout.strip())
    except ValueError:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only synthesize first N clips (smoke test)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--no-resume", action="store_true", help="regenerate even if wav exists")
    args = ap.parse_args()

    api_key = get_api_key()
    print(f"API key found (len={len(api_key)})")

    sents = json.loads(SENT_FILE.read_text())["sentences"]
    print(f"{len(sents)} sentences x {len(VOICES)} voices = {len(sents)*len(VOICES)} clips")

    jobs = []
    for si, s in enumerate(sents):
        split = "eval" if (si % 5 == 0) else "train"
        for voice in VOICES:
            idx = si * len(VOICES) + VOICES.index(voice)
            jobs.append({"idx": idx, "sent_id": s["id"], "category": s["category"],
                         "text": s["text"], "voice": voice, "split": split})
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"jobs: {len(jobs)} (train={sum(1 for j in jobs if j['split']=='train')} "
          f"eval={sum(1 for j in jobs if j['split']=='eval')})")

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    MP3_DIR.mkdir(parents=True, exist_ok=True)

    todo = []
    for j in jobs:
        wav = WAV_DIR / f"clip_{j['idx']:05d}.wav"
        if wav.exists() and not args.no_resume:
            j["dur"] = 0.0
            continue
        todo.append(j)
    print(f"resume: {len(jobs)-len(todo)} already done, {len(todo)} to synthesize")

    import urllib.request  # noqa: F401 (kept for clarity)
    failures = []

    def one(j):
        tag = f"clip_{j['idx']:05d}"
        mp3 = MP3_DIR / f"{tag}_{j['voice']}.mp3"
        wav = WAV_DIR / f"{tag}.wav"
        # one clip idx maps to exactly one (sentence, voice) so mp3 name is unique
        audio = tts_request(None, j["text"], j["voice"], api_key)
        mp3.write_bytes(audio)
        dur = mp3_to_wav16k(mp3, wav)
        # keep mp3 for provenance (small); wav is the training artefact
        return dur

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(one, j): j for j in todo}
            done = 0
            for fut in as_completed(futs):
                j = futs[fut]
                done += 1
                try:
                    j["dur"] = round(fut.result(), 2)
                except Exception as e:  # noqa: BLE001
                    failures.append({"idx": j["idx"], "err": str(e)[:200]})
                    print(f"  FAILED clip_{j['idx']:05d} [{j['voice']}]: {e}", flush=True)
                if done % 10 == 0 or done == len(todo):
                    print(f"  progress {done}/{len(todo)}", flush=True)

    # (re)build manifest from all wavs present
    rows, missing = [], 0
    for j in jobs:
        wav = WAV_DIR / f"clip_{j['idx']:05d}.wav"
        if not wav.exists():
            missing += 1
            continue
        dur = j.get("dur", 0.0)
        if not dur:
            try:
                import soundfile as sf
                info = sf.info(str(wav))
                dur = round(info.duration, 2)
            except Exception:
                dur = 0.0
        rows.append({"audio_path": str(wav), "text": j["text"], "source": "flux_tts",
                     "voice": j["voice"], "sent_id": j["sent_id"],
                     "category": j["category"], "split": j["split"],
                     "duration_s": dur})
    with open(MANIFEST, "w") as f:
        for r in sorted(rows, key=lambda r: r["audio_path"]):
            f.write(json.dumps(r) + "\n")
    total_dur = sum(r.get("duration_s", 0) for r in rows)
    print(f"manifest: {len(rows)} rows ({total_dur/60:.1f} min audio), missing={missing}")
    if failures:
        print(f"FAILURES: {len(failures)}")
        for fl in failures[:10]:
            print(f"  {fl}")
    print(f"wavs -> {WAV_DIR}\nmanifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
