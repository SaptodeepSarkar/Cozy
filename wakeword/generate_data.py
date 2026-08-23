#!/usr/bin/env python3
"""Generates all training audio for the Cozy wake word.

Buckets produced
  work/synthetic/          synthetic "cozy" positives (bulk, git-ignored)
  work/similar/<word>/     hard negatives that sound like "cozy" (bulk)
  work/negative/           everyday speech that must never trigger (bulk)
  data/cozy/               + small seed subset copied here (kept in git)
  data/similar/            + small seed subset copied here (kept in git)

data/cozy additionally holds YOUR real recordings made with record_samples.py;
those are always included in training automatically.

Usage
  python generate_data.py                       # full dataset from config.yaml
  python generate_data.py --mode smoke          # tiny fast validation run
  python generate_data.py --only similar --force
"""
from __future__ import annotations

import argparse
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from download_models import MULTISPEAKER_NAME, VOICES_DIR, WORK  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text())
SR = int(CFG["audio"]["sample_rate"])
CLIP = int(float(CFG["audio"]["clip_seconds"]) * SR)
MULTISPEAKER_PT = WORK / "models" / MULTISPEAKER_NAME

# Everyday sentences the detector must stay silent on.
NEGATIVE_PHRASES = [
    "hey there", "how are you", "what time is it", "open the browser",
    "play some music", "set an alarm for seven", "remember to buy milk",
    "call me back later", "the meeting starts soon", "send that file over",
    "turn off the lights", "what is the weather today",
    "remind me about the gym", "lower the volume please",
    "search for flights to Delhi", "take a screenshot", "lock the screen",
    "good morning everyone", "let us start the demo",
    "where did I put my keys", "add this to the shopping list",
    "book a table for two", "navigate to the office", "read me the headlines",
    "skip this song", "who is calling me", "type this message for me",
    "close all the tabs", "restart the laptop", "check my calendar",
    "that will be all thanks", "tell me a joke", "translate this paragraph",
    "summarize the article", "mute the microphone",
    "increase the brightness", "connect to the bluetooth speaker",
    "order my usual coffee", "join the video call", "share my screen now",
    "save this document", "print two copies", "empty the recycle bin",
    "update the system", "backup my photos", "find my phone",
    "stop the timer", "draft an email to the team",
]


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


def voice_models() -> list:
    onnx = sorted(VOICES_DIR.glob("*.onnx"))
    if not onnx:
        raise SystemExit("[generate_data] no Piper voices found - "
                         "run download_models.py first")
    return onnx


def multispeaker_ready() -> bool:
    enabled = CFG["piper"].get("multispeaker_generator", True)
    return bool(enabled) and MULTISPEAKER_PT.exists()


def piper_generate(text, scratch, count, models, batch_size, max_speakers):
    """Runs the piper-sample-generator CLI into a clean scratch dir."""
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    cmd = [sys.executable, "-m", "piper_sample_generator", text,
           "--max-samples", str(count),
           "--output-dir", str(scratch),
           "--batch-size", str(batch_size)]
    for model in models:
        cmd += ["--model", str(model)]
    if max_speakers is not None:
        cmd += ["--max-speakers", str(max_speakers)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wavs = sorted(scratch.glob("*.wav"))
    if not wavs:
        tail_out = (proc.stdout or "")[-1500:]
        tail_err = (proc.stderr or "")[-1500:]
        sys.stderr.write(tail_out)
        sys.stderr.write("\n")
        sys.stderr.write(tail_err)
        sys.stderr.write("\n")
        raise SystemExit("[generate_data] piper produced nothing for "
                         + repr(text))
    return wavs


def postprocess(src, rng):
    """16 kHz mono, silence-trimmed, fixed length float32 in [-1, 1]."""
    pcm, sr = sf.read(str(src), dtype="float32", always_2d=True)
    pcm = pcm.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        divisor = math.gcd(sr, SR)
        pcm = resample_poly(pcm, SR // divisor, sr // divisor)
    mag = np.abs(pcm)
    loud = np.where(mag > max(1e-4, float(mag.max()) * 0.02))[0]
    if loud.size == 0:
        return None
    pcm = pcm[loud[0]:loud[-1] + 1]
    if pcm.size >= CLIP:
        start = int(rng.integers(0, pcm.size - CLIP + 1))
        return pcm[start:start + CLIP]
    pad = CLIP - pcm.size
    head = int(rng.integers(0, pad + 1))
    return np.pad(pcm, (head, pad - head))


def save(pcm, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), pcm.astype(np.float32), SR, subtype="PCM_16")


def synth_bucket(texts, out_dir, file_stem, tgt, models, max_speakers,
                 batch_size, rng):
    """Generates tgt processed clips cycling through texts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    round_no = 0
    while made < tgt and round_no < 80:
        text = texts[round_no % len(texts)]
        want = min(max(batch_size * 4, 64), tgt - made)
        wavs = piper_generate(text, WORK / "scratch" / file_stem, want,
                              models, batch_size, max_speakers)
        for wav in wavs:
            if made >= tgt:
                break
            pcm = postprocess(wav, rng)
            if pcm is None:
                continue
            dest = out_dir / (file_stem + "_" + format(made, "05d") + ".wav")
            save(pcm, dest)
            made += 1
        round_no += 1
    return made


def copy_seeds(src_dir, dest_dir, prefix, count):
    if count <= 0:
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    for stale in dest_dir.glob(prefix + "_*.wav"):
        stale.unlink()
    seeds = sorted(src_dir.glob("*.wav"))[:count]
    for i, wav in enumerate(seeds):
        name = prefix + "_" + format(i, "03d") + ".wav"
        shutil.copy2(wav, dest_dir / name)


def cached_count(out_dir):
    if not out_dir.exists():
        return 0
    return len(list(out_dir.glob("*.wav")))


def targets(mode):
    counts = dict(CFG["counts"])
    if mode == "smoke":
        counts.update({
            "positive": 24,
            "similar_per_word": 6,
            "random_negative": 24,
            "seed_copy": min(int(counts["seed_copy"]), 3),
        })
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument("--only", default="positive,similar,negative",
                        help="comma list: positive,similar,negative")
    parser.add_argument("--no-multispeaker", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="regenerate even if enough files already exist")
    args = parser.parse_args()
    wanted = set(part.strip() for part in args.only.split(","))

    rng = np.random.default_rng(2025)
    t = targets(args.mode)

    gpu = False
    try:
        import torch
        gpu = torch.cuda.is_available()
    except ImportError:
        gpu = False
    bs_cfg = int(CFG["piper"]["batch_size"])
    batch_size = bs_cfg if gpu else min(bs_cfg, 8)
    print("[generate_data] mode=" + args.mode + " gpu=" + str(gpu)
          + " batch=" + str(batch_size))

    voices = voice_models()
    variants = list(CFG["wake_word"]["phonetic_variants"])

    if "positive" in wanted:
        use_ms = multispeaker_ready() and not args.no_multispeaker
        models = [MULTISPEAKER_PT] if use_ms else voices
        ms = int(CFG["piper"]["max_speakers"]) if use_ms else None
        engine = "multi-speaker generator" if use_ms else "voice bank"
        print("[generate_data] positives: cozy x"
              + str(t["positive"]) + " (" + engine + ")")
        out = WORK / "synthetic"
        if args.force or cached_count(out) < t["positive"]:
            n = synth_bucket(variants, out, "cozy", t["positive"], models,
                             ms, batch_size, rng)
            print("  wrote " + str(n))
        else:
            print("  = cached")
        copy_seeds(out, HERE / "data" / "cozy", "synth", int(t["seed_copy"]))
        print("  +" + str(t["seed_copy"]) + " seeds copied to data/cozy")

    if "similar" in wanted:
        print("[generate_data] hard negatives: similar words x"
              + str(t["similar_per_word"]) + " each")
        for word in CFG["similar_words"]:
            out = WORK / "similar" / slug(word)
            if args.force or cached_count(out) < t["similar_per_word"]:
                n = synth_bucket([word], out, slug(word),
                                 t["similar_per_word"], voices, None,
                                 batch_size, rng)
                print("  + " + word + ": " + str(n))
            else:
                print("  = " + word + ": cached")
            copy_seeds(out, HERE / "data" / "similar", "synth_" + slug(word),
                       int(t["seed_copy"]))

    if "negative" in wanted:
        out = WORK / "negative"
        if args.force or cached_count(out) < t["random_negative"]:
            print("[generate_data] random negatives: "
                  + str(t["random_negative"]) + " everyday sentences")
            phrases = list(NEGATIVE_PHRASES)
            random.Random(7).shuffle(phrases)
            n = synth_bucket(phrases, out, "neg", t["random_negative"],
                             voices, None, batch_size, rng)
            print("  wrote " + str(n))
        else:
            print("[generate_data] random negatives: cached")

    print("")
    print("[generate_data] done. Next: python train_wakeword.py")


if __name__ == "__main__":
    main()
