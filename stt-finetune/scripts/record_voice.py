#!/usr/bin/env python3
"""
Cozy STT — interactive voice recorder.
Pure stdlib + ffmpeg/arecord (no venv needed). Records 16 kHz mono WAV clips
of you reading the prompts in data/prompts.json.

Usage:
    python3 scripts/record_voice.py            # next unfinished session
    python3 scripts/record_voice.py --session 3  # specific session
    python3 scripts/record_voice.py --list

Per line: read it aloud -> auto-stops when you go silent -> [Enter] keep,
r = retry, s = skip, p = play back, q = quit (progress is saved, resume anytime).
"""
import argparse
import array
import json
import math
import os
import select
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "data" / "prompts.json"
REC_DIR = ROOT / "recordings"
SR = 16000
MAX_SEC = 15          # hard cap per take
SILENCE_SEC = 0.9     # stop after this much trailing quiet
START_SEC = 0.35      # ignore leading silence when trimming
RMS_SIL = 350         # RMS below this counts as silence (tune with --threshold)


def load_progress(session_dir: Path) -> dict:
    f = session_dir / "state.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"done": {}, "skipped": []}


def save_progress(session_dir: Path, st: dict):
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "state.json").write_text(json.dumps(st, indent=1))


def record_take(threshold=RMS_SIL, manual_only=False) -> tuple[Path | None, float]:
    """Record until silence/max (or Enter if manual_only). Returns (wav_path, peak_rms)."""
    cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(SR), "-c", "1", "-t", "raw"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frames = []
    peak = 0
    silent_tail = 0.0
    chunk_n = SR // 10  # 100 ms
    elapsed = 0.0
    if manual_only:
        print("    [recording] press ENTER to stop (no auto-stop) ...", end="", flush=True)
    else:
        print("    [recording] press ENTER to stop, or wait for silence...", end="", flush=True)
    try:
        while True:
            # manual stop: any Enter pressed mid-recording ends the take
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r and sys.stdin.read(1) == "\n":
                break
            raw = proc.stdout.read(chunk_n * 2)
            if not raw:
                break
            samples = array.array("h")
            samples.frombytes(raw)
            n = len(samples)
            if n == 0:
                continue
            rms = math.sqrt(sum((s * s) for s in samples[::4]) / max(1, n // 4))
            peak = max(peak, rms)
            frames.append(samples.tobytes())
            elapsed += 0.1
            if int(elapsed * 10) % 10 == 0:
                if manual_only:
                    print(f"\r    [recording {elapsed:0.0f}s] ENTER=stop ",
                          end="", flush=True)
                else:
                    print(f"\r    [recording {elapsed:0.0f}s] ENTER=stop / silence stops it ",
                          end="", flush=True)
            if not manual_only:
                if rms < threshold:
                    silent_tail += 0.10
                    if len(frames) > 5 and silent_tail >= SILENCE_SEC:
                        break
                else:
                    silent_tail = 0.0
            total = sum(len(f) for f in frames) / 2 / SR
            if total >= MAX_SEC:
                break
    finally:
        print("\r" + " " * 70 + "\r", end="", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    buf = b"".join(frames)
    # trim leading silence
    samples = array.array("h")
    samples.frombytes(buf)
    i = 0
    win = SR // 50
    while i + win < len(samples):
        seg = samples[i:i + win]
        r = math.sqrt(sum(s * s for s in seg) / len(seg))
        if r >= threshold:
            break
        i += win
    start = max(0, i - SR // 20)
    trimmed = samples[start:]
    tmp = REC_DIR / "_take.wav"
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(trimmed.tobytes())
    dur = len(trimmed) / SR
    return tmp, peak


def play(path: Path):
    subprocess.run(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def mic_check():
    print("\n--- Mic check: say something normal, e.g. \"testing one two three\" ---")
    tmp, peak = record_take()
    if tmp is None:
        sys.exit("arecord failed - is a microphone connected?")
    dur = wave.open(str(tmp)).getnframes() / SR
    verdict = ("good" if peak > RMS_SIL * 3 else
               "TOO QUIET - move closer / raise mic gain" if peak > RMS_SIL else
               "BASICALLY SILENT - check mic input device!")
    print(f"    level={peak:.0f} ({verdict}), {dur:.1f}s captured")
    play(tmp)
    ok = input("    Happy with the mic? [Y/n] ").strip().lower()
    os.remove(tmp)
    if ok == "n":
        sys.exit("Fix the mic (pactl set-source-mute @DEFAULT_SOURCE@ 0 / alsamixer), then rerun.")


def run_session(sess: dict, threshold, manual_only=False):
    sid = sess["id"]
    sdir = REC_DIR / f"session_{sid}"
    sdir.mkdir(parents=True, exist_ok=True)  # exists before first os.replace
    st = load_progress(sdir)
    lines = sess["lines"]
    # session-config flag wins unless CLI explicitly forced (manual_only
    # argument passed true overrides False from config if explicitly set)
    manual = bool(manual_only) or sess.get("manual_only", False)
    todo = [i for i in range(len(lines))
            if str(i) not in st["done"] and i not in st["skipped"]]
    print(f"\n=== Session {sid}: {sess['title']} ===")
    mode_str = "MANUAL only (press ENTER to stop)" if manual else \
               "auto-stop on silence"
    print(f"    {len(todo)} of {len(lines)} lines remaining "
          f"({len(st['done'])} kept, {len(st['skipped'])} skipped) | "
          f"mode: {mode_str}")
    if not todo:
        print("Session complete! Run again without --session for the next one.")
        return
    input("Press ENTER to start, then speak after each prompt...")
    for idx in todo:
        text = lines[idx]
        print(f"\n[{idx + 1}/{len(lines)}]  >>> {text}")
        if manual:
            print("    Speak now — press ENTER when done (no auto-stop)")
        else:
            print("    Speak now — press ENTER when you finish "
                  "(or it auto-stops on silence)")
        take, peak = record_take(threshold, manual_only=manual)
        dur = wave.open(str(take)).getnframes() / SR
        if not manual and (peak < RMS_SIL or dur < 0.3):
            print("    !! too quiet or empty - retake automatically")
            take, peak = record_take(threshold)
            dur = wave.open(str(take)).getnframes() / SR
        print(f"    got {dur:.1f}s (level {peak:.0f})  "
              f"[Enter]=keep r=retry p=play s=skip q=quit")
        while True:
            c = input("    > ").strip().lower()
            if c == "":
                final = sdir / f"{idx:03d}.wav"
                os.replace(take, final)
                (sdir / f"{idx:03d}.txt").write_text(text + "\n")
                st["done"][str(idx)] = final.name
                break
            if c == "r":
                take, _ = record_take(threshold, manual_only=manual)
                dur = wave.open(str(take)).getnframes() / SR
                print(f"    new take {dur:.1f}s  [Enter]=keep r=retry p=play s=skip")
            elif c == "p":
                play(take)
            elif c == "s":
                st["skipped"].append(idx)
                os.remove(take)
                break
            elif c == "q":
                save_progress(sdir, st)
                print("Progress saved. Rerun this command to resume.")
                return
        save_progress(sdir, st)
    print(f"\nSession {sid} finished - thank you!")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--threshold", type=float, default=RMS_SIL,
                    help="silence RMS cutoff (raise if takes never stop)")
    ap.add_argument("--manual-only", action="store_true",
                    help="only stop recording when you press ENTER "
                         "(no silence auto-stop, no auto-retry on quiet). "
                         "Auto-enabled for sessions that declare manual_only "
                         "in data/prompts.json (e.g. session 8 pure Hindi).")
    args = ap.parse_args()
    data = json.loads(PROMPTS.read_text())
    if args.list:
        for s in data["sessions"]:
            done = len(load_progress(REC_DIR / f"session_{s['id']}")["done"])
            tag = " [manual-only]" if s.get("manual_only") else ""
            print(f"  session {s['id']}: {s['title']} "
                  f"[{done}/{len(s['lines'])} recorded]{tag}")
        return
    REC_DIR.mkdir(exist_ok=True)
    mic_check()
    if args.session:
        sess = next(s for s in data["sessions"] if s["id"] == args.session)
        run_session(sess, args.threshold, manual_only=args.manual_only)
    else:
        for sess in data["sessions"]:
            sdir = REC_DIR / f"session_{sess['id']}"
            st = load_progress(sdir)
            remaining = [i for i in range(len(sess["lines"]))
                         if str(i) not in st["done"] and i not in st["skipped"]]
            if remaining:
                run_session(sess, args.threshold, manual_only=args.manual_only)
                break
        else:
            print("All sessions complete!")


if __name__ == "__main__":
    main()
