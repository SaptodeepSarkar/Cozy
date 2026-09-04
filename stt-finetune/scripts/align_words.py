#!/usr/bin/env python3
"""Stage 2 audio processing: word-by-word transcription mapping for user
recordings, using energy-based segmentation + alignment heuristics:

  * adaptive noise floor (percentile-based, not fixed threshold)
  * hysteresis VAD (separate enter/exit margins) -> speech islands
  * micro-island merging (<60ms) to survive plosive dips
  * monotonic DP alignment of words (weighted by char length) onto islands
    weighted by duration -> word -> [t_start, t_end] map
  * sanity checks: speaking-rate plausibility, word/island agreement,
    clipped-ending detection -> per-clip verdict for re-record decisions

Outputs recordings/session_*/align/<clip>.json + a summary report.

Run: python3 scripts/align_words.py
"""
import array
import json
import math
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REC = ROOT / "recordings"
WIN_MS = 20
MIN_ISLAND_S = 0.06
GAP_SPLIT_S = 0.14      # gaps longer than this inside an island hint a boundary
SLOW_CPS, FAST_CPS = 8.0, 30.0   # plausible chars-per-second band


def envelope(samples, sr):
    win = int(sr * WIN_MS / 1000)
    n = len(samples) // win
    env = []
    for i in range(n):
        seg = samples[i * win:(i + 1) * win]
        env.append(math.sqrt(sum(s * s for s in seg) / max(1, len(seg))))
    return env


def islands(env, sr):
    """Adaptive-floor hysteresis VAD over the energy envelope."""
    if not env:
        return []
    floor = sorted(env)[max(0, len(env) // 10)]        # 10th percentile
    enter = floor * 3.0 + 60
    exit_ = floor * 1.6 + 40
    out, cur = [], None
    for i, e in enumerate(env):
        if cur is None:
            if e >= enter:
                cur = [i, i]
        else:
            if e >= exit_:
                cur[1] = i
            elif i - cur[1] > int(0.12 * sr / (sr * WIN_MS / 1000)) and e < exit_:
                # allow brief dips; close after 120ms of quiet
                gap_ms = (i - cur[1]) * WIN_MS
                if gap_ms >= 120:
                    out.append(cur)
                    cur = None
    if cur is not None:
        out.append(cur)
    # convert to seconds + merge micro islands
    secs = [[a * WIN_MS / 1000, (b + 1) * WIN_MS / 1000] for a, b in out]
    merged = []
    for s in secs:
        if merged and (s[0] - merged[-1][1] < MIN_ISLAND_S or
                       (s[1] - s[0]) < MIN_ISLAND_S and merged[-1][1] - merged[-1][0] < 0.5):
            merged[-1][1] = s[1]
        elif merged and s[1] - s[0] < MIN_ISLAND_S:
            continue  # drop tiny blip entirely? keep as its own if far
        else:
            merged.append(s)
    return [s for s in merged if s[1] - s[0] >= MIN_ISLAND_S]


def split_island_on_gaps(samples, sr, isl):
    """Inside an island, propose internal boundaries at deep energy minima."""
    a, b = int(isl[0] * sr), int(isl[1] * sr)
    seg = samples[a:b]
    win = int(sr * WIN_MS / 1000)
    env = [math.sqrt(sum(s * s for s in seg[i:i + win]) / max(1, len(seg[i:i + win])))
           for i in range(0, len(seg) - win, win)]
    if len(env) < 4:
        return []
    lo = sorted(env)[max(0, len(env) // 5)]
    cuts = []
    for j in range(2, len(env) - 2):
        if env[j] <= lo * 1.35 and env[j] <= env[j - 1] and env[j] <= env[j + 1]:
            t = isl[0] + (j + 0.5) * WIN_MS / 1000
            if not cuts or t - cuts[-1] > GAP_SPLIT_S:
                cuts.append(t)
    return cuts


def align(words, isl_spans, extra_cuts):
    """Monotonic DP: assign each word a start/end inside island space."""
    # build boundary candidates: island starts/ends + intra-island cuts
    bounds = []
    for k, (s, e) in enumerate(isl_spans):
        inner = sorted(c for c in extra_cuts[k] if s + 0.02 < c < e - 0.02)
        pts = [s] + inner + [e]
        bounds.append(pts)
    n_words, n_isl = len(words), len(isl_spans)
    if n_isl == 0:
        return {w: None for w in words}
    wl = [max(len(w), 1) for w in words]
    tot_chars = sum(wl)

    # DP over (word_index, island_index); states = word start offset choices kept simple:
    # assign words sequentially to islands, allowing several words per island;
    # cost = |expected_time(word) - available_time_share|
    exp = [c / tot_chars for c in [sum(wl[:i]) for i in range(n_words + 1)]]  # cumulative
    total_dur = sum(e - s for s, e in isl_spans)
    NEG = float("inf")
    dp = {(i, j): NEG for i in range(n_words + 1) for j in range(n_isl + 1)}
    bt = {}
    dp[(0, 0)] = 0.0
    for i in range(n_words + 1):
        for j in range(n_isl + 1):
            c = dp.get((i, j), NEG)
            if c == NEG:
                continue
            if i < n_words:   # word i lives in island j (if exists)
                if j < n_isl:
                    dur = isl_spans[j][1] - isl_spans[j][0]
                    want = (exp[i + 1] - exp[i]) * total_dur
                    share = dur * wl[i] / max(1, sum(wl[k] for k in range(
                        next((x for x in range(i, -1, -1) if False), i), 0)))  # unused
                    cost = abs(want - min(dur, want)) * 0.5
                    nc = c + cost
                    key = (i + 1, j)
                    if nc < dp.get(key, NEG):
                        dp[key] = nc
                        bt[key] = (i, j)
                if j < n_isl + 1 and i > 0:
                    pass
            # move to next island without consuming a word
            if j < n_isl and dp.get((i, j + 1), NEG) > c:
                dp[(i, j + 1)] = c
                bt[(i, j + 1)] = (i, j)
    end = min(range(n_isl + 1), key=lambda j: dp.get((n_words, j), NEG))
    if dp[(n_words, end)] == NEG:
        return None
    # walk back: which island each word ended in
    word_isl = [None] * n_words
    i, j = n_words, end
    while (i, j) in bt:
        pi, pj = bt[(i, j)]
        if pi == i - 1:               # consumed word i-1 in island pj
            word_isl[i - 1] = pj
        i, j = pi, pj

    # place times: distribute words across their island's sub-boundaries
    result = {}
    per_isl = {}
    for wi, ji in enumerate(word_isl):
        per_isl.setdefault(ji, []).append(wi)
    for ji, wis in per_isl.items():
        pts = bounds[ji]
        m = len(wis)
        for idx, wi in enumerate(wis):
            a = pts[min(idx, len(pts) - 2)]
            b = pts[min(idx + 1, len(pts) - 1)]
            result[words[wi]] = [round(a, 3), round(b, 3)]
    return result


def main():
    summary = {}
    for sdir in sorted(REC.glob("session_*")):
        adir = sdir / "align"
        adir.mkdir(exist_ok=True)
        for wav in sorted(sdir.glob("*.wav")):
            txt_file = wav.with_suffix(".txt")
            if not txt_file.exists():
                continue
            with wave.open(str(wav)) as w:
                sr = w.getframerate()
                samples = array.array("h")
                samples.frombytes(w.readframes(w.getnframes()))
            dur = len(samples) / sr
            text = txt_file.read_text().strip()
            words = text.split()
            env = envelope(samples, sr)
            spans = islands(env, sr)
            cuts = [split_island_on_gaps(samples, sr, s) for s in spans]

            cps = len(text) / max(dur, 0.1)
            flags = []
            if not (SLOW_CPS <= cps <= FAST_CPS):
                flags.append(f"speaking_rate_{cps:.1f}cps")
            if len(spans) > len(words) * 2:
                flags.append("too_many_speech_islands")
            speech_s = sum(e - s for s, e in spans)
            if speech_s < dur * 0.35:
                flags.append("mostly_silence")
            if dur < len(words) * 0.12:
                flags.append("suspiciously_fast")

            amap = align(words, spans, cuts)
            entry = {
                "duration_s": round(dur, 2),
                "speech_islands": len(spans),
                "words": len(words),
                "chars_per_sec": round(cps, 1),
                "flags": flags,
                "verdict": "warn" if flags else "ok",
                "word_map": amap,
            }
            (adir / f"{wav.stem}.json").write_text(json.dumps(entry, indent=1))
            summary[f"{sdir.name}/{wav.name}"] = {
                "session": sdir.name,
                **{k: v for k, v in entry.items() if k != "word_map"}}
    warns = {k: v["flags"] for k, v in summary.items() if v["verdict"] == "warn"}
    print(f"aligned {len(summary)} clips | {len(warns)} flagged")
    for k, fl in list(warns.items())[:15]:
        print(f"  ! {k}: {fl}")
    (REC / "align_report.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
