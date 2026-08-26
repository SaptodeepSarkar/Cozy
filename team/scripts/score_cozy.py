
import sys
from pathlib import Path

import soundfile as sf
from openwakeword.model import Model

WW = Path(__file__).resolve().parents[2] / "wakeword"
m = Model(wakeword_models=[str(WW / "models" / "cozy_v1.onnx")],
          inference_framework="onnx")
name = next(iter(m.models.keys()))
CHUNK = 1280
THRESH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.40


def score(path):
    pcm, sr = sf.read(str(path), dtype="int16")
    m.reset()
    best = 0.0
    for i in range(0, len(pcm) - CHUNK + 1, CHUNK):
        out = m.predict(pcm[i:i + CHUNK])
        best = max(best, float(out[name]))
    return best


import sys  # noqa: E402

def first(pattern, folder="data"):
    hits = sorted((WW / folder).glob(pattern))
    return hits[0] if hits else None


cases = [
    ("POS hey you r010+", first("recording_01*.wav", "data/cozy"), True),
    ("POS hey you r015", WW / "data/cozy/recording_015.wav", True),
    ("POS BARE you b001", first("bare_001.wav", "data/cozy"), True),
    ("POS BARE you b003", first("bare_003.wav", "data/cozy"), True),
    ("POS synth hey", first("synth_*.wav", "data/cozy"), True),
    ("NEG BARE rosy", first("bare_recording_rosy*.wav", "data/similar"), False),
    ("NEG BARE nosy", first("bare_recording_nosy*.wav", "data/similar"), False),
    ("NEG hey rosy", first("recording_hey_rosy_001.wav", "data/similar"), False),
    ("NEG hey josie", first("recording_hey_josie_001.wav", "data/similar"), False),
]


def first(pattern, folder="data"):
    hits = sorted((WW / folder).glob(pattern))
    return hits[0] if hits else None


def first(pattern, folder="data"):
    hits = sorted((WW / folder).glob(pattern))
    return hits[0] if hits else None


good = ran = 0
for label, path, should in cases:
    if path is None or not Path(path).exists():
        print("SKIP", label)
        continue
    ran += 1
    s = score(path)
    fired = s >= THRESH
    ok = fired == should
    good += ok
    print(("PASS" if ok else "FAIL"), label.ljust(20),
          "score=" + format(s, ".3f"), "(fired)" if fired else "(silent)")

negs = sorted((WW / "work/negative").glob("*.wav"))[:80]
worst = 0.0
for p in negs:
    worst = max(worst, score(p))
print("80 random negatives worst:", format(worst, ".3f"))
print("RESULT:", str(good) + "/" + str(ran), "at threshold", THRESH)
