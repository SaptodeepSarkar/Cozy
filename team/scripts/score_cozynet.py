
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

WW = Path(__file__).resolve().parents[2] / "wakeword"
meta = json.loads((WW / "models" / "cozynet_v1_meta.json").read_text())
T = meta["time_frames"]
BINS = meta["mel_bins"]
MEAN = np.array(meta["mel_mean"], dtype=np.float32)
STD = np.array(meta["mel_std"], dtype=np.float32)
THRESH = float(sys.argv[1]) if len(sys.argv) > 1 else \
    float(meta.get("safe_threshold_zero_fpr") or 0.5)


class CozyNet(nn.Module):
    def __init__(self):
        super().__init__()
        l1 = T // 4
        l2 = l1 + 1
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(), nn.MaxPool2d((4, 1)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.ZeroPad2d((0, 0, 1, 0)),
            nn.MaxPool2d((l2, 1)),
            nn.Flatten(),
            nn.Linear(32 * 32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


model = CozyNet()
model.load_state_dict(torch.load(WW / "models" / "cozynet_v1.pt",
                                 map_location="cpu"))
# Keep BN in train mode (uses batch stats) - running stats were corrupted
model.train()

from openwakeword.model import AudioFeatures  # noqa: E402

af = AudioFeatures()


def mel_of(path):
    pcm, sr = sf.read(str(path), dtype="int16")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    # Pad to 32000 samples EXACTLY like training cache
    x = np.zeros(32000, dtype=np.int16)
    n = min(len(pcm), 32000)
    x[:n] = pcm[:n]
    out = af._get_melspectrogram_batch(np.stack([x]), batch_size=8, ncpu=4)
    m = np.asarray(out)[0].astype(np.float32)
    # Trim trailing zero frames like training
    e = np.abs(m).sum(axis=1)
    nz = np.flatnonzero(e > 1e-6)
    if nz.size > 0:
        m = m[nz[0]:nz[-1]+1]
    # Edge-pad to T if shorter (like training)
    if m.shape[0] < T:
        m = np.pad(m, ((0, T - m.shape[0]), (0, 0)), mode="edge")
    return m[:T]


CLIP_SAMPLES_LOCAL = 32000


def score(path):
    m = mel_of(path)
    if m.shape[0] < T:
        m = np.pad(m, ((0, T - m.shape[0]), (0, 0)), mode="edge")
    m = (m - MEAN) / STD
    best = 0.0
    with torch.no_grad():
        for s0 in range(0, max(1, m.shape[0] - T + 1), 31):
            win = m[s0:s0 + T][None, None]
            xb = torch.from_numpy(win).float()
            s = float(torch.sigmoid(model(xb)).reshape(-1)[0])
            best = max(best, s)
    return best


cases = []
for p in sorted((WW / "data/cozy").glob("recording_*.wav"))[:6]:
    cases.append(("POS you " + p.name, p, True))
for p in sorted((WW / "data/cozy").glob("bare_*.wav"))[:4]:
    cases.append(("POS BARE you " + p.name, p, True))
for p in sorted((WW / "data/cozy").glob("synth_*.wav"))[:2]:
    cases.append(("POS synth", p, True))
for w_ in ["hey_rosy", "hey_nosy", "hey_josie", "hey_noisy"]:
    p = WW / "data/similar" / ("recording_" + w_ + "_001.wav")
    if p.exists():
        cases.append(("NEG you " + w_, p, False))
for w_ in ["rosy", "nosy", "noisy"]:
    p = WW / "data/similar" / ("bare_recording_" + w_ + "_001.wav")
    if p.exists():
        cases.append(("NEG BARE you " + w_, p, False))

good = ran = 0
print("threshold:", THRESH)
for label, p, should in cases:
    s = score(p)
    fired = s >= THRESH
    ok = fired == should
    good += ok
    ran += 1
    print(("PASS" if ok else "FAIL"), label.ljust(24),
          "score=" + format(s, ".3f"), "(fired)" if fired else "(silent)")

negs = sorted((WW / "work/negative").glob("*.wav"))[:80]
worst = max(score(p) for p in negs)
fires = sum(1 for p in negs if score(p) >= THRESH)
print("80 random negatives: worst =", format(worst, ".3f"),
      "| fires at threshold:", fires)
print("RESULT:", str(good) + "/" + str(ran))
