"""CPU-only clean retrain - zero VRAM risk, ~5 min for tiny model."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

WW = Path("/home/saptodeepsarkar/Projects/Cozy/wakeword")
T = 63
HOP = 31

cache = np.load(WW / "work" / "mels_cache.npz", allow_pickle=True)
mels = cache["mels"]
names = list(cache["names"])

POS_DIRS = [WW / "data" / "cozy", WW / "work" / "synthetic",
            WW / "work" / "synthetic_bare"]
NEG_DIRS = [WW / "data" / "similar", WW / "work" / "similar",
            WW / "work" / "negative"]
pos_files = sum([sorted(d.rglob("*.wav")) for d in POS_DIRS if d.exists()], [])
neg_files = sum([sorted(d.rglob("*.wav")) for d in NEG_DIRS if d.exists()], [])
all_files = [str(f) for f in pos_files] + [str(f) for f in neg_files]
name_to_label = {f: (1 if i < len(pos_files) else 0)
                 for i, f in enumerate(all_files)}
labels = np.array([name_to_label.get(n, 0) for n in names], dtype=np.float32)


def trim_pad(m):
    e = np.abs(m).sum(axis=1)
    nz = np.flatnonzero(e > 1e-6)
    if len(nz) == 0:
        return m
    return m[nz[0]:nz[-1] + 1]


mels_clean = [trim_pad(m).astype(np.float32) for m in mels]

X_list, y_list, src_list = [], [], []
for m, lab, src in zip(mels_clean, labels, names):
    F = m.shape[0]
    if F < T:
        continue
    for s0 in range(0, F - T + 1, HOP):
        X_list.append(m[s0:s0 + T])
        y_list.append(lab)
        src_list.append(src)

X = np.stack(X_list).astype(np.float32)
y = np.array(y_list, dtype=np.float32)
src_all = np.array(src_list)

MEAN = X.mean(axis=0)
STD = X.std(axis=0) + 1e-6
X_norm = ((X - MEAN) / STD)[:, None, :, :]

user_mask = np.array([s.startswith(str(WW / "data")) for s in src_all])
user_idx = np.flatnonzero(user_mask)
if len(user_idx) > 0:
    X_norm = np.concatenate([X_norm, X_norm[user_idx]], axis=0)
    y = np.concatenate([y, y[user_idx]], axis=0)
print("windows:", X_norm.shape, "pos:", int(y.sum()),
      "neg:", int((y == 0).sum()), "user_dupes:", len(user_idx))

test_idx = np.arange(0, len(y), 5)
train_idx = np.setdiff1d(np.arange(len(y)), test_idx)


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


device = torch.device("cpu")
model = CozyNet().to(device)
opt = optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

best_auc, best_state, patience = -1.0, None, 0
bs = 64


def evaluate(Xva, yva):
    model.eval()
    probs = np.zeros(len(yva), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(yva), bs):
            xb = torch.from_numpy(Xva[i:i + bs]).float()
            p = torch.sigmoid(model(xb)).cpu().numpy()
            probs[i:i + bs] = p
    pos_m = yva == 1
    if pos_m.sum() == 0 or len(yva) == pos_m.sum():
        return 0.0
    order = np.argsort(probs)
    ranks = np.empty(len(probs))
    ranks[order] = np.arange(1, len(probs) + 1)
    auc = float((ranks[pos_m].sum() - pos_m.sum() * (pos_m.sum() + 1) / 2)
                / (pos_m.sum() * (len(probs) - pos_m.sum())))
    return auc


Xtr_np = X_norm[train_idx]
ytr_np = y[train_idx]
Xva_np = X_norm[test_idx]
yva_np = y[test_idx]

for epoch in range(1, 21):
    model.train()
    perm = np.random.permutation(len(Xtr_np))
    tot = 0.0
    for i in range(0, len(perm), bs):
        idx = perm[i:i + bs]
        xb = torch.from_numpy(Xtr_np[idx]).float()
        yb = torch.from_numpy(ytr_np[idx]).float()
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        tot += float(loss) * len(idx)
    auc = evaluate(Xva_np, yva_np)
    print("epoch", epoch, "loss=" + format(tot / len(perm), ".4f"),
          "val_auc=" + format(auc, ".4f"), flush=True)
    if auc > best_auc:
        best_auc, patience = auc, 0
        best_state = {k: v.cpu().clone()
                      for k, v in model.state_dict().items()}
    else:
        patience += 1
        if patience >= 5:
            print("early stop")
            break

if best_state:
    model.load_state_dict(best_state)
auc = evaluate(Xva_np, yva_np)
print("FINAL val_auc=" + format(auc, ".4f"))

model.eval()
torch.save(model.state_dict(), WW / "models" / "cozynet_v1.pt")
dummy = torch.zeros(1, 1, T, 32)
torch.onnx.export(model, dummy, str(WW / "models" / "cozynet_v1.onnx"),
                  input_names=["input"], output_names=["scores"],
                  opset_version=13, dynamo=False)
meta = {
    "type": "cozy_v1", "time_frames": T, "mel_bins": 32,
    "mel_mean": MEAN.tolist(),
    "mel_std": STD.tolist(),
    "val_auc": round(auc, 4),
    "safe_threshold_zero_fpr": None,
    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
(WW / "models" / "cozynet_v1_meta.json").write_text(json.dumps(meta, indent=2))
print("DONE")
