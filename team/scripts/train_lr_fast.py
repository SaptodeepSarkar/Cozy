#!/usr/bin/env python3
"""Fast CPU Logistic Regression + scaler on mel windows, exported ONNX."""
from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

WW = Path(__file__).resolve().parents[2] / "wakeword"
CACHE = WW / "work" / "mels_cache.npz"
T = 63
HOP = 31

data = np.load(CACHE, allow_pickle=True)
mels = data["mels"]
names = data["names"]

POS_DIRS = [WW / "data" / "cozy", WW / "work" / "synthetic",
            WW / "work" / "synthetic_bare"]
NEG_DIRS = [WW / "data" / "similar", WW / "work" / "similar",
            WW / "work" / "negative"]

pos_files = []
for d in POS_DIRS:
    if d.exists():
        pos_files += sorted(d.rglob("*.wav"))
neg_files = []
for d in NEG_DIRS:
    if d.exists():
        neg_files += sorted(d.rglob("*.wav"))

all_files = [str(f) for f in pos_files] + [str(f) for f in neg_files]
name_to_label = {f: (1 if i < len(pos_files) else 0)
                 for i, f in enumerate(all_files)}
labels = np.array([name_to_label.get(n, 0) for n in names], dtype=np.float32)

X_list, y_list = [], []
for m, lab in zip(mels, labels):
    m = m.astype(np.float32)
    F = m.shape[0]
    if F < T:
        m = np.pad(m, ((0, T - F), (0, 0)), mode="edge")
        F = T
    for s0 in range(0, max(1, F - T + 1), HOP):
        win = m[s0:s0 + T]
        if win.shape[0] < T:
            continue
        X_list.append(win.reshape(-1))
        y_list.append(lab)

X = np.stack(X_list).astype(np.float32)
y = np.array(y_list, dtype=np.float32)
print("windows:", X.shape, "pos:", int(y.sum()), "neg:", int((y == 0).sum()))

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=200, n_jobs=-1)
lr.fit(Xs, y)
print("LR trained, score:", lr.score(Xs, y))

# ONNX
onnx_model = convert_sklearn(lr, initial_types=[("input", FloatTensorType([None, 63 * 32]))],
                              target_opset=13)
onnx_path = WW / "models" / "cozynet_lr_v1.onnx"
onnx_path.parent.mkdir(exist_ok=True)
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print("ONNX exported:", onnx_path.stat().st_size, "bytes")

joblib.dump((scaler, lr), WW / "models" / "cozynet_lr_v1.joblib")
print("DONE")
