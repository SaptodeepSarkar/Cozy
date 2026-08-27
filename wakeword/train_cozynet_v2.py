#!/usr/bin/env python3
"""CozyNet v2 - clean wake word trainer (v3, simplified)."""
from __future__ import annotations
import json, time, random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

random.seed(0); np.random.seed(0); torch.manual_seed(0)

WW = Path("/home/saptodeepsarkar/Projects/Cozy/wakeword")
T = 97       # exactly 1.0s
BINS = 32
WIN_SAMPLES = 16000

from openwakeword.model import AudioFeatures
_af = AudioFeatures()


def mel_of(pcm_int16):
    if pcm_int16.ndim > 1:
        pcm_int16 = pcm_int16.mean(axis=1)
    pcm_int16 = pcm_int16.astype(np.int16)
    out = _af._get_melspectrogram_batch(pcm_int16[None, :], batch_size=8, ncpu=4)
    return np.asarray(out)[0].astype(np.float32)


def windows_from_file(path, label, max_windows=4):
    pcm, sr = sf.read(str(path), dtype="int16")
    if sr != 16000:
        from scipy.signal import resample_poly
        pcm = resample_poly(pcm.astype(np.int16), 16000, sr).astype(np.int16)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1).astype(np.int16)
    if len(pcm) < WIN_SAMPLES:
        pcm = np.pad(pcm, (0, WIN_SAMPLES - len(pcm)))
    m = mel_of(pcm)
    out = []
    if m.shape[0] <= T:
        seg = np.pad(m, ((0, max(0, T - m.shape[0])), (0, 0)))
        out.append((seg, label, str(path)))
    else:
        step = max(1, (m.shape[0] - T) // max(1, max_windows - 1))
        for s0 in range(0, m.shape[0] - T + 1, step):
            seg = m[s0:s0 + T]
            out.append((seg, label, str(path)))
            if len(out) >= max_windows:
                break
    return out


def build_dataset(pos_files, neg_files, max_neg_windows=2):
    X, y, src = [], [], []
    for p in pos_files:
        for seg, lab, s in windows_from_file(p, 1, max_windows=2):
            X.append(seg); y.append(lab); src.append(s)
    pos_count = len(X)
    for p in neg_files:
        for seg, lab, s in windows_from_file(p, 0, max_windows=max_neg_windows):
            X.append(seg); y.append(lab); src.append(s)
    return (np.stack(X).astype(np.float32),
            np.array(y, dtype=np.float32),
            np.array(src))


class CozyNetV2(nn.Module):
    """Lightweight 1D-CNN over mel spectrogram. No BatchNorm in eval (uses GroupNorm)."""
    def __init__(self):
        super().__init__()
        # 1D conv along time, treating each mel bin independently
        self.conv1 = nn.Conv1d(BINS, 32, kernel_size=5, padding=2)
        self.gn1 = nn.GroupNorm(8, 32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.gn2 = nn.GroupNorm(8, 64)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(8, 64)
        # After 2 pools: 97 -> 48 -> 24
        self.fc1 = nn.Linear(64 * 24, 64)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B, 1, T, 32) -> (B, 32, T)
        x = x.squeeze(1).transpose(1, 2)
        x = self.pool1(torch.relu(self.gn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.gn2(self.conv2(x))))
        x = torch.relu(self.gn3(self.conv3(x)))
        x = x.flatten(1)
        x = self.drop(torch.relu(self.fc1(x)))
        return self.fc2(x).squeeze(-1)


def main():
    device = torch.device("cpu")
    print("device:", device)

    user_pos = sorted((WW / "work" / "user_pos").glob("*.wav"))
    synth_pos = sorted((WW / "work" / "synthetic").glob("*.wav"))[:1500]
    synth_bare = sorted((WW / "work" / "synthetic_bare").glob("*.wav"))[:500]
    user_neg = sorted((WW / "work" / "user_neg").glob("*.wav"))
    synth_neg = sorted((WW / "work" / "negative").glob("*.wav"))[:1500]
    similar_files = []
    for d in (WW / "work" / "similar").iterdir():
        if d.is_dir():
            similar_files += sorted(d.glob("*.wav"))[:200]
    random.shuffle(similar_files); similar_files = similar_files[:1000]

    pos_files = user_pos + synth_pos + synth_bare
    neg_files = user_neg + synth_neg + similar_files
    random.shuffle(neg_files); neg_files = neg_files[:2000]
    print(f"train pos: {len(pos_files)}  train neg: {len(neg_files)}")

    Xtr, ytr, _ = build_dataset(pos_files, neg_files, max_neg_windows=2)
    print(f"  positive windows: {(ytr == 1).sum()}")
    print(f"  negative windows: {(ytr == 0).sum()}")

    MEAN = Xtr.mean(axis=0)
    STD = Xtr.std(axis=0) + 1e-6
    Xtr_n = ((Xtr - MEAN) / STD)[:, None, :, :]

    val_pos_files = sorted((WW / "work" / "val_pos").glob("*.wav"))
    val_neg_files = sorted((WW / "work" / "val_neg").glob("*.wav"))
    print(f"val pos: {len(val_pos_files)}  val neg: {len(val_neg_files)}")
    Xva, yva, _ = build_dataset(val_pos_files, val_neg_files, max_neg_windows=4)
    Xva_n = ((Xva - MEAN) / STD)[:, None, :, :]

    Xtr_t = torch.from_numpy(Xtr_n).float()
    ytr_t = torch.from_numpy(ytr).float()
    Xva_t = torch.from_numpy(Xva_n).float()
    print(f"shapes: train={tuple(Xtr_t.shape)} val={tuple(Xva_t.shape)}")

    n_pos = (ytr == 1).sum(); n_neg = (ytr == 0).sum()
    pos_weight = torch.tensor([min(3.0, max(1.5, n_neg / n_pos))])
    print(f"pos_weight={float(pos_weight):.2f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = CozyNetV2().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)

    bs = 64
    best_state, best_loss = None, float("inf")
    history = []

    def evaluate():
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(Xva_t)).numpy()
        yv = yva
        pos_m = yv == 1; npos = int(pos_m.sum()); nneg = int(len(yv) - npos)
        if npos == 0 or nneg == 0:
            return 0.0, probs, 0.5, 0.0, 1.0
        order = np.argsort(probs)
        ranks = np.empty(len(probs)); ranks[order] = np.arange(1, len(probs) + 1)
        auc = float((ranks[pos_m].sum() - npos * (npos + 1) / 2) / (npos * nneg))
        best_t = 0.5; best_score = -1.0
        for t in np.arange(0.20, 0.95, 0.005):
            fpr = float((probs[yv == 0] >= t).mean())
            tpr = float((probs[yv == 1] >= t).mean())
            score = tpr - 3.0 * fpr
            if score > best_score:
                best_score = score
                best_t = float(t)
        fpr_05 = float((probs[yv == 0] >= 0.5).mean())
        tpr_05 = float((probs[yv == 1] >= 0.5).mean())
        return auc, probs, best_t, tpr_05, fpr_05

    for epoch in range(1, 81):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = Xtr_t[idx]; yb = ytr_t[idx]
            B, _, T_, F_ = xb.shape
            for _ in range(2):
                w = int(torch.randint(8, 20, (1,)))
                t0 = int(torch.randint(0, max(1, T_ - w), (1,)))
                xb[:, :, t0:t0 + w, :] = 0
            for _ in range(2):
                w = int(torch.randint(2, 5, (1,)))
                f0 = int(torch.randint(0, max(1, F_ - w), (1,)))
                xb[:, :, :, f0:f0 + w] = 0
            # gain perturbation (volume augmentation)
            gain = float(np.random.uniform(0.7, 1.3))
            xb = xb * gain
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step()
        auc, probs, thr, tpr, fpr = evaluate()
        line = (f"epoch {epoch:02d} loss={tot/len(perm):.4f} "
                f"val_auc={auc:.4f} tpr@0.5={tpr:.3f} fpr@0.5={fpr:.3f} "
                f"best_t={thr:.3f}")
        history.append(line)
        if epoch % 5 == 0 or epoch == 1:
            print(line, flush=True)
        if tot < best_loss:
            best_loss = tot
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    auc, probs, thr, tpr, fpr = evaluate()
    print(f"\nFINAL val_auc={auc:.4f} thr={thr:.3f} tpr@0.5={tpr:.3f} fpr@0.5={fpr:.3f}")
    print(f"Positive scores: {sorted(probs[yva == 1])}")
    print(f"Top 10 neg scores: {sorted(probs[yva == 0])[-10:]}")

    model.eval()
    (WW / "models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), WW / "models" / "cozynet_v2.pt")
    meta = {
        "type": "cozynet_v2",
        "time_frames": T,
        "mel_bins": BINS,
        "mel_mean": MEAN.tolist(),
        "mel_std": STD.tolist(),
        "val_auc_real_voice": round(float(auc), 4),
        "suggested_threshold": float(thr),
        "tpr_at_threshold": float(tpr),
        "fpr_at_threshold": float(fpr),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (WW / "models" / "cozynet_v2_meta.json").write_text(json.dumps(meta, indent=2))

    dummy = torch.zeros(1, 1, T, BINS)
    onnx_path = WW / "models" / "cozynet_v2.onnx"
    torch.onnx.export(model, dummy, str(onnx_path),
                      input_names=["input"], output_names=["scores"],
                      opset_version=13, dynamo=False)
    print(f"exported {onnx_path}")
    print("COZYNET_V2_DONE")


if __name__ == "__main__":
    main()
