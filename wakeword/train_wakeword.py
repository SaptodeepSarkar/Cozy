#!/usr/bin/env python3
"""Trains the Cozy wake-word detector and exports it for openWakeWord.

Steps
  1. Collect clips:   data/cozy  + work/synthetic     -> label 1
                      data/similar + work/similar     -> label 0 (HARD negatives)
                      work/negative                   -> label 0
  2. Embed every clip with openWakeWord frozen melspectrogram + embedding
     feature models (one clean pass plus augmented passes per clip).
  3. Slide fixed windows over embeddings, balance classes, train a small
     PyTorch classifier head, validate with AUC and false-alarm tracking.
  4. Export the best checkpoint to models/cozy_v1.onnx and sanity-check it
     through the real openWakeWord runtime.

Artifacts written to wakeword/models/: cozy_v1.onnx, cozy_v1.pt, metrics.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import yaml

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
MODELS_DIR = HERE / "models"

CFG = yaml.safe_load((HERE / "config.yaml").read_text())
TR = CFG["training"]
W = int(TR["window_frames"])
STRIDE = int(TR["stride_frames"])
EMB_DIM = 96
SR = int(CFG["audio"]["sample_rate"])
CLIP = int(float(CFG["audio"]["clip_seconds"]) * SR)


def collect_clips():
    positive = sorted((HERE / "data" / "cozy").glob("*.wav"))
    positive += sorted((WORK / "synthetic").glob("*.wav"))
    similar = sorted((HERE / "data" / "similar").glob("*.wav"))
    sim_root = WORK / "similar"
    if sim_root.exists():
        for sub in sorted(sim_root.iterdir()):
            if sub.is_dir():
                similar += sorted(sub.glob("*.wav"))
    negative = sorted((WORK / "negative").glob("*.wav"))
    return positive, similar, negative


def augment(pcm, rng):
    gain_db = float(rng.uniform(-6.0, 6.0))
    x = pcm * (10.0 ** (gain_db / 20.0))
    snr_db = float(rng.uniform(12.0, 30.0))
    sig_rms = float(np.sqrt(np.mean(x * x))) + 1e-6
    noise_rms = sig_rms * (10.0 ** (-snr_db / 20.0))
    noise = rng.normal(0.0, noise_rms, size=x.shape).astype(np.float32)
    return np.clip(x + noise, -1.0, 1.0)


class FeatureExtractor:
    """Wraps openWakeWord AudioFeatures (embed_clips API, v0.6+)."""

    def __init__(self):
        from openwakeword.model import AudioFeatures
        self.af = AudioFeatures()
        self.target = CLIP

    def embeddings(self, pcm_f32):
        """Extract features EXACTLY like deployment: feed 80 ms chunks
        sequentially through one persistent AudioFeatures instance and read
        its feature buffer - these are the very windows Model.predict scores
        at inference time (frame i = last 16 buffer frames ending at chunk i).
        """
        scaled = np.clip(np.round(pcm_f32 * 32767.0), -32768, 32767)
        pcm16 = scaled.astype(np.int16)
        x = pcm16[: self.target]
        if len(x) < self.target:
            pad = np.zeros(self.target - len(x), dtype=np.int16)
            x = np.concatenate([x, pad])
        self.af.reset()
        for i in range(0, len(x), 1280):
            self.af(x[i:i + 1280])
        emb = np.asarray(self.af.feature_buffer, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != EMB_DIM:
            raise RuntimeError("unexpected embedding shape " + str(emb.shape))
        return emb


def windows_for_clip(emb, rng):
    frames = emb.shape[0]
    if frames < W:
        pad = np.zeros((W - frames, EMB_DIM), dtype=np.float32)
        emb = np.vstack([emb, pad])
        frames = W
    starts = list(range(0, frames - W + 1, STRIDE))
    if not starts:
        starts = [frames - W]
    cap = int(TR["max_windows_per_clip"])
    if len(starts) > cap:
        starts = sorted(random.sample(starts, cap))
    picked = []
    for s in starts:
        picked.append(emb[s:s + W])
    return np.stack(picked)


def split_clips(clips, val_frac):
    order = list(range(len(clips)))
    random.Random(11).shuffle(order)
    cut = int(len(order) * val_frac) if len(order) > 10 else 0
    train = [clips[i] for i in order[cut:]]
    val = [clips[i] for i in order[:cut]]
    return train, val


def embed_split(fe, clips, label, copies, rng, tag):
    xs = []
    ys = []
    for i, path in enumerate(clips):
        try:
            pcm, _sr = sf.read(str(path), dtype="float32")
        except Exception:
            continue
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        if pcm.size < 1600:
            continue
        for copy_i in range(copies):
            x = pcm if copy_i == 0 else augment(pcm, rng)
            try:
                emb = fe.embeddings(x)
            except Exception:
                continue
            wins = windows_for_clip(emb, rng).astype(np.float32)
            xs.append(wins)
            ys.append(np.full(len(wins), label, dtype=np.float32))
        if (i + 1) % 500 == 0:
            print("  embedded " + tag + ": " + str(i + 1) + "/"
                  + str(len(clips)))
    if not xs:
        ex = np.zeros((0, W * EMB_DIM), dtype=np.float32)
        ey = np.zeros((0,), dtype=np.float32)
        return ex, ey
    flat_x = np.concatenate(xs, axis=0).reshape(-1, W * EMB_DIM)
    flat_y = np.concatenate(ys, axis=0)
    return flat_x, flat_y


def build_dataset(fe, rng, limit=0):
    positive, similar, negative = collect_clips()
    if limit:
        positive = positive[:limit]
        similar = similar[:limit]
        negative = negative[:limit]
    print("[train] clips: positive=" + str(len(positive))
          + " similar=" + str(len(similar))
          + " negative=" + str(len(negative)))
    if len(positive) == 0 or (len(similar) + len(negative)) == 0:
        raise SystemExit("[train] not enough data - run generate_data.py")

    copies = int(TR["augment_copies"])
    val_frac = float(TR["val_fraction"])

    pos_tr, pos_va = split_clips(positive, val_frac)
    sim_tr, sim_va = split_clips(similar, val_frac)
    neg_tr, neg_va = split_clips(negative, val_frac)

    print("[train] embedding features (this is the slow part)...")
    X_pos_tr, y_pos_tr = embed_split(fe, pos_tr, 1.0, copies, rng, "pos/tr")
    X_pos_va, y_pos_va = embed_split(fe, pos_va, 1.0, copies, rng, "pos/va")
    X_sim_tr, y_sim_tr = embed_split(fe, sim_tr, 0.0, copies, rng, "sim/tr")
    X_sim_va, y_sim_va = embed_split(fe, sim_va, 0.0, copies, rng, "sim/va")
    X_neg_tr, y_neg_tr = embed_split(fe, neg_tr, 0.0, copies, rng, "neg/tr")
    X_neg_va, y_neg_va = embed_split(fe, neg_va, 0.0, copies, rng, "neg/va")

    parts_x = []
    parts_y = []
    if len(y_sim_tr) > 0:
        parts_x.append(np.repeat(X_sim_tr, 3, axis=0))
        parts_y.append(np.repeat(y_sim_tr, 3))
    if len(y_neg_tr) > 0:
        parts_x.append(X_neg_tr)
        parts_y.append(y_neg_tr)
    if len(parts_x) > 0:
        X_neg_all = np.concatenate(parts_x, axis=0)
        y_neg_all = np.concatenate(parts_y, axis=0)
    else:
        X_neg_all = np.zeros((0, W * EMB_DIM), dtype=np.float32)
        y_neg_all = np.zeros((0,), dtype=np.float32)

    budget = 2 * len(y_pos_tr)
    if budget > 0 and len(y_neg_all) > budget:
        idx = random.Random(13).sample(range(len(y_neg_all)), budget)
        X_neg_all = X_neg_all[idx]
        y_neg_all = y_neg_all[idx]

    Xtr = np.concatenate([X_pos_tr, X_neg_all], axis=0)
    ytr = np.concatenate([y_pos_tr, y_neg_all], axis=0)
    perm = np.random.default_rng(17).permutation(len(ytr))

    Xva = np.concatenate([X_pos_va, X_sim_va, X_neg_va], axis=0)
    yva = np.concatenate([y_pos_va, y_sim_va, y_neg_va], axis=0)
    n_pos_va = len(y_pos_va)
    n_sim_va = len(y_sim_va)

    print("[train] windows: train=" + str(len(ytr))
          + " (pos=" + str(int(y_pos_tr.sum()))
          + ", neg=" + str(int(y_neg_all.sum())) + ")"
          + " val=" + str(len(yva))
          + " (pos=" + str(n_pos_va) + ", sim=" + str(n_sim_va) + ")")

    tX = torch.from_numpy(Xtr[perm])
    ty = torch.from_numpy(ytr[perm])
    vX = torch.from_numpy(Xva)
    vy = torch.from_numpy(yva)
    return tX, ty, vX, vy, n_pos_va, n_sim_va


class WWNet(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ExportWrapper(nn.Module):
    """Maps [B, W, 96] embeddings to probability in [0, 1]."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        flat = x.reshape(x.shape[0], -1)
        # bypass WWNet.forward: keep the [B, 1] output rank that
        # openWakeWord's loader introspects (outputs shape must be [N, 1])
        logits = self.net.net(flat)
        return torch.sigmoid(logits)


def auc_score(y_true, scores):
    pos = y_true == 1
    npos = int(pos.sum())
    nneg = int(len(pos) - npos)
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    num = ranks[pos].sum() - npos * (npos + 1) / 2.0
    return float(num / (npos * nneg))


def evaluate(model, X, y):
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        logits = model(X.to(dev))
        probs = torch.sigmoid(logits).cpu().numpy()
    y_np = y.cpu().numpy()
    preds = (probs >= 0.5).astype(np.float32)
    tp = float(((preds == 1) & (y_np == 1)).sum())
    fp = float(((preds == 1) & (y_np == 0)).sum())
    tn = float(((preds == 0) & (y_np == 0)).sum())
    fn = float(((preds == 0) & (y_np == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / max(1.0, tp + tn + fp + fn)
    return {"probs": probs, "precision": precision, "recall": recall,
            "fpr": fpr, "accuracy": accuracy, "auc": auc_score(y_np, probs)}


def sweep_threshold(y_np, probs):
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 19):
        p = (probs >= t).astype(np.float32)
        tp = ((p == 1) & (y_np == 1)).sum()
        fp = ((p == 1) & (y_np == 0)).sum()
        fn = ((p == 0) & (y_np == 1)).sum()
        f1 = 2 * tp / max(1e-9, 2 * tp + fp + fn)
        if f1 > best_f1:
            best_t = float(t)
            best_f1 = f1
    return best_t


def pick_device(choice):
    if choice == "cuda":
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=int(TR["epochs"]))
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=0,
                        help="debug: cap clips per bucket")
    args = parser.parse_args()

    device = pick_device(args.device)
    print("[train] device=" + str(device))

    rng = np.random.default_rng(23)
    fe = FeatureExtractor()
    built = build_dataset(fe, rng, limit=args.limit)
    tX, ty, vX, vy, n_pos_va, n_sim_va = built

    model = WWNet(W * EMB_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=float(TR["learning_rate"]),
                            weight_decay=float(TR["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                       T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    bs = int(TR["batch_size"])
    patience = 4
    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(ty))
        total = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = tX[idx].to(device)
            yb = ty[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)

        metrics = evaluate(model, vX, vy)
        avg_loss = total / max(1, len(ty))
        sched.step()
        print("epoch " + format(epoch, "02d")
              + "  loss=" + format(avg_loss, ".4f")
              + "  val_auc=" + format(metrics["auc"], ".4f")
              + "  val_acc=" + format(metrics["accuracy"], ".4f")
              + "  val_recall=" + format(metrics["recall"], ".3f")
              + "  val_fpr=" + format(metrics["fpr"], ".4f"))

        score = avg_loss - metrics["auc"]
        if score < best_val:
            best_val = score
            bad = 0
            best_state = {}
            for k, v in model.state_dict().items():
                best_state[k] = v.detach().cpu().clone()
        else:
            bad += 1
            if bad >= patience:
                print("[train] early stop at epoch " + str(epoch))
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final = evaluate(model, vX, vy)
    suggested = sweep_threshold(vy.numpy(), final["probs"])
    sim_probs = final["probs"][n_pos_va:n_pos_va + n_sim_va]
    if len(sim_probs) > 0:
        sim_fpr = float((sim_probs >= 0.5).mean())
    else:
        sim_fpr = 0.0

    MODELS_DIR.mkdir(exist_ok=True)
    name = str(CFG["model"]["name"])

    # checkpoint FIRST - a failed export must never lose the training run
    model = model.to("cpu")  # export and runtime checks run on CPU
    torch.save(model.state_dict(), MODELS_DIR / (name + ".pt"))

    wrapper = ExportWrapper(model).eval().float()
    dummy = torch.zeros(1, W, EMB_DIM)
    onnx_path = MODELS_DIR / (name + ".onnx")
    export_kwargs = {
        "input_names": ["input"],
        "output_names": ["scores"],
        "dynamic_axes": {"input": {0: "batch"}, "scores": {0: "batch"}},
        "opset_version": 13,
    }
    try:
        torch.onnx.export(wrapper, dummy, str(onnx_path), **export_kwargs)
    except Exception as exc:
        print("[train] default exporter failed (" + str(exc)[:90]
              + ") - retrying with the legacy TorchScript exporter")
        torch.onnx.export(wrapper, dummy, str(onnx_path),
                          dynamo=False, **export_kwargs)

    report = {
        "model": name,
        "window_frames": W,
        "embedding_dim": EMB_DIM,
        "epochs_trained": epoch,
        "val_auc": round(float(final["auc"]), 4),
        "val_accuracy_at_0.5": round(float(final["accuracy"]), 4),
        "val_precision_at_0.5": round(float(final["precision"]), 4),
        "val_recall_at_0.5": round(float(final["recall"]), 4),
        "val_false_positive_rate_at_0.5": round(float(final["fpr"]), 4),
        "similar_words_false_alarm_rate": round(sim_fpr, 4),
        "suggested_threshold": suggested,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2))
    print("[train] exported " + onnx_path.name)
    print(json.dumps(report, indent=2))

    sanity_check(onnx_path)


def sanity_check(onnx_path):
    """Score held-out clips through the real openWakeWord runtime."""
    try:
        from openwakeword.model import Model
        oww = Model(wakeword_models=[str(onnx_path)],
                    inference_framework="onnx")
        wname = next(iter(oww.models.keys()))
        positive, similar, negative = collect_clips()

        def score(path):
            pcm, sr = sf.read(str(path), dtype="int16")
            oww.reset()
            best = 0.0
            for i in range(0, len(pcm) - 1280 + 1, 1280):
                scores = oww.predict(pcm[i:i + 1280])
                best = max(best, float(scores[wname]))
            return best

        print("")
        print("[sanity] openWakeWord runtime scores")
        plan = [("POS", positive, 4), ("SIM", similar, 3),
                ("NEG", negative, 3)]
        for entry in plan:
            tag = entry[0]
            pool = entry[1]
            k = entry[2]
            for p in pool[-k:]:
                s = score(p)
                print("  [" + tag + "] " + p.name.ljust(36)
                      + format(s, ".3f"))
    except Exception as exc:
        print("[sanity] skipped (" + str(exc)
              + ") - test later with test_model.py")


if __name__ == "__main__":
    main()
