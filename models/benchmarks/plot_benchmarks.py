#!/usr/bin/env python3
"""Render the three v1.0 vs v1.1 benchmark plots and a summary table
for the README.

Inputs (already on disk after eval_wakeword / eval_stt / eval_llm):
    models/hey_cozy-v1.0/hey_cozy_eval.json
    models/hey_cozy-v1.1/hey_cozy_eval.json
    models/benchmarks/stt_eval.json
    models/benchmarks/llm_eval.json

Outputs:
    models/benchmarks/wakeword_v1_vs_v1.1.png
    models/benchmarks/stt_wer_v1_vs_v1.1.png
    models/benchmarks/llm_toolcall_v1_vs_v1.1.png
    models/benchmarks/summary.csv
    models/benchmarks/summary.md

Run from the repo root:
    python models/benchmarks/plot_benchmarks.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MODELS = REPO / "models"


# ---------------------------------------------------------------------- wake
def load_wake(version: str) -> dict:
    p = MODELS / f"hey_cozy-{version}" / "hey_cozy_eval.json"
    return json.loads(p.read_text())


def plot_wake() -> None:
    v10 = load_wake("v1.0")
    v11 = load_wake("v1.1")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # FPPH (lower better)
    ax = axes[0]
    bars = ax.bar(["v1.0", "v1.1"],
                  [v10["fpph"], v11["fpph"]],
                  color=["#888", "#2a9d8f"])
    ax.set_ylabel("FPPH (false-positives / hour)  lower is better")
    ax.set_title(f"False-positives / hour\n(threshold 0.5)")
    for b, v in zip(bars, [v10["fpph"], v11["fpph"]]):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:.2f}", ha="center", va="bottom", fontsize=11)

    # Recall (higher better)
    ax = axes[1]
    bars = ax.bar(["v1.0", "v1.1"],
                  [v10["recall"] * 100, v11["recall"] * 100],
                  color=["#888", "#2a9d8f"])
    ax.set_ylabel("Recall (%)  higher is better")
    ax.set_ylim(0, 100)
    ax.set_title("Recall at default threshold")
    for b, v in zip(bars, [v10["recall"] * 100, v11["recall"] * 100]):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=11)

    # AUT
    ax = axes[2]
    bars = ax.bar(["v1.0", "v1.1"],
                  [v10["aut"] * 100, v11["aut"] * 100],
                  color=["#888", "#2a9d8f"])
    ax.set_ylabel("AUT (×100, lower is better)")
    ax.set_title(f"AUT (Area Under Threshold curve)\n"
                 f"val = {v10['n_negative']:,} neg / {v10['n_positive']} pos")
    for b, v in zip(bars, [v10["aut"] * 100, v11["aut"] * 100]):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:.2f}", ha="center", va="bottom", fontsize=11)

    fig.suptitle(f"hey_cozy wake word  (v1.0 → v1.1 retrain on 138 user-voice + 2568 negatives)",
                 fontsize=13)
    fig.tight_layout()
    out = HERE / "wakeword_v1_vs_v1.1.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------- STT
def plot_stt() -> None:
    p = HERE / "stt_eval.json"
    if not p.exists():
        print(f"  skip STT plot ({p} missing)")
        return
    d = json.loads(p.read_text())
    v10 = d["v1.0"]
    v11 = d["v1.1"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    bars = ax.bar(["v1.0\nbase", "v1.1\nLoRA"],
                  [v10["wer"] * 100, v11["wer"] * 100],
                  color=["#888", "#2a9d8f"])
    ax.set_ylabel("WER (%)  lower is better")
    ax.set_title(f"Word Error Rate on {v10['n_clips']} held-out Indian-English clips")
    for b, v in zip(bars, [v10["wer"] * 100, v11["wer"] * 100]):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=11)

    ax = axes[1]
    bars = ax.bar(["v1.0\nbase", "v1.1\nLoRA"],
                  [v10["rtf"], v11["rtf"]],
                  color=["#888", "#2a9d8f"])
    ax.set_ylabel("Real-Time Factor  lower is faster")
    ax.set_title(f"Inference speed (RTF)\n"
                 f"engine: {v10.get('engine', '?')}")

    fig.suptitle("cozy_stt  (v1.0 = openai/whisper-small base  →  v1.1 = Cozy LoRA on 1425 Hinglish clips)",
                 fontsize=13)
    fig.tight_layout()
    out = HERE / "stt_wer_v1_vs_v1.1.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------- LLM
def plot_llm() -> None:
    p = HERE / "llm_eval.json"
    if not p.exists():
        print(f"  skip LLM plot ({p} missing)")
        return
    d = json.loads(p.read_text())
    v10 = d["v1.0"]
    v11 = d["v1.1"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(3)
    width = 0.35
    metrics = ["accuracy", "tool_accuracy", "chat_accuracy"]
    labels = ["Overall", "Tool calls", "Chitchat\n(no-tool)"]
    v10_vals = [v10["accuracy"] * 100, v10["tool_accuracy"] * 100, v10["chat_accuracy"] * 100]
    v11_vals = [v11["accuracy"] * 100, v11["tool_accuracy"] * 100, v11["chat_accuracy"] * 100]
    b1 = ax.bar(x - width/2, v10_vals, width, color="#888", label="v1.0")
    b2 = ax.bar(x + width/2, v11_vals, width, color="#2a9d8f", label="v1.1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)  higher is better")
    ax.set_ylim(0, 100)
    ax.set_title(f"Qwen3-0.6B SFT  -  tool-call & chitchat accuracy\n"
                 f"({v10['n_total']} probes, greedy decoding)")
    ax.legend()
    for b, v in zip(b1, v10_vals):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:.0f}", ha="center", va="bottom", fontsize=10)
    for b, v in zip(b2, v11_vals):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:.0f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    out = HERE / "llm_toolcall_v1_vs_v1.1.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------- summary
def write_summary() -> None:
    wake10 = load_wake("v1.0")
    wake11 = load_wake("v1.1")
    stt = json.loads((HERE / "stt_eval.json").read_text()) if (HERE / "stt_eval.json").exists() else None
    llm = json.loads((HERE / "llm_eval.json").read_text()) if (HERE / "llm_eval.json").exists() else None

    rows = [
        ("hey_cozy",  "FPPH (1/hr)",     f"{wake10['fpph']:.2f}",   f"{wake11['fpph']:.2f}"),
        ("hey_cozy",  "Recall",          f"{wake10['recall']:.2f}", f"{wake11['recall']:.2f}"),
        ("hey_cozy",  "AUT",             f"{wake10['aut']:.3f}",    f"{wake11['aut']:.3f}"),
    ]
    if stt:
        rows.append(("cozy_stt", "WER",      f"{stt['v1.0']['wer']:.3f}",     f"{stt['v1.1']['wer']:.3f}"))
        rows.append(("cozy_stt", "RTF",      f"{stt['v1.0']['rtf']:.3f}",     f"{stt['v1.1']['rtf']:.3f}"))
    if llm:
        rows.append(("cozy-llm", "tool-call acc", f"{llm['v1.0']['tool_accuracy']:.3f}", f"{llm['v1.1']['tool_accuracy']:.3f}"))
        rows.append(("cozy-llm", "chitchat acc",  f"{llm['v1.0']['chat_accuracy']:.3f}",  f"{llm['v1.1']['chat_accuracy']:.3f}"))
        rows.append(("cozy-llm", "overall acc",   f"{llm['v1.0']['accuracy']:.3f}",      f"{llm['v1.1']['accuracy']:.3f}"))

    csv = HERE / "summary.csv"
    with open(csv, "w") as f:
        f.write("model,metric,v1.0,v1.1\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    md = HERE / "summary.md"
    lines = ["# Cozy model benchmark summary\n",
             "Trained on RTX 3050 6 GB, all numbers from `models/benchmarks/`.\n",
             "| model | metric | v1.0 | v1.1 |",
             "|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    md.write_text("\n".join(lines) + "\n")
    print(f"  -> {csv}")
    print(f"  -> {md}")


def main() -> None:
    print("[1/3] wake word ...")
    plot_wake()
    print("[2/3] STT ...")
    plot_stt()
    print("[3/3] LLM ...")
    plot_llm()
    print("[summary]")
    write_summary()


if __name__ == "__main__":
    main()
