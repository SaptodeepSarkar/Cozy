# Cozy Release History

## v1.53 (working tree)
**Typed terminal UI + reproducible training pipeline**
- Replaced the single-file Ink `.mjs` UI with strict TypeScript/TSX components,
  responsive layout, bounded activity log, and supervised backend restart/error
  handling (`Ctrl+R`).
- Fixed the JSON runtime contract to emit `done`, serialized commands, and
  surfaced audio-loop failures instead of silently losing the voice thread.
- Added `train.sh`: resumable SFT → RLVR → DPO → STT export/benchmark runs with
  per-stage logs, manifests, GPU/dtype/peak-VRAM telemetry, and smoke/standard/
  quality profiles.
- Added a held-out 30-probe RLVR set and exact tool/name/parameter verifier;
  fixed DPO's reference model to use the frozen base adapter-disabled path.
- Fixed LLM dataset tuple rows and train/validation prompt leakage; STT
  manifests now resolve relative audio paths from any working directory.
- Restored the wakeword memory-mapped feature dataset module used by classifier
  training. Synthetic wakeword generation remains an explicit upstream data
  dependency in this checkout.

## v1.52 (previous)
**Wake + STT + LLM re-trained from scratch; versioned models/ tree + benchmarks**
- **hey_cozy v1.1**: 1500 synth + 32 user-voice positives, 1500 adversarial
  + 1500 background negatives, 3 rounds of augmentation, 5000 steps
  adaptive. AUT **0.002** (8.3× better), FPPH **0.00** (was 1.66), Recall
  **96%** (was 69%).
- **cozy_stt v1.1**: r=32 LoRA on 1425 Hinglish clips (cv_indian +
  santhosh_indian), 2 epochs. WER **9.55%** on 125-clip held-out
  Indian-English eval (was 21.98% on the base whisper-small).
- **cozy-llm v1.1**: Qwen3-0.6B + LoRA r=16, 1 epoch on 2739 SFT rows.
  Tool-call accuracy **63%** (down from 84% on the un-tuned base — see
  benchmark for root cause; needs 3 epochs + DPO, planned for v1.53).
- **Versioned models/** tree (gitignored): `models/{hey_cozy,cozy_stt,cozy-llm}-v1.{0,1}/`
  + `models/benchmarks/{wakeword,stt,llm}_v1_vs_v1.1.png` + `summary.md/csv`.
- **cozy-vision/ deleted** (~7.5K lines of unused GUI-agent code).
- Re-trained models are now symlinked into the runtime paths
  (`wakeword/output/`, `stt-finetune/output/`, `assistant/model/`) so
  `cozy` launches use v1.1 transparently.

## v1.51
**Arch Linux compatibility + cross-platform path fixes**
- `setup.sh` rewritten with `uv` (seeded venvs + `uv pip install`); fixes
  the v1.49 silent "No module named pip" error and runs in ~3 min on a
  fresh Arch install.
- `setup.sh` now auto-downloads Qwen3-0.6B into
  `assistant/model/cozy-llm-v1/` so `cozy` works on first launch without
  a separate training run.
- All hardcoded `/home/saptodeepsarkar/...` paths replaced with
  `Path(__file__).resolve().parents[N] / ...` — fixes the broken `cozy`
  shell alias and four `team/scripts/*.py` files.
- `stt-finetune/scripts/train_me.py:111` `SyntaxError: unmatched ')'` fixed.
- 13 stt-finetune scripts: shebang `#!/usr/bin/env python` → `python3`
  (Arch ships no `/usr/bin/python`).
- `assistant/rlm_harness/`: 10 broken `from rlm_harness.X` absolute
  imports replaced with relative `from .X` (would crash on first
  `FastHarness` use).
- `wakeword/src/livekit/wakeword/data/` package restored as stubs that
  raise an informative error — the `livekit.wakeword` lazy imports in
  `cli.py` and `__init__.py` no longer crash on `ModuleNotFoundError`.
- `tui_textual.py:791` wake-model path now uses `Path(__file__).parent.parent`
  instead of a hardcoded absolute path (voice mode in the Textual TUI).
- `cozy-vision/setup.sh` now detects `pacman` and installs the Arch
  equivalents (`ydotool grim slurp wl-clipboard xdotool xwayland-run
  sway libxkbcommon libxcb portaudio ffmpeg espeak-ng wlroots`).
- `assistant/executor.py`: Wayland/PipeWire-first tool fallbacks —
  `wpctl` before `pactl` before `amixer`; `grim` before `gnome-screenshot`;
  `swaylock`/`hyprlock`/`waylock` added to the lock chain; KDE / XFCE
  app aliases added (`thunar`, `dolphin`, `konsole`, `alacritty`,
  `kcalc`, `kate`, etc.).
- 7 `from_pretrained(..., dtype=torch.bfloat16)` calls converted to
  `torch_dtype=` (works on both `transformers` 4.x and 5.x).
- `cozy` sanity check no longer requires the STT models (which are
  loaded lazily by `CozySTT.transcribe_*`); only wake + LLM are needed
  at launch.
- `stt-finetune/data/santhosh_indian/manifest.jsonl`: 50 audio paths
  rebased to be relative to `stt-finetune/`, so the corpus works
  regardless of the user's home directory.

## v1.50
**Node Ink TUI + fast RLM harness + prime-agent audit**
(unchanged from previous tag)

## v1.49
**Wire assistant to livekit-wakeword + user-voice retrain**
- Assistant runtime switched from openwakeword (cozy_v1.onnx) to
  livekit-wakeword (hey_cozy.onnx)
- Wake word model retrained with 138 user-voice positives + 2568 negatives
  (AUT 0.020, FPPH 1.66, Recall 69%)
- Default threshold updated to 0.30
- Three-venv setup (`wakeword/`, `stt-finetune/`, `assistant/`)
- Comprehensive READMEs in every folder
- `setup.sh` and `run.sh` at repo root for one-shot install + launch

## v1.48
**Wakeword → livekit-wakeword + trained hey_cozy v1**
- Vendored livekit-wakeword v0.2.0 (replaces custom CozyNet v1/v2)
- Trained hey_cozy model (122 KB ONNX, AUT 0.051, FPPH 12.18, Recall 68%)
- 5 YAML configs, full pytest suite, Swift package
- Removed 1.1 GB LLM safetensors that blocked pushes

## v1.47 (pre-wakeword rewrite)
- Full repo snapshot before wakeword pipeline swap
- Custom CozyNet v1/v2 with openwakeword embeddings
- 32 user "cozy" recordings, 60+ hard-negative similar-word recordings
- ASHA-based training with energy gate
- AUC ~0.997 on synthetic test set, 0.76 stream-level AUC

## Earlier
- LLM SFT: Qwen3-0.6B + LoRA, 1.4 k function-call samples
- STT: whisper-small + LoRA v3, WER 9.92% on user holdout
- Cozy assistant runtime: STT → LLM → executor (15 tools)
- 657+ user voice recordings across 44 sessions
- Indian English accent adaptation (kaushalgawri, santhosh corpora)
