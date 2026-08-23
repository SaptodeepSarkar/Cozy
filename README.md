# Cozy 🎙️

**Cozy** is a local, private, voice-controlled assistant. Say the word **"cozy"**
and your PC (and your AI agents) come alive.

This repository currently ships the foundation layer:

- **wakeword/** — a complete, reproducible pipeline that trains a custom
  wake-word model for the word *"cozy"* using synthetic multi-speaker TTS data
  ([Piper](https://github.com/rhasspy/piper-sample-generator)) and
  [openWakeWord](https://github.com/dscripka/openWakeWord)-style feature
  extraction. One command downloads every model, builds the dataset, trains
  the detector and exports it to ONNX.

## Repository layout

    Cozy/
    ├── wakeword/                  <- wake-word training pipeline
    │   ├── config.yaml            <- words, counts, hyperparameters
    │   ├── setup.sh               <- creates venv + installs everything
    │   ├── download_models.py     <- downloads TTS voices + feature models
    │   ├── generate_data.py       <- synthesizes all training audio
    │   ├── train_wakeword.py      <- trains + exports models/cozy_v1.onnx
    │   ├── record_samples.py      <- record YOUR voice into data/cozy
    │   ├── test_model.py          <- live microphone / WAV testing
    │   ├── run_all.sh             <- one command: download -> data -> train
    │   ├── models/                <- trained artifacts land here (committed)
    │   └── data/
    │       ├── cozy/              <- recordings of the word "cozy"   (positives)
    │       └── similar/           <- recordings of similar sounds  (hard negatives)
    └── assistant/                 <- (roadmap) the brain that listens and controls

## Quickstart

    git clone https://github.com/SaptodeepSarkar/Cozy.git
    cd Cozy/wakeword
    bash setup.sh            # one-time: venv + PyTorch + dependencies
    bash run_all.sh smoke    # ~10 min end-to-end sanity check
    bash run_all.sh full     # full dataset (≈18k clips) + real training

Then speak to your machine:

    python test_model.py --mic          # say "cozy"
    python record_samples.py --num 20   # teach it your voice (recommended)

## How the model learns "all aspects" of the word

| Folder          | Role                                                        |
| --------------- | ----------------------------------------------------------- |
| data/cozy     | Real + synthetic recordings of **"cozy"** (incl. the British spelling *cosy*) |
| data/similar  | Confusable words — *nosy, rosy, Josie, Ozzie, dozy, posy, noisy* … so the model learns what is **not** cozy |

On top of that, generate_data.py synthesizes thousands of everyday
sentences that must never trigger, so the detector stays quiet during
normal conversation.

## Roadmap — the full Cozy assistant

1. Listener daemon: wake word -> faster-whisper STT -> intent router
2. Agent bridge: route commands to local coding agents (e.g. DSH sessions)
3. PC control: app launching, media, keyboard/mouse, smart-home hooks
4. Safety: audible confirmation + kill phrase ("goodnight cozy")

## Credits & license

Built on the excellent open-source work of
[openWakeWord](https://github.com/dscripka/openWakeWord) (Apache-2.0) and
[Piper](https://github.com/rhasspy/piper) (MIT). Code in this repo is MIT —
see [LICENSE](LICENSE).
