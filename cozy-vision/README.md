# Cozy-Vision

A local desktop GUI agent for Pop!_OS / COSMIC. Sits next to the Cozy
voice stack but lives in its own folder because it adds vision
models.

## What it does

You give it a natural-language goal (typed or spoken via the cozy
STT). It:

1. **Plans** with a 3B Vision-Language Model (Qwen2.5-VL-3B-Instruct,
   NF4 4-bit) — given the current screen + the live OS context
   (active window, open windows, focused element, working area), the
   VLM breaks the goal into a precise, ordered, *checkable* todo
   list. Each todo item has `action`, `target`, `check`, and optional
   `params`.
2. **Executes** each todo item with a 2B Vision-Language-Action model
   (UI-TARS-2B-SFT, NF4 4-bit) — a native GUI agent that directly
   maps (screenshot + instruction) to a single action per turn
   (`click(x, y)`, `type(text)`, `hotkey(k1, k2)`, `scroll(dx, dy)`,
   `wait()`, `finished()`).
3. **Verifies** each todo with the reward module (process check,
   window-title check, URL check, OS-state diff, click-in-bbox).
4. **Answers** free-form visual questions about the screen
   ("is the terminal focused?", "what does the title bar say?")
   without planning or execution.

The whole pipeline is captured as JSONL traces so the VLM and VLA
can be fine-tuned (QLoRA, 6 GB-friendly) on the user's own desktop
behaviour.

## Models (downloaded by setup.sh)

| Role | Model | Disk | GPU @ NF4 4-bit |
|------|-------|------|-----------------|
| VLM (planner + Q&A) | `Qwen/Qwen2.5-VL-3B-Instruct` | 7.5 GB | 2.4 GB |
| VLA (executor) | `ByteDance-Seed/UI-TARS-2B-SFT` | 9.8 GB | 1.5 GB |

Both run on the same 6 GB dGPU alongside the existing Cozy stack
(cozy-llm-v1 1.2 GB + STT 0.5 GB + wake 0.1 GB = 5.7 GB peak).

## Quick start

```bash
cd cozy-vision
bash setup.sh                # one-shot venv + model fetch (~15 min)
bash run.sh smoke            # load both models, synthetic inference
bash run.sh plan "open firefox"
bash run.sh ask "what is the title of the active window?"
bash run.sh run "close the current window"
bash run.sh collect --tasks 5 # collect SFT traces
bash run.sh train             # QLoRA SFT on the VLM
bash run.sh train-vla         # QLoRA SFT on the VLA
```

System tools the agent needs (install once with `sudo apt install
ydotool xdotool grim wl-clipboard i3-wm`):
- `ydotool` (or `xdotool`) — mouse + keyboard synthesis
- `grim` (or `wlr-screencopy`) — screenshot
- `swaymsg` (or `xdotool`) — OS context (active window, open windows)

## Architecture

```
[User prompt]  ──voice──>  cozy STT ──text──>  [Text goal]
                                                  │
                                                  ▼
              ┌──────────────────────────────────────────────┐
              │  OSContextCollector (swaymsg / xdotool /     │
              │  AT-SPI)                                     │
              │   - active window, open windows, focused     │
              │     element, working area, clipboard,        │
              │     cursor, recent apps                      │
              └────────────────────┬─────────────────────────┘
                                   │  context
                                   ▼
                          ┌─────────────────┐
                          │   VLM (3B NF4)  │  <── screen + goal + context
                          │  (planner + Q&A)│
                          └────────┬────────┘
                                   │  todo list
                                   ▼
              ┌─────────────────────────────────────────────┐
              │  VisionRunner                                │
              │  for each todo:                              │
              │    for each step:                            │
              │      ┌─────────────────┐                     │
              │      │  VLA (2B NF4)   │ <── screen         │
              │      │  (executor)     │ ──> action         │
              │      └────────┬────────┘                     │
              │               │                              │
              │               ▼                              │
              │         PopOSDriver                          │
              │         (ydotool / grim / D-Bus)             │
              │               │                              │
              │               ▼                              │
              │         reward.check() ──── verify           │
              └─────────────────────────────────────────────┘
```

## Folder layout

```
cozy-vision/
├── README.md                    this file
├── pyproject.toml               Python deps
├── setup.sh                     one-shot venv + model fetch
├── run.sh                       launcher with GPU memory caps
├── models/
│   ├── ui-tars-2b-sft/          VLA (fp16 on disk, NF4 at runtime)
│   └── qwen2.5-vl-3b/           VLM (fp16 on disk, NF4 at runtime)
├── data/
│   ├── gemini_chat_source.txt   design notes from the Gemini chat
│   ├── sft_planner_seed.jsonl   synthetic SFT data for VLM
│   ├── sft_vla_seed.jsonl       synthetic SFT data for VLA
│   ├── screenshots/             fake desktop PNGs (synthetic)
│   └── traces/                  real collected JSONL traces
├── scripts/
│   ├── download_models.py       one-shot HF fetch with aria2c
│   ├── make_sft_seed.py         synthetic SFT data generator
│   └── smoke_test.py            load both models + run inference
├── harness/
│   ├── __init__.py
│   ├── context.py               OSContextCollector (sway / xdotool / AT-SPI)
│   ├── driver.py                PopOSDriver (ydotool / grim)
│   ├── planner.py               VLM wrapper (plan + answer)
│   ├── grounder.py              VLA wrapper (UI-TARS action parser)
│   ├── runner.py                VisionRunner (planner + VLA + driver)
│   ├── reward.py                RLVR-style reward functions
│   ├── tasks.py                 seed task JSONL
│   ├── train_sft.py             QLoRA SFT for the VLM
│   ├── train_sft_vla.py         QLoRA SFT for the VLA
│   └── cli.py                   cozy-vision subcommands
└── checkpoints/                 LoRA adapters land here
```

## Hardware budget (RTX 3050 6 GB Laptop GPU)

| Component              | Device | Rest  | Peak   |
|------------------------|--------|------:|-------:|
| cosy-llm-v1 (Qwen3-0.6B bf16) | cuda:0 | 1.2 GB | 1.4 GB |
| faster-whisper-small int8      | cuda:0 | 0.5 GB | 1.0 GB |
| wake (livekit ONNX)             | cpu    | 0      | 0.1 GB |
| **VLM Qwen2.5-VL-3B (NF4)**     | cuda:0 | 2.4 GB | 2.4 GB |
| **VLA UI-TARS-2B (NF4)**        | cuda:0 | 1.5 GB | 1.5 GB |
| AT-SPI / ydotool / grim         | cpu    | 0      | 0.1 GB |
| dGPU free headroom              |        | 0.4 GB |  —     |

Both vision models live entirely on the dGPU (NF4 4-bit, no CPU
offload). cosy-llm-v1 is the existing Cozy voice chat LLM; when the
vision agent is running, cosy-llm-v1 should be unloaded to keep the
dGPU headroom healthy.

## License

Both models are Apache-2.0. This folder is part of the Cozy project
(also Apache-2.0).
