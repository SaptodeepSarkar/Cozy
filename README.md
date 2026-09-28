# Cozy

Cozy is a voice-first personal Linux desktop assistant built on the open-source
[Hermes Agent](https://github.com/NousResearch/hermes-agent). The upstream
agent supplies the planner, tools and computer-use runtime; this repository
tracks a Cozy-branded fork as a Git submodule and keeps the local wake-word
model and audio research alongside it.

## Install and launch

Requirements: Linux, Git, `uv`, a working microphone/audio stack, and (for
desktop computer use) a supported graphical session. Hyprland support is
experimental: test basic screenshots, focus, clicks and typing before trusting
it with important work. Never enable unattended destructive actions.

```bash
git clone --recurse-submodules https://github.com/SaptodeepSarkar/Cozy.git
cd Cozy
bash setup.sh
./cozy
```

If you already cloned without submodules, run `git submodule update --init
--recursive hermes-agent` first. The first setup/desktop launch downloads
dependencies and may take a while. `./cozy --cli` opens a terminal chat with
computer-use tools; `./cozy --status` runs Hermes diagnostics.

Cozy's local wake classifier is integrated into the Hermes fork as the
`livekit` provider. Setup copies `wakeword/output/hey_cozy/hey_cozy.onnx` to
`$HERMES_HOME/wakewords/hey_cozy.onnx` (normally `~/.hermes/wakewords/`). Voice
and wake-word behavior can be configured in Hermes settings. Don't lower the
wake threshold blindly: a local 20-second negative calibration sample reached
0.72, while known positive samples peaked around 0.72. That overlap means a
single threshold cannot currently provide both reliable recall and few false
wakes; the detector needs more representative negative data and model work.

## What controls the computer

Hermes's `computer_use` toolset provides the agent's desktop interactions.
The Hermes desktop is the primary interface; the voice button supports push to
talk, while wake-word availability depends on local audio/configuration. The
agent may launch applications, operate windows and use supported computer-use
tools. Review commands with external or destructive effects before approving
them. Hyprland and other Wayland compositors vary in their input/accessibility
support, so computer-use coverage must be validated on the actual machine.

## Project layout

- `hermes-agent/` — Cozy's fork of Hermes Agent, tracked as a submodule.
- `wakeword/` — “Hey Cozy” dataset, training configuration and model workflow.
- `stt-finetune/` — retained speech-recognition experiments and training data.
- `audio/` — Linux audio routing configuration and helper scripts.

The retired `assistant/` runtime, bespoke RLM harness, FoxMCP integration and
custom LLM artifacts were removed from the working tree. Their previous state
is recoverable from the preceding Cozy GitHub commit if needed.

## Upstream and platform notes

- Hermes computer use: [docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/computer-use)
- Hermes voice mode: [docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/voice-mode)
- Hermes wake word: [docs](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/wake-word.md)
- Hermes Desktop: [docs](https://hermes-agent.nousresearch.com/docs/user-guide/desktop)
- CUA Linux platform support: [matrix](https://cua.ai/docs/reference/cua-driver/platform-support) (Hyprland status is experimental)

The Cozy fork is published at
https://github.com/SaptodeepSarkar/cozy-hermes and retains upstream licensing
and attribution. See `hermes-agent/LICENSE` for license details.
