# Cozy repository notes

Cozy is a voice-first desktop interface around the Hermes Agent fork. Keep
agent-runtime changes in the `hermes-agent/` submodule and commit/push them to
`SaptodeepSarkar/cozy-hermes` before updating this repository's submodule pin.

## Layout

- `hermes-agent/` — forked Hermes harness, GUI and tool integrations.
- `wakeword/` — “Hey Cozy” model, training and evaluation.
- `stt-finetune/` — retained speech-to-text experiments.
- `audio/` — local Linux audio helpers.
- `cozy`, `run.sh`, `setup.sh` — thin launch/setup wrappers.

## Critical local audio paths

- `wakeword/output/hey_cozy/hey_cozy.onnx` is the local wake model. Do not
  overwrite it without a reproducible evaluation; setup copies it to Hermes'
  wakewords directory.
- `stt-finetune/output/` contains user-trained STT artifacts; do not delete or
  regenerate them as part of agent work.
- Wake scores overlap: do not claim a robust trigger threshold based only on a
  short positive or negative clip. Collect labelled samples and report both
  false-positive and miss rates.

## Development

- Python 3.11+ and `uv`; Hermes owns its own dependency manager/environment.
- Initialize source with `git submodule update --init --recursive hermes-agent`.
- `bash setup.sh` installs the runtime and local LiveKit wake-word extra.
- Hermes tests must be run using `hermes-agent/scripts/run_tests.sh`.
- Keep commits under the owner's identity: `Saptodeep Sarkar
  <saptodeepsarkar28@gmail.com>`.
- Root commit format: `feat(...):` / `fix(...):`, or `vX.YZ:` for snapshots.

## Safety and platform

Computer use can operate the user's desktop and execute impactful actions.
Preserve confirmation and approval safeguards in Hermes; don't add silent
destructive behavior. Hyprland coverage is experimental; test supported input,
screenshots and accessibility paths on-device rather than assuming parity with
X11 or GNOME/KDE.
