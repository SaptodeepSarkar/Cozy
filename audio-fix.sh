#!/usr/bin/env bash
# Configure Cozy's Linux microphone path: safe ALSA gain + PipeWire RNNoise.
# This is user-local and never requires sudo.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
CFG_DIR="$HOME/.config/pipewire/pipewire.conf.d"
UNIT_DIR="$HOME/.config/systemd/user"

mkdir -p "$CFG_DIR" "$UNIT_DIR"
install -m 0644 "$ROOT/audio/linux/cozy-rnnoise.conf" "$CFG_DIR/cozy-rnnoise.conf"
install -m 0644 "$ROOT/audio/linux/cozy-mic-gain.service" "$UNIT_DIR/cozy-mic-gain.service"
install -m 0755 "$ROOT/audio/linux/cozy-audio-route" "$HOME/.local/bin/cozy-audio-route"
install -m 0644 "$ROOT/audio/linux/cozy-audio-route.service" "$UNIT_DIR/cozy-audio-route.service"

systemctl --user daemon-reload
systemctl --user enable --now cozy-mic-gain.service
systemctl --user enable --now cozy-audio-route.service
systemctl --user restart pipewire pipewire-pulse wireplumber
sleep 3
pactl set-source-volume alsa_input.pci-0000_00_1f.3.analog-stereo 100%
pactl set-default-source effect_output.cozy-rnnoise

echo "Cozy audio fix applied. Default source: $(pactl get-default-source)"
