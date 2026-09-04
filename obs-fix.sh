#!/usr/bin/env bash
# Recover OBS screen capture on Hyprland after stale portal sessions.
set -euo pipefail

if pgrep -x obs >/dev/null 2>&1; then
  echo "Close OBS before running this helper." >&2
  exit 1
fi

systemctl --user restart xdg-desktop-portal-hyprland xdg-desktop-portal-gtk xdg-desktop-portal
sleep 2
if ! systemctl --user is-active --quiet xdg-desktop-portal-hyprland; then
  echo "Hyprland screen portal failed to start." >&2
  exit 1
fi
echo "OBS screen-capture portal is ready. Reopen OBS and add Screen Capture (PipeWire)."
