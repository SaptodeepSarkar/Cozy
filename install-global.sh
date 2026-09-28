#!/usr/bin/env bash
# Install a user-local global launcher (no sudo required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
BIN="${COZY_BIN_DIR:-$HOME/.local/bin}"
mkdir -p "$BIN"
TARGET="$BIN/cozy"
if [[ -e "$TARGET" && ! -L "$TARGET" ]]; then
  echo "Refusing to overwrite existing $TARGET" >&2
  exit 1
fi
ln -sfn "$ROOT/cozy" "$TARGET"
ln -sfn "$ROOT/cozy" "$BIN/cozystatus"

# Install a user application-menu entry so the voice-first UI is launchable
# without opening a terminal. The entry always targets this checked-out fork.
APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$APP_DIR"
ICON="$ROOT/hermes-agent/apps/desktop/assets/icon.png"
if [[ ! -f "$ICON" ]]; then ICON="utilities-terminal"; fi
sed -e "s|@COZY_LAUNCHER@|$TARGET|g" -e "s|@COZY_ICON@|$ICON|g" \
  "$ROOT/cozy.desktop.in" > "$APP_DIR/com.nousresearch.hermes.desktop"
chmod 0644 "$APP_DIR/com.nousresearch.hermes.desktop"
# Hermes installs the launcher under this desktop-file ID too. Reuse that ID
# above, and remove the redundant Cozy entry created by earlier installer runs.
rm -f "$APP_DIR/cozy.desktop"
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$APP_DIR" >/dev/null 2>&1 || true
fi
for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
  touch "$rc"
  if ! grep -qF "# Cozy user-local bin" "$rc" 2>/dev/null; then
    printf '\n# Cozy user-local bin\nexport PATH="%s:$PATH"\n' "$BIN" >> "$rc"
  fi
done
case ":${PATH}:" in *":$BIN:"*) ;; *) export PATH="$BIN:$PATH" ;; esac
echo "Cozy is globally available as: $TARGET"
echo "Cozy desktop launcher installed at: $APP_DIR/com.nousresearch.hermes.desktop"
