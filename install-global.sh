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
for name in cozystop cozystatus; do
  ln -sfn "$ROOT/cozy" "$BIN/$name"
done
for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
  touch "$rc"
  if ! grep -qF "# Cozy user-local bin" "$rc" 2>/dev/null; then
    printf '\n# Cozy user-local bin\nexport PATH="%s:$PATH"\n' "$BIN" >> "$rc"
  fi
done
case ":${PATH}:" in *":$BIN:"*) ;; *) export PATH="$BIN:$PATH" ;; esac
echo "Cozy is globally available as: $TARGET"
