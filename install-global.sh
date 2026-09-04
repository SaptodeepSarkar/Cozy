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
case ":${PATH}:" in *":$BIN:"*) ;; *) echo "Add $BIN to PATH (for example: export PATH=\"$BIN:\$PATH\")" ;; esac
echo "Cozy is globally available as: $TARGET"
