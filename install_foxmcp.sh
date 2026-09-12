#!/usr/bin/env bash
# Install FoxMCP into Cozy's private runtime directory.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="${COZY_FOXMCP_ROOT:-$ROOT/.cozy/foxmcp}"
if [[ -e "$DEST/.git" ]]; then
  git -C "$DEST" pull --ff-only
else
  mkdir -p "$(dirname "$DEST")"
  git clone --depth 1 https://github.com/ThinkerYzu/foxmcp.git "$DEST"
fi
make -C "$DEST" install
echo "FoxMCP installed at $DEST"
echo "Install the FoxMCP Firefox extension, then run: bash run.sh"
