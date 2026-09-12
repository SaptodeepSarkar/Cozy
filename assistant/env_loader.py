"""Small dependency-free loader for the repository's local .env file."""
from __future__ import annotations

import os
from pathlib import Path


def load_project_env() -> None:
    """Load root .env values without overriding explicitly exported values."""
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
