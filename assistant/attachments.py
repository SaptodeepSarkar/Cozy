"""Bounded ``@file`` attachments for Cozy prompts.

Attachments are read only when the user explicitly references them. Text is
size-limited before it enters the LLM context; binary files contribute
metadata instead of unsafe or enormous decoded content.
"""
from __future__ import annotations

import mimetypes
import os
import re
from pathlib import Path

_REF = re.compile(r"(?<!\S)@(?P<path>(?:[^\s'\"]|\\.)+)")
TEXT_TYPES = {".txt", ".md", ".rst", ".json", ".jsonl", ".yaml", ".yml",
              ".toml", ".ini", ".cfg", ".csv", ".py", ".js", ".ts", ".tsx",
              ".jsx", ".html", ".css", ".sh", ".sql", ".xml"}


def find_paths(text: str) -> list[Path]:
    """Return existing, user-referenced paths in first-seen order."""
    out: list[Path] = []
    for match in _REF.finditer(text):
        raw = match.group("path").replace("\\ ", " ")
        path = Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
        if path.exists() and path not in out:
            out.append(path)
    return out


def render(text: str, *, max_file_chars: int = 6000,
           max_total_chars: int = 12000) -> tuple[str, list[dict]]:
    """Append bounded attachment blocks and return ``(prompt, metadata)``."""
    paths = find_paths(text)
    if not paths:
        return text, []
    blocks: list[str] = []
    metadata: list[dict] = []
    used = 0
    for path in paths:
        try:
            size = path.stat().st_size
            suffix = path.suffix.lower()
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            item = {"path": str(path), "name": path.name, "size": size, "mime": mime}
            if suffix in TEXT_TYPES or mime.startswith("text/") or mime == "application/json":
                content = path.read_text(encoding="utf-8", errors="replace")
                remaining = max(0, min(max_file_chars, max_total_chars - used))
                content = content[:remaining]
                used += len(content)
                item["chars"] = len(content)
                blocks.append(f"\n[Attached file: {path}]\n```\n{content}\n```")
            else:
                blocks.append(f"\n[Attached binary file: {path} ({mime}, {size} bytes)]")
            metadata.append(item)
        except OSError as exc:
            metadata.append({"path": str(path), "error": str(exc)})
    return text + "".join(blocks), metadata
