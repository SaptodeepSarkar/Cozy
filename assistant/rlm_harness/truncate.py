"""Tool-output truncation - copy of Prime Agent's truncate.ts.

The single biggest risk in Cozy is a long STT transcript or web
search result round-tripping a wall of text back into the LLM
context. This module ensures tool outputs never exceed a sane size
in the prompt.
"""
from __future__ import annotations

DEFAULT_MAX_LINES = 200
DEFAULT_MAX_BYTES = 8_000  # ~2k tokens


def truncate_tail(text: str, max_lines: int = DEFAULT_MAX_LINES,
                  max_bytes: int = DEFAULT_MAX_BYTES) -> str:
    """Return ``text`` truncated to the last ``max_lines`` lines, capped
    at ``max_bytes`` characters. If truncation happened, a pointer to
    the full output is appended so the LLM can ask for it.
    """
    if not text:
        return ""
    if len(text) <= max_bytes:
        return text
    # Split into lines, keep the last max_lines
    lines = text.splitlines()
    total_lines = len(lines)
    if total_lines > max_lines:
        lines = lines[-max_lines:]
        suffix = f"\n[Showing last {max_lines} of {total_lines} lines. Full: /tmp/cozy-truncated.log]"
    else:
        suffix = ""
    truncated = "\n".join(lines)
    # If still over bytes, hard-trim
    if len(truncated) > max_bytes:
        truncated = truncated[-max_bytes:]
        suffix = f"\n[...truncated to {max_bytes} bytes. Full: /tmp/cozy-truncated.log]"
    return truncated + suffix


def truncate_head(text: str, max_lines: int = DEFAULT_MAX_LINES,
                  max_bytes: int = DEFAULT_MAX_BYTES) -> str:
    """Keep the first max_lines / max_bytes."""
    if not text:
        return ""
    if len(text) <= max_bytes:
        return text
    lines = text.splitlines()
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n[...{len(lines)-max_lines} more lines...]"
    return text[:max_bytes] + f"\n[...truncated to {max_bytes} bytes...]"
