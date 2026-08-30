"""Cozy harness state - persistent prompt notes, memory, refinements.

Three CRUD stores on disk, modeled on Prime Agent's harness state:

  MemoryStore: user facts the LLM should always remember
    ~/.cozy/state/memory.json  (key -> value)
  NotesStore: persistent rules the LLM reads every turn
    ~/.cozy/state/notes.json   (id -> {title, content, scope})
  RefinementStore: agent-curated improvements over time
    ~/.cozy/state/refinements.jsonl  (append-only)

The system prompt composer in build_system_prompt() reads all three
and weaves them into the LLM's system message.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

STATE_DIR = Path.home() / ".cozy" / "state"
MEMORY_FILE = STATE_DIR / "memory.json"
NOTES_FILE = STATE_DIR / "notes.json"
REFINEMENTS_FILE = STATE_DIR / "refinements.jsonl"


def _ensure_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        # A corrupt file must not crash the caller.
        return default


def _write_json(path: Path, data) -> None:
    """Atomic write: temp + rename."""
    _ensure_dir()
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------- memory
class MemoryStore:
    """User facts the LLM should always remember.

    Schema: {key: value}. Keys are short identifiers (e.g. "user_name",
    "preferred_music"). Values are arbitrary strings.
    """
    def __init__(self, path: Path = MEMORY_FILE):
        self.path = path
        self._data: dict[str, str] = _read_json(path, {})

    def list(self) -> list[tuple[str, str]]:
        return sorted(self._data.items())

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def add(self, key: str, value: str) -> None:
        self._data[key] = value
        _write_json(self.path, self._data)

    def remove(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            _write_json(self.path, self._data)
            return True
        return False

    def as_prompt_block(self) -> str:
        if not self._data:
            return ""
        lines = ["# Memory (always remember these facts about the user)"]
        for k, v in sorted(self._data.items()):
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)


# ---------- notes (persistent rules the LLM reads every turn)
@dataclass
class Note:
    id: str
    title: str
    content: str
    created_at: str = ""
    updated_at: str = ""

    @staticmethod
    def new(title: str, content: str) -> "Note":
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        return Note(id=uuid.uuid4().hex[:12], title=title, content=content,
                    created_at=now, updated_at=now)


class NotesStore:
    """Persistent rules the LLM reads every turn.

    Each note is a (title, content) pair with a stable id. Notes are
    auto-rendered into the system prompt (one bullet per note).
    """
    def __init__(self, path: Path = NOTES_FILE):
        self.path = path
        self._data: dict[str, dict] = _read_json(path, {})

    def list(self) -> list[Note]:
        return [Note(**v) for v in self._data.values()]

    def get(self, id: str) -> Note | None:
        d = self._data.get(id)
        return Note(**d) if d else None

    def add(self, title: str, content: str) -> Note:
        n = Note.new(title, content)
        self._data[n.id] = asdict(n)
        _write_json(self.path, self._data)
        return n

    def update(self, id: str, title: str | None = None,
                content: str | None = None) -> bool:
        d = self._data.get(id)
        if not d:
            return False
        if title is not None:
            d["title"] = title
        if content is not None:
            d["content"] = content
        d["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write_json(self.path, self._data)
        return True

    def remove(self, id: str) -> bool:
        if id in self._data:
            del self._data[id]
            _write_json(self.path, self._data)
            return True
        return False

    def as_prompt_block(self) -> str:
        notes = self.list()
        if not notes:
            return ""
        lines = ["# Persistent rules (always follow these)"]
        for n in notes:
            content = n.content.strip().splitlines()[0] if n.content else ""
            lines.append(f"- [{n.title}] {content}")
        return "\n".join(lines)


# ---------- refinements (append-only event log of agent-curated changes)
class RefinementStore:
    """Append-only JSONL of agent-curated improvements.

    Each event has {kind, ts, ref, before, after}. The harness state
    itself is mtime-guarded; the event log is append-only so an
    out-of-process /refine command can never lose work.
    """
    def __init__(self, path: Path = REFINEMENTS_FILE):
        self.path = path

    def record(self, kind: str, ref: str, before: Any = None,
                after: Any = None, note: str = "") -> None:
        _ensure_dir()
        evt = {
            "ts": time.time(),
            "kind": kind,    # "memory.add" | "note.add" | "skill.add" | "system_prompt.update"
            "ref": ref,      # the key/id of what changed
            "before": before,
            "after": after,
            "note": note,
        }
        try:
            with open(self.path, "a", buffering=1) as f:
                f.write(json.dumps(evt, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def tail(self, n: int = 20) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out[-n:]


# ---------- system prompt composer
def build_system_prompt(harness) -> str:
    """Compose the system prompt from: base + memory + notes + skills.

    Returns a single string suitable to use as the system message.
    """
    base = getattr(harness, "system",
                   "You are Cozy, a voice assistant running fully offline. "
                   "Respond fast and short. When the user wants an action, "
                   "call exactly one tool with compact JSON. For plain chat, "
                   "answer briefly and warmly without tools.")
    parts = [base]
    # Memory (only if there are entries)
    m = MemoryStore()
    block = m.as_prompt_block()
    if block:
        parts.append(block)
    # Notes
    n = NotesStore()
    block = n.as_prompt_block()
    if block:
        parts.append(block)
    # Skills block
    from .skills import discover_skills, format_skills_for_prompt
    skills = discover_skills()
    if skills:
        parts.append(format_skills_for_prompt(skills))
    return "\n\n".join(parts)
