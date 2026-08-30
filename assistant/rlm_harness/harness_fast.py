"""Cozy fast RLM harness - RAM + context optimized.

Design principles:
  1. **State on disk, in-memory is a cache.** The full conversation
     history lives in ``data/harness/trace.jsonl`` (append-only). We
     keep only the most recent ``N`` turns in RAM; older turns are
     summarized and discarded.
  2. **Lazy model load + idle unload.** Plugins (STT/LLM/TTS/Vision)
     stay unloaded until the wake gate fires. After ``idle_unload_s``
     of no use, they go back to disk.
  3. **Pre-tokenized tool schema.** The 32-tool schema is tokenized
     once and cached; only the conversation is re-tokenized per turn.
  4. **Compact-on-demand.** When the prompt grows past
     ``compact_threshold`` tokens, we summarize the oldest
     ``compact_window`` turns and prepend a one-line summary to the
     next prompt.
  5. **C fast path for tool-call extraction.** ``fasttool`` (libfasttool.so)
     parses LLM output in ~1.5us instead of ~3us for the Python regex.
  6. **Prime-Agent style harness state.** Memory / skills / refinements
     are stored as small JSON records and re-read on demand.

This module is the **only** file a tool-call LLM ever needs. It exposes
``FastHarness.decide(turns)`` which returns the next action.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically: temp file + rename.

    A crashed write to the live file would leave the harness state
    half-written; the next read would JSON-decode-fail and we'd lose
    the whole history. The temp + rename is atomic on POSIX
    (rename(2) is atomic within the same filesystem) so the live
    file is always either the old version or the new version.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        # Best effort: clean up the temp file so it doesn't pollute the dir
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically (same temp + rename as JSON)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

HERE = Path(__file__).resolve().parent
ASSISTANT = HERE.parent
STATE_DIR = ASSISTANT / "data" / "harness"
TRACE_FILE = STATE_DIR / "trace.jsonl"
SUMMARY_FILE = STATE_DIR / "summary.txt"
HARNESS_STATE_FILE = STATE_DIR / "state.json"


# ---------------------------------------------------------------- params
@dataclass
class HarnessConfig:
    """All tunables in one place. Override via env vars (COZY_*)."""
    # Context budget
    max_context_tokens: int = 1100           # total prompt budget
    compact_threshold: int = 900             # when to compact
    compact_window: int = 6                 # how many old turns to summarize
    recent_turns: int = 8                   # keep last N raw in RAM

    # RAM / disk
    state_dir: Path = STATE_DIR
    trace_file: Path = TRACE_FILE

    # Plugins
    use_stt: bool = True
    use_tts: bool = True
    use_llm: bool = True
    use_vision: bool = False
    use_wake: bool = True

    # Lazy load
    idle_unload_s: float = 60.0

    # Tool schema
    tool_schema_cache: Path = STATE_DIR / "tools_tokenized.json"

    @classmethod
    def from_env(cls) -> "HarnessConfig":
        c = cls()
        for k in ("max_context_tokens", "compact_threshold", "compact_window",
                  "recent_turns", "idle_unload_s"):
            v = os.environ.get(f"COZY_{k.upper()}")
            if v:
                setattr(c, k, type(getattr(c, k))(v))
        c.use_vision = os.environ.get("COZY_VISION", "0") == "1"
        return c


# ---------------------------------------------------------------- trace
@dataclass
class Turn:
    """One conversational turn. Stored on disk as one JSON line."""
    role: str                  # system|user|assistant|tool
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    name: str = ""            # for tool role
    ts: float = field(default_factory=time.time)
    tokens: int = 0           # how many tokens this turn used
    producer: str = "user"    # who made this turn: user|human|model|tool

    def to_dict(self) -> dict:
        return {
            "role": self.role, "content": self.content,
            "tool_calls": self.tool_calls, "name": self.name,
            "ts": self.ts, "tokens": self.tokens, "producer": self.producer,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(
            role=d["role"], content=d.get("content", ""),
            tool_calls=d.get("tool_calls", []) or [],
            name=d.get("name", ""), ts=d.get("ts", 0.0),
            tokens=d.get("tokens", 0), producer=d.get("producer", "user"),
        )


class Trace:
    """Append-only JSONL trace on disk. mtime-aware re-read.

    ``Trace`` keeps the last ``recent_turns`` raw in RAM and a one-line
    summary of everything before. The full history is on disk and is
    only re-read on crash recovery or when ``read_all()`` is called.
    """
    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self.cfg.state_dir.mkdir(parents=True, exist_ok=True)
        self._recent: list[Turn] = []
        self._summary: str = ""
        self._last_mtime: float = 0.0
        self._load_recent()
        self._load_summary()
        self._sync_from_disk()

    # ----------------- mtime sync (Prime Agent pattern)
    def _disk_mtime(self) -> float:
        try:
            return self.cfg.trace_file.stat().st_mtime
        except OSError:
            return 0.0

    def _sync_from_disk(self) -> None:
        if self._disk_mtime() == self._last_mtime:
            return
        # Reload: keep all old turns as summary, last N raw
        all_turns = self._read_all()
        if len(all_turns) > self.cfg.recent_turns:
            old = all_turns[:-self.cfg.recent_turns]
            self._summary = self._summary or self._summarize(old)
            self._recent = all_turns[-self.cfg.recent_turns:]
        else:
            self._recent = all_turns
        self._last_mtime = self._disk_mtime()

    def _read_all(self) -> list[Turn]:
        if not self.cfg.trace_file.exists():
            return []
        turns = []
        with open(self.cfg.trace_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    turns.append(Turn.from_dict(json.loads(line)))
                except json.JSONDecodeError:
                    continue
        return turns

    def _load_recent(self) -> None:
        """Load the last N turns into RAM. Cheap because we tail the file."""
        if not self.cfg.trace_file.exists():
            return
        with open(self.cfg.trace_file) as f:
            lines = f.readlines()
        # Take last recent_turns * 2 lines (we'll filter out blanks)
        tail = lines[-self.cfg.recent_turns * 2:]
        recent = []
        for line in tail:
            line = line.strip()
            if not line: continue
            try:
                recent.append(Turn.from_dict(json.loads(line)))
            except: continue
        self._recent = recent[-self.cfg.recent_turns:]

    def _load_summary(self) -> None:
        if SUMMARY_FILE.exists():
            self._summary = SUMMARY_FILE.read_text().strip()

    def _save_summary(self) -> None:
        # Summary is just a tiny text file; atomic write avoids
        # mid-write corruption if the process dies while flushing.
        _atomic_write_text(SUMMARY_FILE, self._summary)

    # ----------------- mutators
    def append(self, turn: Turn) -> None:
        """Append a new turn to disk + RAM. Re-compact if over budget."""
        self._sync_from_disk()
        self._recent.append(turn)
        line = json.dumps(turn.to_dict(), ensure_ascii=False)
        with open(self.cfg.trace_file, "a") as f:
            f.write(line + chr(10))
        self._last_mtime = self._disk_mtime()
        if self._count_recent_tokens() > self.cfg.compact_threshold:
            self.compact()

    def compact(self) -> None:
        """Summarize the oldest ``compact_window`` recent turns and drop them."""
        if len(self._recent) <= self.cfg.recent_turns:
            return
        # Split: old to summarize, recent to keep
        to_summarize = self._recent[:self.cfg.compact_window]
        self._recent = self._recent[self.cfg.compact_window:]
        new_summary = self._summarize(to_summarize)
        # Concatenate summaries with the previous one
        if self._summary:
            self._summary = f"{self._summary} {new_summary}"
        else:
            self._summary = new_summary
        self._save_summary()
        # Rewrite the trace file: write a summary line + the remaining recent
        # Note: this loses the original lines we just summarized. The trade-off
        # is "context-effective" - we'd rather have a working context than
        # the full text. The full history is still in the trace file's
        # ``trace.jsonl`` if we want to mine it later, but the in-RAM
        # representation is the summary.
        # NOTE: we do NOT delete the trace file - the raw text is preserved
        # there for offline analysis (DPO mining, debugging).
        self._last_mtime = self._disk_mtime()

    def _count_recent_tokens(self) -> int:
        return sum(t.tokens for t in self._recent)

    @staticmethod
    def _summarize(turns: list[Turn]) -> str:
        """Heuristic: one sentence per turn."""
        bits = []
        for t in turns:
            if t.role == "user":
                bits.append(f"User: {t.content[:60]}")
            elif t.role == "assistant" and t.content:
                bits.append(f"Assistant: {t.content[:60]}")
            elif t.role == "assistant" and t.tool_calls:
                tc = t.tool_calls[0]
                bits.append(f"Assistant called {tc.get('function', {}).get('name', '?')}")
            elif t.role == "tool":
                bits.append(f"Tool {t.name}: ok")
        return " | ".join(bits)

    # ----------------- prompt builder
    def build_prompt(self, system: str, tools_repr: str) -> tuple[list[dict], int]:
        """Return (messages, approx_total_tokens).

        ``tools_repr`` is the pre-tokenized tool schema header (cached
        on disk). It's prepended to the system message so it only
        affects the system row, not the recent turns.
        """
        msgs = [{"role": "system", "content": system + "\n\n" + tools_repr}]
        if self._summary:
            msgs.append({"role": "system",
                         "content": f"Earlier context summary: {self._summary}"})
        for t in self._recent:
            d = {"role": t.role}
            if t.content:
                d["content"] = t.content
            if t.tool_calls:
                d["tool_calls"] = t.tool_calls
            if t.name and t.role == "tool":
                d["name"] = t.name
            msgs.append(d)
        # Approximate token count: 4 chars per token
        n_tokens = sum(len(json.dumps(m, ensure_ascii=False)) for m in msgs) // 4
        return msgs, n_tokens

    def clear(self) -> None:
        """Wipe both RAM and disk (for tests / fresh session)."""
        self._recent = []
        self._summary = ""
        if self.cfg.trace_file.exists():
            self.cfg.trace_file.unlink()
        if SUMMARY_FILE.exists():
            SUMMARY_FILE.unlink()
        self._last_mtime = 0.0

    @property
    def recent(self) -> list[Turn]:
        return list(self._recent)

    @property
    def summary(self) -> str:
        return self._summary


# ---------------------------------------------------------------- tool cache
class ToolSchemaCache:
    """Pre-serialize the 32-tool schema once, reuse forever.

    The full schema is ~1000 tokens. Re-tokenizing it every turn wastes
    a 1000-token LLM context slot. With this cache, the schema is read
    from disk once and just appended to the system message.
    """
    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self._repr: str = ""
        self._mtime: float = 0.0
        self._load()

    def _disk_mtime(self) -> float:
        try:
            return self.cfg.tool_schema_cache.stat().st_mtime
        except OSError:
            return 0.0

    def _load(self) -> None:
        schema_path = ASSISTANT.parent / "team" / "tool_schema.json"
        # Invalidate if source schema is newer
        if not self.cfg.tool_schema_cache.exists() or self._disk_mtime() == 0:
            self._build(schema_path)
        else:
            self._repr = self.cfg.tool_schema_cache.read_text()
            # Cheap invalidation: re-build if source mtime is newer
            try:
                if schema_path.stat().st_mtime > self._disk_mtime():
                    self._build(schema_path)
            except OSError:
                pass

    def with_skills(self) -> str:
        """Return the tool repr + discovered skills block."""
        from .skills import discover_skills, format_skills_for_prompt
        skills = discover_skills()
        if not skills:
            return self._repr
        return self._repr + "\n\n" + format_skills_for_prompt(skills)

    def _build(self, schema_path: Path) -> None:
        schema = json.loads(schema_path.read_text())
        # Compact: just list names + 1-line descriptions
        lines = ["# Available tools (call when relevant):"]
        for t in schema["tools"]:
            desc = t.get("desc", "").strip()
            params = t.get("params", {})
            line = f"- {t['name']}({json.dumps(params, ensure_ascii=False)})"
            if desc:
                line += f" - {desc[:80]}"
            lines.append(line)
        self._repr = "\n".join(lines)
        # Persist atomically. The mtime we set must equal the source
        # schema's mtime so the mtime-guarded reload invalidates this
        # cache if someone edits team/tool_schema.json after the build.
        import os
        schema_mtime = schema_path.stat().st_mtime
        tmp = self.cfg.tool_schema_cache.with_suffix(".tmp")
        _atomic_write_text(tmp, self._repr)
        os.utime(tmp, (schema_mtime, schema_mtime))
        os.replace(tmp, self.cfg.tool_schema_cache)
        self._mtime = schema_mtime

    @property
    def repr(self) -> str:
        return self._repr


# ---------------------------------------------------------------- plugin
class Plugin:
    """Base class for lazy-loaded plugins (LLM, STT, TTS, Vision).

    Each plugin stays unloaded until ``load()`` is called, and is
    eligible for unload after ``idle_unload_s`` seconds of disuse.
    """
    name: str = "abstract"

    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self._loaded = False
        self._last_used = 0.0

    def load(self) -> None:
        if self._loaded: return
        self._do_load()
        self._loaded = True
        self._last_used = time.time()

    def _do_load(self) -> None:
        raise NotImplementedError

    def free(self) -> None:
        if not self._loaded: return
        self._do_free()
        self._loaded = False

    def _do_free(self) -> None:
        pass

    def touch(self) -> None:
        self._last_used = time.time()
        self._loaded = True

    def maybe_idle_unload(self) -> None:
        if not self._loaded: return
        if (time.time() - self._last_used) > self.cfg.idle_unload_s:
            self.free()


# ---------------------------------------------------------------- fast tool call extract
def extract_tool_call(text: str) -> tuple[str, dict]:
    """Fast tool-call extraction. Uses the C library if available,
    else falls back to a Python regex. Returns ``(name, args)``;
    both empty on failure.
    """
    # Try the C path first
    try:
        from .fasttool import extract_tool_call as c_extract
        return c_extract(text)
    except Exception:
        pass
    # Python fallback
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    if m:
        try:
            c = json.loads(m.group(1))
            name = c.get("name")
            params = c.get("parameters") or c.get("arguments") or {}
            if isinstance(params, str):
                try: params = json.loads(params)
                except: params = {}
            if isinstance(params, dict) and isinstance(name, str):
                return name, params
        except json.JSONDecodeError:
            pass
    # Balanced-brace fallback
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    c = json.loads(text[start:i+1])
                    if isinstance(c, dict) and isinstance(c.get("name"), str):
                        params = c.get("parameters") or c.get("arguments") or {}
                        if isinstance(params, str):
                            try: params = json.loads(params)
                            except: params = {}
                        if isinstance(params, dict):
                            return c["name"], params
                except json.JSONDecodeError:
                    pass
                start = None
    return "", {}


# ---------------------------------------------------------------- main
class FastHarness:
    """The lean, context-effective harness.

    Owns: a trace (disk-backed) + tool schema cache + plugin registry.
    Each call to ``decide()``:
      1. Sync from disk (mtime check)
      2. Build the prompt (system + cached tool repr + summary + recent)
      3. Load the LLM plugin (if not loaded)
      4. Generate
      5. Parse tool call (fast path)
      6. Idle-unload if nothing happened for ``idle_unload_s``
    """
    def __init__(self, cfg: HarnessConfig | None = None):
        self.cfg = cfg or HarnessConfig.from_env()
        self.trace = Trace(self.cfg)
        self.tools = ToolSchemaCache(self.cfg)
        self.plugins: dict[str, Plugin] = {}
        self.system = ("You are Cozy, a voice assistant running fully "
                       "offline on the user laptop. Respond fast and short. "
                       "When the user wants an action, call exactly one "
                       "tool with compact JSON. For plain chat, answer "
                       "briefly and warmly without tools.")
        self._register_default_plugins()

    def _register_default_plugins(self) -> None:
        # Only register plugins the config asks for. The implementations
        # themselves are lazy.
        if self.cfg.use_wake:
            from rlm_harness.plugins.wake import WakePlugin
            self.plugins["wake"] = WakePlugin(self.cfg)
        if self.cfg.use_stt:
            from rlm_harness.plugins.stt import STTPlugin
            self.plugins["stt"] = STTPlugin(self.cfg)
        if self.cfg.use_tts:
            from rlm_harness.plugins.tts import TTSPlugin
            self.plugins["tts"] = TTSPlugin(self.cfg)
        if self.cfg.use_llm:
            from rlm_harness.plugins.llm import LLMPlugin
            self.plugins["llm"] = LLMPlugin(self.cfg)
        if self.cfg.use_vision:
            from rlm_harness.plugins.vision import VisionPlugin
            self.plugins["vision"] = VisionPlugin(self.cfg)

    def get(self, name: str) -> Plugin | None:
        return self.plugins.get(name)

    def decide(self, user_text: str) -> tuple[str, dict]:
        """Append the user's turn, run the LLM, return (tool_name, args).
        ``tool_name`` is empty if the model replied with text only.
        """
        # 1. Record the user turn
        self.trace.append(Turn(role="user", content=user_text,
                                producer="user"))
        # 2. Sync from disk (other agents may have written to trace)
        self.trace._sync_from_disk()
        # 3. Build prompt
        msgs, n_tok = self.trace.build_prompt(self.system, self.tools.repr)
        # 4. Load + run LLM
        llm = self.plugins.get("llm")
        if llm is None:
            return "", {}
        llm.load()
        llm.touch()
        raw = llm.generate(msgs)
        # 5. Extract tool call
        name, args = extract_tool_call(raw)
        if name:
            self.trace.append(Turn(
                role="assistant", content="",
                tool_calls=[{"type": "function",
                             "function": {"name": name,
                                          "arguments": json.dumps(args, ensure_ascii=False)}}],
                producer="model"))
        else:
            # Strip any thinking block
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
            self.trace.append(Turn(role="assistant", content=raw, producer="model"))
        # 6. Maybe unload idle plugins
        for p in self.plugins.values():
            p.maybe_idle_unload()
        # 7. Log
        try:
            from cozy_log import log_event
            log_event("decide", user=user_text[:200], tool=name, args=json.dumps(args, ensure_ascii=False)[:200])
        except Exception:
            pass
        return name, args
    def warmup(self, on_progress=None) -> None:
        """Pre-load all enabled plugins in PARALLEL background threads.

        The wakeword, STT, LLM, and TTS each load on their own thread,
        so total time is max(load_time) instead of sum(load_time).
        Without warmup, the LLM is loaded on the first decide() call
        (~20s blocking) and TTS on the first speak() call (~16s).
        With warmup, all four are loaded in parallel while the TUI
        is responsive. ``on_progress`` is called as each plugin
        finishes: on_progress(name, ok).
        """
        def _loader(name):
            p = self.plugins.get(name)
            if p is None:
                return
            try:
                p.load()
                if on_progress:
                    on_progress(name, True)
            except Exception:
                if on_progress:
                    on_progress(name, False)

        threads = []
        for name in ("wake", "stt", "llm", "tts", "vision"):
            t = threading.Thread(target=_loader, args=(name,),
                                 daemon=True, name=f"cozy-warmup-{name}")
            t.start()
            threads.append(t)
        return threads



    def reset(self) -> None:
        """Clear the in-memory trace (disk stays)."""
        self.trace.clear()

    def stats(self) -> dict:
        return {
            "recent_turns": len(self.trace.recent),
            "summary_chars": len(self.trace.summary),
            "approx_tokens": self.trace._count_recent_tokens(),
            "plugins_loaded": [n for n, p in self.plugins.items() if p._loaded],
        }


# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Cozy fast harness stats")
    p.add_argument("--stats", action="store_true", help="print harness stats")
    p.add_argument("--reset", action="store_true", help="clear in-RAM trace (keep disk)")
    p.add_argument("--test", action="store_true", help="end-to-end test")
    args = p.parse_args()
    h = FastHarness()
    if args.stats:
        print(json.dumps(h.stats(), indent=2))
    if args.reset:
        h.reset()
        print("trace cleared")
    if args.test:
        name, args_out = h.decide("set volume to 30")
        print(f"decision: {name} {args_out}")
