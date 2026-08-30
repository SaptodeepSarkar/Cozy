"""Cozy RLM (recursion) - spawn child agents with scoped tools.

The simplest possible version of Prime Agent's RLM:

  await rlm_delegate(task, parent_harness, allow=None) -> str

Creates a child FastHarness with:
  - The same plugins (LLM, TTS, etc.) - the parent already has them loaded,
    the child shares via the same model objects.
  - A scoped tool schema: only the tools whose name is in ``allow`` (or all
    32 if ``allow`` is None).
  - Its own trace file (depth in filename), so child turns are recoverable.
  - Depth tracking: depth = parent_depth + 1. Hard limit at 3.

Returns when the child finishes (synchronous). Result is the last
assistant turn's text content. The child writes a
``{kind: "spawn", depth, task, parent_id}`` and a
``{kind: "return", child_id, result}`` entry to its own trace.jsonl.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from .harness_fast import FastHarness, HarnessConfig, Turn, ASSISTANT
from . import state as state_mod


MAX_DEPTH = 3
_SPAWN_COUNTER = 0  # monotonic counter for child ids


def _next_child_id() -> str:
    global _SPAWN_COUNTER
    _SPAWN_COUNTER += 1
    return f"c{_SPAWN_COUNTER}_{uuid.uuid4().hex[:6]}"


def rlm_delegate(task: str, parent: FastHarness,
                  allow: list[str] | None = None,
                  depth: int = 1) -> str:
    """Spawn a child agent to handle ``task``.

    ``allow`` filters the tool schema. ``depth`` is the child's depth
    (parent is at depth 0; children at depth 1; grandchildren at depth 2).
    Hard cap: MAX_DEPTH. Anything beyond raises RuntimeError.
    """
    if depth > MAX_DEPTH:
        raise RuntimeError(
            f"RLM depth {depth} exceeds MAX_DEPTH={MAX_DEPTH}")

    child_id = _next_child_id()

    # Build a child config that points to its own state dir.
    child_state = parent.cfg.state_dir.parent / "rlm_children" / child_id
    child_cfg = HarnessConfig(
        state_dir=child_state / "harness",
        trace_file=child_state / "harness" / "trace.jsonl",
        tool_schema_cache=child_state / "harness" / "tools.json",
        # Inherit flags from the parent
        use_wake=parent.cfg.use_wake,
        use_stt=parent.cfg.use_stt,
        use_tts=parent.cfg.use_tts,
        use_llm=parent.cfg.use_llm,
        use_vision=parent.cfg.use_vision,
        idle_unload_s=parent.cfg.idle_unload_s,
    )

    # Create the child harness. We share the parent's plugins so we
    # don't double-load the LLM / TTS models.
    child = FastHarness(child_cfg)
    child.plugins = parent.plugins  # share plugins, including the LLM

    # Scope the tool schema. If ``allow`` is given, write a filtered
    # cache so the child only sees those tools.
    if allow is not None:
        # Write a filtered tool cache file
        schema_path = ASSISTANT.parent / "team" / "tool_schema.json"
        full = json.loads(schema_path.read_text())
        full["tools"] = [t for t in full["tools"] if t["name"] in allow]
        # The ToolSchemaCache will build a fresh _repr when it loads.
        # We just write the filtered schema to a temp file and point
        # the child at it.
        scoped_schema_path = child_state / "tool_schema.json"
        scoped_schema_path.parent.mkdir(parents=True, exist_ok=True)
        scoped_schema_path.write_text(json.dumps(full, ensure_ascii=False, indent=2))
        # Force the child to rebuild its repr from the filtered schema.
        from .harness_fast import ToolSchemaCache
        child_cfg.tool_schema_cache = child_state / "harness" / "tools.json"
        child.tools = _ScopedToolSchemaCache(child_cfg, scoped_schema_path)

    # Log spawn
    child.trace.append(Turn(
        role="system", content=f"spawned for task: {task}",
        producer="rlm-parent", tokens=10))
    child.trace.append(Turn(role="user", content=task, producer="user"))

    # Also log the spawn event to the PARENT's trace
    parent.trace._sync_from_disk()
    parent.trace.append(Turn(
        role="system",
        content=f"rlm: spawned child {child_id} depth={depth} for task: {task[:100]}",
        producer="rlm-spawn", tokens=15))
    from .cozy_log import log_event
    log_event("rlm.spawn", child_id=child_id, depth=depth, task=task[:200],
              allow=allow)

    # Run the child to completion
    try:
        child.decide(task)  # the user's task is the input
        # The child will append a tool result + assistant reply if the
        # tool fired. If it didn't fire, we need the assistant text.
        last_text = ""
        for t in reversed(child.trace.recent):
            if t.role == "assistant" and t.content:
                last_text = t.content
                break
        if not last_text:
            last_text = f"[child {child_id} produced no text reply]"

        # Record the return
        child.trace.append(Turn(
            role="system", content=f"return to parent",
            producer="rlm-return", tokens=5))
        log_event("rlm.return", child_id=child_id, result_len=len(last_text))
        parent.trace.append(Turn(
            role="tool",
            name=f"rlm.delegate.{child_id}",
            content=last_text[:4000],
            producer="rlm-return"))
        return last_text
    except Exception as exc:
        log_event("rlm.error", child_id=child_id, error=str(exc))
        parent.trace.append(Turn(
            role="tool",
            name=f"rlm.delegate.{child_id}",
            content=f"error: {exc}",
            producer="rlm-error"))
        return f"[child {child_id} error: {exc}]"


class _ScopedToolSchemaCache:
    """Like ToolSchemaCache but reads from a specific schema file.

    The child harness uses this when the parent filtered the schema.
    """
    def __init__(self, cfg, schema_path: Path):
        self.cfg = cfg
        self._schema_path = schema_path
        self._repr = ""
        self._mtime = 0.0
        self._load()

    def _disk_mtime(self) -> float:
        try:
            return self._schema_path.stat().st_mtime
        except OSError:
            return 0.0

    def _load(self) -> None:
        try:
            schema = json.loads(self._schema_path.read_text())
        except OSError:
            return
        lines = ["# Available tools (call when relevant, filtered for this child):"]
        for t in schema["tools"]:
            desc = t.get("desc", "").strip()
            params = t.get("params", {})
            line = f"- {t['name']}({json.dumps(params, ensure_ascii=False)})"
            if desc:
                line += f" - {desc[:80]}"
            lines.append(line)
        self._repr = "\n".join(lines)
        self._mtime = self._disk_mtime()

    @property
    def repr(self) -> str:
        return self._repr


# Register delegate as a tool the LLM can call. The LLM emits
# <tool_call name="rlm.delegate" args="..."> and the runtime routes
# it to rlm_delegate() with the appropriate allow list.
DELEGATE_SCHEMA = {
    "name": "rlm.delegate",
    "params": {
        "task": "string - the sub-task to give the child agent",
        "allow": "list of strings - tool names the child can use (optional)",
        "max_depth": "int - max recursion depth (default 1)",
    },
    "desc": "Spawn a child agent to handle a sub-task. Use for multi-step commands "
             "like 'set a reminder and tell my wife' that need multiple tool calls.",
}


def is_delegate_call(name: str) -> bool:
    return name == "rlm.delegate"


def execute_delegate(args: dict, parent_harness) -> str:
    """Runtime hook: handle an LLM-emitted delegate() call."""
    task = args.get("task", "")
    allow = args.get("allow")
    if isinstance(allow, str):
        allow = [a.strip() for a in allow.split(",") if a.strip()]
    if not allow:
        allow = None
    return rlm_delegate(task, parent_harness, allow=allow, depth=1)
