"""Trace data model for the Cozy RLM harness.

A ``Trace`` is one full task rollout: system prompt + a list of turns.
Turns follow the same ``{role, content, tool_calls, ...}`` shape Qwen3's
chat template expects, and the trace serialises to the exact same JSONL
schema ``make_dataset.py`` already produces::

    {"messages": [...], "tools": [...]}

That way anything you collect with this harness is immediately usable as
SFT data - no reformatting.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Turn:
    role: Role
    content: str = ""
    # tool_calls in the OpenAI-ish shape Qwen3's template expects
    tool_calls: list[dict] | None = None
    # tool message metadata
    name: str | None = None  # tool name (for role=tool)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"role": self.role.value}
        if self.content:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc.get("arguments", {}),
                                                ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]
        if self.name and self.role == Role.TOOL:
            d["name"] = self.name
        return d


@dataclass
class Trace:
    task_id: str
    task_text: str
    tools_schema: list[dict] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    # who/what produced each assistant turn: "human" or "model" or "rule"
    producers: list[str] = field(default_factory=list)
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)

    # -- mutators -----------------------------------------------------
    def add_user(self, text: str) -> None:
        self.turns.append(Turn(Role.USER, text))

    def add_assistant_text(self, text: str, producer: str = "human") -> None:
        self.turns.append(Turn(Role.ASSISTANT, text))
        self.producers.append(producer)

    def add_assistant_tool_call(self, name: str, arguments: dict,
                                producer: str = "human") -> None:
        self.turns.append(Turn(
            Role.ASSISTANT, "",
            tool_calls=[{"name": name, "arguments": arguments}],
        ))
        self.producers.append(producer)

    def add_tool_result(self, name: str, result: str) -> None:
        self.turns.append(Turn(Role.TOOL, result, name=name))

    def last_user(self) -> str:
        for t in reversed(self.turns):
            if t.role == Role.USER:
                return t.content
        return ""

    # -- IO -----------------------------------------------------------
    def to_sft_record(self) -> dict:
        """Render to the same schema as assistant/data/sft_train.jsonl."""
        msgs: list[dict] = []
        # Cozy's SFT data starts with a system message describing the
        # assistant's role. Carry through if one was set, otherwise inject
        # the standard one.
        if not self.turns or self.turns[0].role != Role.SYSTEM:
            msgs.append({
                "role": "system",
                "content": (
                    "You are Cozy, a voice assistant running fully offline "
                    "on the user's laptop. Respond fast and short. When the "
                    "user wants an action, call exactly one tool with "
                    "compact JSON. For plain chat, answer briefly and warmly "
                    "without tools."
                ),
            })
        for t in self.turns:
            msgs.append(t.to_dict())
        return {
            "messages": msgs,
            "tools": self.tools_schema,
        }

    def to_jsonl_dict(self) -> dict:
        d = {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "task_text": self.task_text,
            "created_at": self.created_at,
            "producers": self.producers,
            "meta": self.meta,
            "turns": [t.to_dict() for t in self.turns],
        }
        return d

    @classmethod
    def from_jsonl_dict(cls, d: dict) -> "Trace":
        t = cls(
            task_id=d.get("task_id", ""),
            task_text=d.get("task_text", ""),
            tools_schema=d.get("tools_schema", []),
            meta=d.get("meta", {}),
            producers=d.get("producers", []),
            trace_id=d.get("trace_id", uuid.uuid4().hex[:12]),
            created_at=d.get("created_at", time.time()),
        )
        for raw in d.get("turns", []):
            tcs = raw.get("tool_calls")
            parsed_tcs = None
            if tcs:
                parsed_tcs = []
                for tc in tcs:
                    fn = tc.get("function", tc)
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    parsed_tcs.append({"name": fn["name"], "arguments": args})
            t.turns.append(Turn(
                role=Role(raw["role"]),
                content=raw.get("content", ""),
                tool_calls=parsed_tcs,
                name=raw.get("name"),
            ))
        return t


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
