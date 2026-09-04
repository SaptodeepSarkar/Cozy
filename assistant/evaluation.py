"""Shared, dependency-light evaluation helpers for Cozy tool calling."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Parse tagged or bare balanced JSON without accepting a name-only guess."""
    clean = strip_thinking(text)
    candidates = [match.group(1) for match in TOOL_RE.finditer(clean)]
    depth = 0
    start = None
    for index, char in enumerate(clean):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(clean[start:index + 1])
                start = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            continue
        params = value.get("parameters", value.get("arguments", {}))
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return None
        if isinstance(params, dict):
            return {"name": value["name"], "parameters": params}
    return None


def score_probe(output: str, probe: dict[str, Any], valid_tools: set[str]) -> tuple[bool, str]:
    clean = strip_thinking(output)
    call = parse_tool_call(clean)
    if probe["kind"] == "chat":
        if call is not None:
            return False, "unexpected_tool"
        if not clean or clean in {"...", "<|im_end|>"}:
            return False, "empty_chat"
        return True, "ok"
    if call is None:
        return False, "invalid_or_missing_tool_call"
    if call["name"] not in valid_tools:
        return False, "tool_not_in_schema"
    if call["name"] != probe["expected_tool"]:
        return False, "wrong_tool"
    expected = probe.get("expected_parameters") or {}
    if any(call["parameters"].get(key) != value for key, value in expected.items()):
        return False, "wrong_parameters"
    return True, "ok"


def load_probes(path: Path) -> list[dict[str, Any]]:
    probes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [probe["id"] for probe in probes]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate probe ids in {path}")
    return probes


def chosen_message(probe: dict[str, Any]) -> dict[str, Any]:
    if probe["kind"] == "chat":
        return {"role": "assistant", "content": probe["chosen_text"]}
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "type": "function",
            "function": {
                "name": probe["expected_tool"],
                "arguments": json.dumps(probe.get("expected_parameters") or {}, separators=(",", ":")),
            },
        }],
    }
