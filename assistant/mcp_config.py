"""Load OpenCode-compatible MCP server configuration for Cozy.

Accepted files are ``cozy.jsonc`` and ``opencode.jsonc``. The loader is
deliberately transport-agnostic: it normalizes server definitions now, while
the MCP client can decide how to launch stdio, HTTP, or SSE servers.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

_COMMENT = re.compile(r"(^|\s)//.*?$|/\*.*?\*/", re.M | re.S)
_TRAILING = re.compile(r",\s*([}\]])")


def parse_jsonc(text: str) -> dict:
    """Parse JSON with comments and trailing commas, without eval."""
    clean = _COMMENT.sub(lambda m: m.group(1) if m.group(1) else "", text)
    clean = _TRAILING.sub(r"\1", clean)
    return json.loads(clean)


def _expand(value):
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


@dataclass
class MCPServer:
    name: str
    transport: str
    command: list[str] = field(default_factory=list)
    url: str = ""
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


def load_servers(path: Path | str | None = None) -> list[MCPServer]:
    """Load the nearest Cozy/OpenCode config, or an explicitly supplied file."""
    if path is None:
        candidates = [Path.cwd() / "cozy.jsonc", Path.cwd() / "opencode.jsonc",
                      Path.home() / ".config" / "opencode" / "opencode.jsonc"]
        path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return []
    source = _expand(parse_jsonc(Path(path).read_text(encoding="utf-8")))
    raw = source.get("mcp", source.get("mcpServers", {}))
    if isinstance(raw, list):
        raw = {str(item.get("name")): item for item in raw}
    result = []
    for name, item in (raw or {}).items():
        if not isinstance(item, dict) or item.get("enabled", True) is False:
            continue
        command = item.get("command", [])
        if isinstance(command, str):
            command = [command]
        args = item.get("args", [])
        if isinstance(args, str):
            args = [args]
        command = command + list(args)
        transport = item.get("type", item.get("transport", "http" if item.get("url") else "stdio"))
        if transport == "local":
            transport = "stdio"
        elif transport == "remote":
            transport = "http"
        result.append(MCPServer(name=str(name), transport=str(transport),
                                command=list(command), url=item.get("url", ""),
                                env={str(k): str(v) for k, v in item.get("env", {}).items()}))
    return result
