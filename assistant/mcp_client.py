"""Small dependency-free MCP hub for OpenCode/cozy.jsonc servers."""
from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.request

from mcp_config import MCPServer, load_servers


class MCPHub:
    def __init__(self):
        self.servers: dict[str, MCPServer] = {s.name: s for s in load_servers()}
        self.processes: dict[str, subprocess.Popen] = {}
        self.ids = 0
        self.tools: list[dict] = []
        self._lock = threading.Lock()

    def start(self) -> list[dict]:
        self.tools = []
        for server in self.servers.values():
            try:
                self._request(server, "initialize", {"protocolVersion": "2025-03-26",
                    "capabilities": {}, "clientInfo": {"name": "cozy", "version": "3.0"}})
                self._request(server, "notifications/initialized", {})
                result = self._request(server, "tools/list", {})
                for tool in result.get("tools", []) if isinstance(result, dict) else []:
                    item = dict(tool)
                    item["name"] = f"mcp.{server.name}.{tool.get('name', '')}"
                    item["server"] = server.name
                    item["remote_name"] = tool.get("name", "")
                    self.tools.append(item)
            except Exception:
                self._close_server(server.name)
        return self.tools

    def call(self, name: str, arguments: dict) -> str:
        parts = name.split(".", 2)
        if len(parts) != 3 or parts[0] != "mcp":
            raise ValueError("MCP tool must be mcp.<server>.<tool>")
        server_name, remote = parts[1], parts[2]
        server = self.servers.get(server_name)
        if server is None:
            raise ValueError(f"MCP server is not configured: {server_name}")
        result = self._request(server, "tools/call", {"name": remote, "arguments": arguments})
        content = result.get("content", []) if isinstance(result, dict) else []
        return "\n".join(str(x.get("text", "")) for x in content if isinstance(x, dict)) or json.dumps(result)

    def _request(self, server: MCPServer, method: str, params: dict):
        with self._lock:
            self.ids += 1
            payload = {"jsonrpc": "2.0", "id": self.ids, "method": method, "params": params}
            if server.transport in {"http", "sse", "streamable-http"}:
                req = urllib.request.Request(server.url, data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
                with urllib.request.urlopen(req, timeout=60) as response:
                    body = response.read().decode("utf-8", "replace")
            else:
                proc = self._process(server)
                proc.stdin.write(json.dumps(payload) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError(f"MCP server {server.name} closed stdout")
                body = line
            if "data:" in body:
                body = next((x[5:].strip() for x in body.splitlines()
                             if x.startswith("data:")), "{}")
            value = json.loads(body or "{}")
            if "error" in value:
                raise RuntimeError(value["error"].get("message", value["error"]))
            return value.get("result", value)

    def _process(self, server: MCPServer):
        proc = self.processes.get(server.name)
        if proc is None or proc.poll() is not None:
            if not server.command:
                raise ValueError(f"stdio MCP server {server.name} has no command")
            proc = subprocess.Popen(server.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                    env={**__import__('os').environ, **server.env})
            self.processes[server.name] = proc
        return proc

    def _close_server(self, name: str) -> None:
        proc = self.processes.pop(name, None)
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def close(self) -> None:
        for name in list(self.processes):
            self._close_server(name)


_hub: MCPHub | None = None


def start() -> list[dict]:
    global _hub
    _hub = MCPHub()
    return _hub.start()


def call(name: str, arguments: dict) -> str:
    if _hub is None:
        raise RuntimeError("MCP hub is not running")
    return _hub.call(name, arguments)


def stop() -> None:
    if _hub is not None:
        _hub.close()
