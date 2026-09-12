"""Own a local FoxMCP server and call its Streamable HTTP MCP endpoint."""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


class FoxMCPClient:
    def __init__(self) -> None:
        root = os.environ.get("COZY_FOXMCP_ROOT")
        default_root = Path(__file__).resolve().parent.parent / ".cozy" / "foxmcp"
        self.root = Path(root) if root else default_root
        self.base_url = os.environ.get("COZY_FOXMCP_URL", "http://127.0.0.1:3000/mcp")
        self.ws_port = os.environ.get("COZY_FOXMCP_WS_PORT", "8765")
        self.mcp_port = os.environ.get("COZY_FOXMCP_MCP_PORT", "3000")
        self.process: subprocess.Popen | None = None
        self.session_id = ""
        self.tools: list[dict] = []

    def start(self) -> list[dict]:
        if not (self.root / "venv/bin/python").exists():
            raise FileNotFoundError(
                f"FoxMCP is not installed at {self.root}. Run bash install_foxmcp.sh")
        if self.root is not None:
            command = [str(self.root / "venv/bin/python"), str(self.root / "server/server.py"),
                       "--host", "127.0.0.1", "--port", self.ws_port,
                       "--mcp-port", self.mcp_port]
            if not Path(command[0]).exists() or not Path(command[1]).exists():
                raise FileNotFoundError(f"FoxMCP installation is incomplete: {self.root}")
            self.process = subprocess.Popen(command, cwd=self.root,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._wait_for_server()
        self._initialize()
        result = self._request("tools/list", {})
        self.tools = (result.get("tools") if isinstance(result, dict) else []) or []
        return self.tools

    def _wait_for_server(self) -> None:
        deadline = time.time() + float(os.environ.get("COZY_FOXMCP_START_TIMEOUT", "15"))
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(self.base_url, timeout=1):
                    return
            except Exception:
                time.sleep(0.25)
        raise TimeoutError(f"FoxMCP did not start at {self.base_url}")

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        req = urllib.request.Request(self.base_url, data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response:
            sid = response.headers.get("Mcp-Session-Id")
            if sid:
                self.session_id = sid
            body = response.read().decode("utf-8", "replace")
        if body.lstrip().startswith("data:"):
            body = next((line[5:].strip() for line in body.splitlines() if line.startswith("data:")), "{}")
        value = json.loads(body or "{}")
        if "error" in value:
            raise RuntimeError(value["error"].get("message", value["error"]))
        return value.get("result", value)

    def _initialize(self) -> None:
        self._post({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                               "clientInfo": {"name": "cozy", "version": "3.0"}}})
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    def _request(self, method: str, params: dict) -> dict:
        return self._post({"jsonrpc": "2.0", "id": int(time.time() * 1000),
                           "method": method, "params": params})

    def call(self, name: str, arguments: dict) -> str:
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        content = result.get("content", []) if isinstance(result, dict) else []
        parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "\n".join(part for part in parts if part) or json.dumps(result, ensure_ascii=False)

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None


_client: FoxMCPClient | None = None


def start() -> list[dict]:
    global _client
    _client = FoxMCPClient()
    return _client.start()


def call(name: str, arguments: dict) -> str:
    if _client is None:
        raise RuntimeError("FoxMCP is not running")
    return _client.call(name, arguments)


def stop() -> None:
    if _client is not None:
        _client.stop()
