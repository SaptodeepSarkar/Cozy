"""Own a local FoxMCP server and call its Streamable HTTP MCP endpoint."""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
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
        self.log_file = None
        self.session_id = ""
        self.tools: list[dict] = []

    def start(self) -> list[dict]:
        if not (self.root / "venv/bin/python").exists():
            raise FileNotFoundError(
                f"FoxMCP is not installed at {self.root}. Run bash install_foxmcp.sh")
        # Cozy may have been restarted while the previous bridge survived.
        # Reuse a healthy MCP endpoint instead of creating a second process
        # that will fight it for the extension WebSocket port.
        if self._endpoint_is_up():
            try:
                self._initialize()
                result = self._request("tools/list", {})
                self.tools = (result.get("tools") if isinstance(result, dict) else []) or []
                return self.tools
            except Exception:
                self.session_id = ""
        if self.root is not None:
            command = [str(self.root / "venv/bin/python"), str(self.root / "server/server.py"),
                       "--host", "127.0.0.1", "--port", self.ws_port,
                       "--mcp-port", self.mcp_port]
            if not Path(command[0]).exists() or not Path(command[1]).exists():
                raise FileNotFoundError(f"FoxMCP installation is incomplete: {self.root}")
            log_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "cozy"
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / "foxmcp.log"
                self.log_file = log_path.open("w", encoding="utf-8")
            except OSError:
                log_dir = Path(tempfile.gettempdir()) / "cozy"
                log_dir.mkdir(parents=True, exist_ok=True)
                self.log_file = (log_dir / "foxmcp.log").open("w", encoding="utf-8")
            self.process = subprocess.Popen(command, cwd=self.root,
                                            stdout=self.log_file, stderr=subprocess.STDOUT,
                                            text=True)
            self._wait_for_server()
        self._initialize()
        result = self._request("tools/list", {})
        self.tools = (result.get("tools") if isinstance(result, dict) else []) or []
        return self.tools

    def _wait_for_server(self) -> None:
        deadline = time.time() + float(os.environ.get("COZY_FOXMCP_START_TIMEOUT", "15"))
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                details = ""
                log_path = Path(self.log_file.name) if self.log_file else None
                if log_path and log_path.exists():
                    details = log_path.read_text(encoding="utf-8", errors="replace")[-1200:].strip()
                raise RuntimeError(
                    f"FoxMCP exited with code {self.process.returncode}"
                    + (f": {details}" if details else ""))
            self._update_dynamic_mcp_port()
            if self._endpoint_is_up():
                return
            else:
                time.sleep(0.25)
        log_path = Path(self.log_file.name) if self.log_file else None
        details = ""
        if log_path and log_path.exists():
            details = log_path.read_text(encoding="utf-8", errors="replace")[-1200:].strip()
        raise TimeoutError(f"FoxMCP did not start at {self.base_url}"
                           + (f": {details}" if details else ""))

    def _update_dynamic_mcp_port(self) -> None:
        if self.log_file is None or os.environ.get("COZY_FOXMCP_URL"):
            return
        try:
            text = Path(self.log_file.name).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        match = re.search(r"MCP server will use port (\d+)", text)
        if match:
            self.base_url = f"http://127.0.0.1:{match.group(1)}/mcp"

    def _endpoint_is_up(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url, timeout=0.8):
                return True
        except urllib.error.HTTPError as exc:
            # FastMCP commonly answers a GET with 404/405 while its POST
            # JSON-RPC endpoint is already ready.
            return exc.code in {404, 405, 406, 426}
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def _post(self, payload: dict, timeout: float = 60) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        req = urllib.request.Request(self.base_url, data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            sid = response.headers.get("Mcp-Session-Id")
            if sid:
                self.session_id = sid
            body = response.read().decode("utf-8", "replace")
        # Streamable HTTP replies are SSE and commonly start with
        # ``event: message`` before the JSON ``data:`` line.
        if "data:" in body:
            body = next((line[5:].strip() for line in body.splitlines()
                         if line.startswith("data:")), "{}")
        value = json.loads(body or "{}")
        if "error" in value:
            raise RuntimeError(value["error"].get("message", value["error"]))
        return value.get("result", value)

    def _initialize(self) -> None:
        self._post({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                               "clientInfo": {"name": "cozy", "version": "3.0"}}},
                   timeout=8)
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
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None


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
