"""External MCP servers configured in cozy.jsonc or OpenCode config."""
from __future__ import annotations

from ..harness_fast import Plugin


class MCPPlugin(Plugin):
    name = "mcp"

    def _do_load(self) -> None:
        from mcp_client import start
        self.tools = start()

    def _do_free(self) -> None:
        from mcp_client import stop
        stop()
