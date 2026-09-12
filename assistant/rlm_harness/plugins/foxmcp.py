"""FoxMCP browser bridge, loaded during Cozy startup."""
from __future__ import annotations

from ..harness_fast import Plugin


class FoxMCPPlugin(Plugin):
    name = "foxmcp"

    def _do_load(self) -> None:
        from foxmcp import start
        self.tools = start()

    def _do_free(self) -> None:
        from foxmcp import stop
        stop()
