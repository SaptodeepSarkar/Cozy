"""A bounded, persistent IPython kernel for local agent computation."""
from __future__ import annotations

import os
import threading
import time


class KernelUnavailable(RuntimeError):
    pass


class CozyKernel:
    """Keep one kernel alive, but cap execution time and returned output."""
    def __init__(self, timeout: float = 20.0, max_output: int = 6000):
        self.timeout = timeout
        self.max_output = max_output
        self._km = None
        self._kc = None
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        if self._kc is not None:
            return
        try:
            from jupyter_client import KernelManager
        except ImportError as exc:
            raise KernelUnavailable("Install ipykernel and jupyter-client in assistant/.venv") from exc
        self._km = KernelManager(kernel_name=os.environ.get("COZY_KERNEL_NAME", "python3"))
        self._km.start_kernel()
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=self.timeout)

    def execute(self, code: str) -> str:
        if not code.strip():
            return "No Python code provided."
        with self._lock:
            self._ensure()
            msg_id = self._kc.execute(code, silent=False, store_history=True)
            chunks: list[str] = []
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                try:
                    msg = self._kc.get_iopub_msg(timeout=min(1, max(0.05, deadline - time.monotonic())))
                except Exception:
                    continue
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                content = msg.get("content", {})
                kind = msg.get("msg_type")
                if kind == "stream":
                    chunks.append(content.get("text", ""))
                elif kind in {"execute_result", "display_data"}:
                    chunks.append(str(content.get("data", {}).get("text/plain", "")))
                elif kind == "error":
                    chunks.append("\n".join(content.get("traceback", [])))
                elif kind == "status" and content.get("execution_state") == "idle":
                    return "".join(chunks)[-self.max_output:] or "(no output)"
            self.interrupt()
            return (f"Kernel execution timed out after {self.timeout} seconds.\n"
                    + "".join(chunks)[-self.max_output:])

    def interrupt(self) -> None:
        if self._km is not None:
            try:
                self._km.interrupt_kernel()
            except Exception:
                pass

    def close(self) -> None:
        if self._kc is not None:
            self._kc.stop_channels()
        if self._km is not None:
            self._km.shutdown_kernel(now=True)
        self._kc = self._km = None


_kernel: CozyKernel | None = None


def execute(code: str) -> str:
    global _kernel
    if _kernel is None:
        _kernel = CozyKernel()
    return _kernel.execute(code)


def close() -> None:
    if _kernel is not None:
        _kernel.close()
