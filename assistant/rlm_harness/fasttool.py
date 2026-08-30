"""Python wrapper for fasttool.so - the C tool-call extractor.

The C implementation is ~100x faster than the Python regex path on
realistic LLM outputs. Use this anywhere a tool call needs to be
extracted from model output.

Usage:
    from fasttool import extract_tool_call
    name, args = extract_tool_call(model_output)
    if name:
        execute(name, args)
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SO = _HERE / "libfasttool.so"

if not _SO.exists():
    # Build on first import
    import subprocess
    subprocess.check_call([
        "cc", "-O3", "-Wall", "-shared", "-fPIC",
        "-o", str(_SO), str(_HERE / "fasttool.c"),
    ])

_lib = ctypes.CDLL(str(_SO))


class _ToolCall(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 128),
        ("args", ctypes.c_char * 4096),
        ("found", ctypes.c_int),
    ]


_lib.extract_tool_call.argtypes = [
    ctypes.c_char_p, ctypes.c_int,
    ctypes.POINTER(_ToolCall),
]
_lib.extract_tool_call.restype = ctypes.c_int


def extract_tool_call(text: str) -> tuple[str, dict]:
    """Return ``(tool_name, args_dict)``. Both are empty on failure."""
    out = _ToolCall()
    rc = _lib.extract_tool_call(
        text.encode("utf-8"), len(text.encode("utf-8")),
        ctypes.byref(out),
    )
    if not rc or not out.found:
        return "", {}
    name = out.name.decode("utf-8")
    args_raw = out.args.decode("utf-8")
    if args_raw:
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {}
    else:
        args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args


if __name__ == "__main__":
    samples = [
        "<tool_call>{\"name\": \"app.open\", \"arguments\": {\"name\": \"chrome\"}}</tool_call>",
        "{\"name\": \"system.volume.set\", \"arguments\": {\"level\": 30}}",
        "Hi there!",
        "<tool_call>{\"name\":\"media.pause\"}</tool_call>",
    ]
    for s in samples:
        n, a = extract_tool_call(s)
        print(f"in : {s[:60]}")
        print(f"out: name={n!r} args={a}")
        print()
