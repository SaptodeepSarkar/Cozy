#!/usr/bin/env python3
"""Tiny NDJSON backend for manual OpenTUI input/shutdown smoke tests."""
import json
import sys
import time


def emit(kind, **fields):
    print(json.dumps({"kind": kind, "ts": time.time(), **fields}), flush=True)


for index, name in enumerate(("llm",), 1):
    emit("warmup", model=name, state="loading", index=index, total=1, progress=0)
    emit("warmup", model=name, state="done", index=index, total=1,
         progress=1, elapsed_s=0.01)
emit("ready", voice=False)

for line in sys.stdin:
    try:
        command = json.loads(line)
    except json.JSONDecodeError:
        continue
    if command.get("cmd") == "decide" and command.get("text"):
        text = str(command["text"])
        emit("heard", text=text)
        emit("done", text=f"Received: {text}", dt=0.001)
emit("shutdown")
