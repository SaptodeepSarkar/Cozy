"""Interruptible NDJSON input without TextIO buffering/select mismatches."""
import json
import os
import select


def read_commands(stream, stop, submit, reject):
    pending = b""
    fd = stream.fileno()
    while not stop.is_set():
        try:
            readable, _, _ = select.select([fd], [], [], 0.2)
            if not readable:
                continue
            chunk = os.read(fd, 65536)
        except (OSError, ValueError):
            return
        if not chunk:
            stop.set()
            return
        pending += chunk
        lines = pending.split(b"\n")
        pending = lines.pop()
        if len(pending) > 65536:
            reject("command exceeds 64 KiB")
            return
        for line in lines:
            if not line.strip():
                continue
            try:
                command = json.loads(line)
            except (ValueError, UnicodeError):
                reject("invalid command JSON")
                continue
            if (not isinstance(command, dict) or command.get("cmd") != "decide" or
                    not isinstance(command.get("text"), str) or not command["text"].strip()):
                reject("expected a non-empty decide command")
                continue
            submit(command["text"].strip())
