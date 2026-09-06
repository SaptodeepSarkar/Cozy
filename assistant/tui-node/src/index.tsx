import React from "react";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { render } from "ink";
import { App } from "./app.js";
import { parseEngineEvent, type EngineEvent } from "./protocol.js";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..", "..", "..");
const python = process.env.COZY_PYTHON || resolve(root, "assistant", ".venv", "bin", "python");
const runtime = process.env.COZY_RUNTIME || resolve(root, "assistant", "runtime.py");

class EngineSupervisor {
  private child?: ChildProcessWithoutNullStreams;
  private listeners = new Set<(event: EngineEvent) => void>();
  private stopping = false;
  private restarting = false;
  private killTimer?: NodeJS.Timeout;

  subscribe = (listener: (event: EngineEvent) => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };

  private emit(event: EngineEvent) {
    for (const listener of this.listeners) listener(event);
  }

  private shouldShowDiagnostic(message: string) {
    const s = message.toLowerCase();
    if (/torch_dtype.*deprecated|use `dtype` instead|unauthenticated requests|hf_token|huggingface_hub|futurewarning|userwarning/.test(s)) return false;
    return /error|failed|traceback|exception|crash|stopped|attributeerror|object has no attribute|cannot import name/.test(s);
  }

  private emitDiagnostic(message: string) {
    if (this.shouldShowDiagnostic(message)) {
      this.emit({ kind: "error", msg: message.trim(), ts: Date.now() / 1000 });
    }
  }

  start = () => {
    this.stopping = false;
    this.restarting = false;
    this.emit({ kind: "backend_start", ts: Date.now() / 1000 });
    const forwarded = process.argv.slice(2).filter(arg => arg !== "--tui" && arg !== "--json-events");
    this.child = spawn(python, [runtime, "--json-events", ...forwarded], {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, COZY_TUI_MODE: "node", PYTHONUNBUFFERED: "1" },
    });
    let stdoutBuffer = "";
    let stderrBuffer = "";
    this.child.stdout.on("data", chunk => {
      stdoutBuffer += chunk.toString();
      const lines = stdoutBuffer.split("\n");
      stdoutBuffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = parseEngineEvent(line);
        if (event) this.emit(event);
        else this.emitDiagnostic(line);
      }
    });
    this.child.stderr.on("data", chunk => {
      stderrBuffer += chunk.toString();
      const lines = stderrBuffer.split("\n");
      stderrBuffer = lines.pop() || "";
      for (const line of lines.slice(-5)) if (line.trim()) this.emitDiagnostic(line);
    });
    this.child.on("error", error => this.emit({ kind: "backend_crash", message: error.message, ts: Date.now() / 1000 }));
    this.child.on("exit", (code, signal) => {
      if (this.killTimer) clearTimeout(this.killTimer);
      this.killTimer = undefined;
      this.child = undefined;
      if (this.restarting) {
        this.start();
      } else if (!this.stopping) {
        this.emit({ kind: "backend_crash", message: `Engine exited (${signal || `code ${code ?? "unknown"}`}).`, ts: Date.now() / 1000 });
      }
    });
  };

  send = (text: string): boolean => {
    if (!this.child?.stdin.writable) return false;
    this.child.stdin.write(`${JSON.stringify({ cmd: "decide", text })}\n`);
    return true;
  };

  stop = () => {
    this.stopping = true;
    this.restarting = false;
    if (!this.child) return;
    this.child.stdin.end();
    this.child.kill("SIGTERM");
    this.killTimer = setTimeout(() => this.child?.kill("SIGKILL"), 35_000);
  };

  restart = () => {
    if (!this.child) { this.start(); return; }
    this.stopping = true;
    this.restarting = true;
    this.child.stdin.end();
    this.child.kill("SIGTERM");
    // Inference cannot be interrupted safely mid-kernel. Give it time to
    // finish, then use SIGKILL rather than entering Python finalization with
    // live ONNX/CT2 threads. The replacement starts only after exit.
    this.killTimer = setTimeout(() => this.child?.kill("SIGKILL"), 35_000);
  };
}

const supervisor = new EngineSupervisor();
const instance = render(<App eventSource={supervisor} send={supervisor.send} restart={supervisor.restart} stop={supervisor.stop} />, { exitOnCtrlC: false });
supervisor.start();
process.once("SIGTERM", () => { supervisor.stop(); instance.unmount(); });
process.once("SIGINT", () => { supervisor.stop(); instance.unmount(); });
