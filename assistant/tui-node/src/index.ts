import {
  BoxRenderable,
  CliRenderEvents,
  ImageRenderable,
  InputRenderable,
  InputRenderableEvents,
  ScrollBoxRenderable,
  TextRenderable,
  createCliRenderer,
  type CliRenderer,
} from "@opentui/core";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseEngineEvent, textField, type EngineEvent, type ModelName } from "./protocol.ts";
import { initialState, reduceEvent, type CozyState } from "./state.ts";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..", "..", "..");
const python = process.env.COZY_PYTHON || resolve(root, "assistant", ".venv", "bin", "python");
const runtime = process.env.COZY_RUNTIME || resolve(root, "assistant", "runtime.py");
const logo = resolve(here, "..", "assets", "cozy-logo.png");

class EngineSupervisor {
  private child?: ChildProcessWithoutNullStreams;
  private listeners = new Set<(event: EngineEvent) => void>();
  private stopping = false;
  private killTimer?: NodeJS.Timeout;

  subscribe(listener: (event: EngineEvent) => void) { this.listeners.add(listener); }
  private emit(event: EngineEvent) { for (const listener of this.listeners) listener(event); }

  start() {
    this.stopping = false;
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
        const event = parseEngineEvent(line);
        if (event) this.emit(event);
      }
    });
    this.child.stderr.on("data", chunk => {
      stderrBuffer += chunk.toString();
      const lines = stderrBuffer.split("\n");
      stderrBuffer = lines.pop() || "";
      for (const line of lines) {
        if (/traceback|error|failed|exception/i.test(line) && !/warning/i.test(line)) {
          this.emit({ kind: "error", msg: line.trim(), ts: Date.now() / 1000 });
        }
      }
    });
    this.child.on("error", error => this.emit({ kind: "backend_crash", message: error.message, ts: Date.now() / 1000 }));
    this.child.on("exit", (code, signal) => {
      if (this.killTimer) clearTimeout(this.killTimer);
      this.killTimer = undefined;
      this.child = undefined;
      if (!this.stopping) {
        this.emit({ kind: "backend_crash", message: `Engine exited (${signal || `code ${code ?? "unknown"}`}).`, ts: Date.now() / 1000 });
      }
    });
  }

  send(text: string) {
    if (!this.child?.stdin.writable) return false;
    this.child.stdin.write(`${JSON.stringify({ cmd: "decide", text })}\n`);
    return true;
  }

  stop(immediate = false) {
    if (this.stopping) return;
    this.stopping = true;
    const child = this.child;
    if (!child) return;
    child.stdin.end();
    if (immediate) {
      child.kill("SIGKILL");
      return;
    }
    child.kill("SIGTERM");
    // Exactly one fallback signal. This avoids the old repeated signal loop
    // while still recovering if a native inference call never returns.
    this.killTimer = setTimeout(() => {
      if (this.child === child && child.exitCode === null) child.kill("SIGKILL");
    }, 2_000);
  }
}

const colors = {
  page: "#080B14", panel: "#111827", panel2: "#172033", ink: "#F8FAFC",
  muted: "#94A3B8", violet: "#8B5CF6", cyan: "#22D3EE", green: "#34D399",
  red: "#FB7185", track: "#263248",
};

const renderer = await createCliRenderer({ exitOnCtrlC: true, useMouse: true });
const supervisor = new EngineSupervisor();
let state: CozyState = initialState;
let closing = false;

const page = new BoxRenderable(renderer, {
  width: "100%", height: "100%", backgroundColor: colors.page, shouldFill: true,
});
renderer.root.add(page);

// Startup is a separate screen; the assistant workspace is not exposed until
// every required model has actually initialized.
const loadingScreen = new BoxRenderable(renderer, {
  width: "100%", height: "100%", flexDirection: "column", alignItems: "center",
  justifyContent: "center", backgroundColor: colors.page, gap: 1,
});
page.add(loadingScreen);
loadingScreen.add(new ImageRenderable(renderer, { source: logo, width: 18, height: 9, fit: "fit", protocol: "auto" }));
loadingScreen.add(new TextRenderable(renderer, { content: "COZY", fg: colors.ink, height: 1 }));
const loadingHeadline = new TextRenderable(renderer, { content: "Starting your assistant", fg: colors.muted, height: 1 });
loadingScreen.add(loadingHeadline);
const progressTrack = new BoxRenderable(renderer, { width: 48, height: 1, backgroundColor: colors.track, shouldFill: true });
const progressFill = new BoxRenderable(renderer, { width: "1%", height: 1, backgroundColor: colors.violet, shouldFill: true });
progressTrack.add(progressFill);
loadingScreen.add(progressTrack);
const modelRows = new TextRenderable(renderer, { content: "", fg: colors.muted, width: 48, height: 6 });
loadingScreen.add(modelRows);
const startupError = new TextRenderable(renderer, { content: "", fg: colors.red, width: 60, height: 2, wrapMode: "word" });
loadingScreen.add(startupError);
loadingScreen.add(new TextRenderable(renderer, { content: "Ctrl+C  Exit", fg: colors.muted, height: 1 }));

const workspace = new BoxRenderable(renderer, {
  visible: false, width: "100%", height: "100%", flexDirection: "column",
  backgroundColor: colors.page, padding: 1, gap: 1,
});
page.add(workspace);
const header = new BoxRenderable(renderer, {
  width: "100%", height: 3, backgroundColor: colors.panel, paddingX: 2,
  flexDirection: "row", alignItems: "center", justifyContent: "space-between",
});
header.add(new TextRenderable(renderer, { content: "COZY", fg: colors.ink, width: 12, height: 1 }));
const phaseText = new TextRenderable(renderer, { content: "Ready", fg: colors.green, width: 24, height: 1 });
header.add(phaseText);
workspace.add(header);

const body = new BoxRenderable(renderer, { width: "100%", flexGrow: 1, flexDirection: "row", gap: 1 });
const sidebar = new BoxRenderable(renderer, { width: 25, height: "100%", backgroundColor: colors.panel, padding: 1, flexDirection: "column", gap: 1 });
sidebar.add(new TextRenderable(renderer, { content: "VOICE PIPELINE", fg: colors.muted, height: 1 }));
const pipelineText = new TextRenderable(renderer, { content: "", fg: colors.ink, width: "100%", height: 7 });
sidebar.add(pipelineText);
sidebar.add(new TextRenderable(renderer, { content: "Say “Hey Cozy”\nor type below.", fg: colors.muted, width: "100%", height: 3, wrapMode: "word" }));
body.add(sidebar);
const conversation = new ScrollBoxRenderable(renderer, {
  flexGrow: 1, height: "100%", backgroundColor: colors.panel, padding: 2,
  scrollY: true, stickyScroll: true, stickyStart: "bottom",
});
const conversationText = new TextRenderable(renderer, { content: "Cozy is ready.\n", fg: colors.ink, width: "100%", wrapMode: "word" });
conversation.add(conversationText);
body.add(conversation);
workspace.add(body);

const composer = new BoxRenderable(renderer, {
  width: "100%", height: 3, backgroundColor: colors.panel2, paddingX: 2,
  flexDirection: "row", alignItems: "center",
});
composer.add(new TextRenderable(renderer, { content: ">", fg: colors.cyan, width: 3, height: 1 }));
const input = new InputRenderable(renderer, {
  flexGrow: 1, value: "", placeholder: "Type a command…", placeholderColor: colors.muted,
  textColor: colors.ink, cursorColor: colors.cyan, maxLength: 500,
});
composer.add(input);
workspace.add(composer);

function modelLine(name: ModelName, now: number) {
  const status = state.models[name];
  const label = name === "wake" ? "Wake word" : name === "cleanup" ? "Input polish" : name.toUpperCase();
  const elapsed = status === "loading"
    ? Math.max(0, now - state.loadingStartedAt)
    : state.modelElapsed[name];
  const mark = status === "done" ? "●" : status === "failed" ? "×" : status === "loading" ? "◉" : "○";
  return `${mark}  ${label.padEnd(14)}${elapsed === undefined ? "" : `${elapsed.toFixed(1)}s`}`;
}

function eventLines(events: EngineEvent[]) {
  const lines: string[] = [];
  for (const event of events) {
    if (event.kind === "heard" || event.kind === "user_msg") lines.push(`YOU\n${textField(event, "text")}\n`);
    else if (event.kind === "done") lines.push(`COZY\n${textField(event, "text")}\n`);
    else if (event.kind === "llm") lines.push(`ACTION\n${textField(event, "tool")}\n`);
    else if (event.kind === "tool_result") lines.push(`DONE\n${textField(event, "name")}\n`);
    else if (event.kind === "rejected") lines.push(`NOTICE\n${textField(event, "reason")}\n`);
    else if (event.kind === "error") lines.push(`ERROR\n${textField(event, "msg")}\n`);
    else if (event.kind === "backend_crash") lines.push(`ENGINE\n${textField(event, "message")}\n`);
  }
  return lines.join("\n") || "Cozy is ready.\n";
}

function updateUi() {
  const now = Date.now() / 1000;
  const names: ModelName[] = ["wake", "stt", "llm", "cleanup", "tts"];
  const failed = state.phase === "error";
  loadingHeadline.content = failed
    ? "Startup could not finish"
    : state.loadingModel ? `Loading ${state.loadingModel.toUpperCase()}  ${Math.max(0, now - state.loadingStartedAt).toFixed(1)}s` : "Preparing models…";
  modelRows.content = names.map(name => modelLine(name, now)).join("\n");
  progressFill.width = `${Math.max(1, Math.round(state.startupProgress * 100))}%`;
  startupError.content = failed ? state.fatalError : "";

  const ready = state.hasStarted;
  loadingScreen.visible = !ready;
  workspace.visible = ready;
  if (ready) {
    phaseText.content = state.phase === "ready" ? "● Ready" : `● ${state.phase[0].toUpperCase()}${state.phase.slice(1)}`;
    pipelineText.content = names.map(name => modelLine(name, now)).join("\n");
    conversationText.content = eventLines(state.events);
    input.focus();
  }
  renderer.requestRender();
}

input.on(InputRenderableEvents.ENTER, (value: string) => {
  const text = value.trim();
  if (!text || state.phase === "starting" || state.phase === "error") return;
  if (supervisor.send(text)) {
    state = reduceEvent(state, { kind: "user_msg", text, ts: Date.now() / 1000 });
    input.value = "";
    updateUi();
  }
});

async function shutdown(immediate = false) {
  if (closing) return;
  closing = true;
  clearInterval(clock);
  supervisor.stop(immediate);
  renderer.destroy();
}

renderer.keyInput.on("keypress", key => {
  if (key.ctrl && key.name === "c") {
    key.preventDefault();
    void shutdown(true);
  }
});
// Keep an independent byte-level escape hatch. Raw-mode terminal stacks do
// not always turn ETX into SIGINT while an input widget is focused.
renderer.stdin.prependListener("data", (chunk: Buffer | string) => {
  const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
  if (bytes.includes(3)) void shutdown(true);
});
renderer.on(CliRenderEvents.DESTROY, () => {
  clearInterval(clock);
  supervisor.stop();
});
process.once("SIGTERM", () => void shutdown());
process.once("SIGHUP", () => void shutdown());

supervisor.subscribe(event => {
  state = reduceEvent(state, event);
  updateUi();
});
const clock = setInterval(updateUi, 100);
updateUi();
supervisor.start();
