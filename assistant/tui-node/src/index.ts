import {
  BoxRenderable,
  CliRenderEvents,
  InputRenderable,
  InputRenderableEvents,
  RGBA,
  ScrollBoxRenderable,
  StyledText,
  TextRenderable,
  createCliRenderer,
  type CliRenderer,
  type TextChunk,
} from "@opentui/core";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createWriteStream, type WriteStream } from "node:fs";
import { mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseEngineEvent, textField, type EngineEvent, type ModelName } from "./protocol.ts";
import { barsForLevels, formatElapsed, initialState, reduceEvent, type CozyState } from "./state.ts";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..", "..", "..");
const python = process.env.COZY_PYTHON || resolve(root, "assistant", ".venv", "bin", "python");
const runtime = process.env.COZY_RUNTIME || resolve(root, "assistant", "runtime.py");

class EngineSupervisor {
  private child?: ChildProcessWithoutNullStreams;
  private listeners = new Set<(event: EngineEvent) => void>();
  private stopping = false;
  private killTimer?: NodeJS.Timeout;
  private stderrTail: string[] = [];
  private stderrLog?: WriteStream;

  subscribe(listener: (event: EngineEvent) => void) { this.listeners.add(listener); }
  private emit(event: EngineEvent) { for (const listener of this.listeners) listener(event); }

  start() {
    this.stopping = false;
    this.emit({ kind: "backend_start", ts: Date.now() / 1000 });
    // Persist full engine stderr: the on-screen box only fits a few lines,
    // and native crashes (MKL/CUDA/dlopen) print their cause there.
    try {
      const dir = join(homedir(), ".cache", "cozy");
      mkdirSync(dir, { recursive: true });
      this.stderrLog?.end();
      this.stderrLog = createWriteStream(join(dir, "engine-stderr.log"), { flags: "w" });
      this.stderrLog.write(`--- cozy engine started ${new Date().toISOString()} ---\n`);
    } catch {
      this.stderrLog = undefined;
    }
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
      this.stderrLog?.write(chunk);
      stderrBuffer += chunk.toString();
      const lines = stderrBuffer.split("\n");
      stderrBuffer = lines.pop() || "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed) {
          this.stderrTail.push(trimmed);
          if (this.stderrTail.length > 5) this.stderrTail.shift();
        }
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
        const tail = this.stderrTail.length ? ` ${this.stderrTail.join(" | ")}` : "";
        this.emit({ kind: "backend_crash", message: `Engine exited (${signal || `code ${code ?? "unknown"}`}).${tail}`, ts: Date.now() / 1000 });
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
    this.stderrLog?.end();
    this.stderrLog = undefined;
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

// A roomy terminal workspace: conversation first, operational context beside it.
const colors = {
  page: "#0b0d10",
  ink: "#d0d6e0",
  muted: "#5c6370",
  faint: "#3b3b3b",
  green: "#c3e88d",
  blue: "#7aa2f7",
  peach: "#e0c07c",
  red: "#ff6b6b",
  track: "#23272f",
};

const CAT = "|\\__/,|\n_.|o o  |_\n-(((---(((";
const SEP = "─".repeat(200);
const MODEL_NAMES: ModelName[] = ["wake", "stt", "llm", "foxmcp", "mcp", "cleanup", "tts"];

const ink = RGBA.fromHex(colors.ink);
const muted = RGBA.fromHex(colors.muted);
const green = RGBA.fromHex(colors.green);
const blue = RGBA.fromHex(colors.blue);
const peach = RGBA.fromHex(colors.peach);
const red = RGBA.fromHex(colors.red);

function chunk(text: string, fg: RGBA = muted): TextChunk {
  return { __isChunk: true, text, fg };
}

const renderer: CliRenderer = await createCliRenderer({ exitOnCtrlC: true, useMouse: true });
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
const loadCol = new BoxRenderable(renderer, {
  width: 76, flexDirection: "column", gap: 1,
});
loadingScreen.add(loadCol);
loadCol.add(new TextRenderable(renderer, { content: CAT, fg: colors.ink, height: 3 }));
const loadingHeadline = new TextRenderable(renderer, { content: "preparing models…", fg: colors.muted, height: 1 });
loadCol.add(loadingHeadline);
const progressTrack = new BoxRenderable(renderer, { width: 60, height: 1, backgroundColor: colors.track });
const progressFill = new BoxRenderable(renderer, { width: "1%", height: 1, backgroundColor: colors.blue });
progressTrack.add(progressFill);
loadCol.add(progressTrack);
const loadRows: TextRenderable[] = MODEL_NAMES.map(() => {
  const el = new TextRenderable(renderer, { content: "", fg: colors.muted, width: 76, height: 1 });
  loadCol.add(el);
  return el;
});
const startupError = new TextRenderable(renderer, { content: "", fg: colors.red, width: 76, height: 4, wrapMode: "word" });
loadCol.add(startupError);
const TIPS = [
  "tip: voice transcripts are cleaned before planning",
  "tip: browser tasks can use FoxMCP",
  "tip: type a task and press enter",
  "tip: context is summarized as it fills",
];
const tipsEl = new TextRenderable(renderer, { content: TIPS[0], fg: colors.faint, width: 76, height: 1 });
loadCol.add(tipsEl);
loadCol.add(new TextRenderable(renderer, { content: "ctrl+c quit", fg: colors.faint, height: 1 }));

const workspace = new BoxRenderable(renderer, {
  visible: false, width: "100%", height: "100%", flexDirection: "column",
  backgroundColor: colors.page, padding: 1,
});
page.add(workspace);
const mainRow = new BoxRenderable(renderer, {
  width: "100%", height: "100%", flexDirection: "row", gap: 2,
});
workspace.add(mainRow);
const col = new BoxRenderable(renderer, {
  flexGrow: 1, height: "100%", flexDirection: "column", gap: 1,
});
mainRow.add(col);
const header = new BoxRenderable(renderer, {
  width: "100%", height: 3, flexDirection: "row", alignItems: "center",
  justifyContent: "space-between",
});
header.add(new TextRenderable(renderer, { content: CAT, fg: colors.ink, width: 30, height: 3 }));
const headerRight = new BoxRenderable(renderer, {
  flexDirection: "column", alignItems: "flex-end",
});
headerRight.add(new TextRenderable(renderer, { content: "Cozy ui v0.1", fg: colors.muted, height: 1 }));
const phaseText = new TextRenderable(renderer, { content: "● starting", fg: colors.muted, height: 1 });
headerRight.add(phaseText);
header.add(headerRight);
col.add(header);
const pillsLine = new TextRenderable(renderer, { content: "", fg: colors.muted, width: "100%", height: 1 });
col.add(pillsLine);
col.add(new TextRenderable(renderer, { content: SEP, fg: colors.track, height: 1 }));

// Live listening visualizer (the Stitch listening screen, for real):
// status line + flat mic-level waveform, visible only while the engine is
// capturing or transcribing speech.
const vizBox = new BoxRenderable(renderer, {
  visible: false, width: "100%", flexDirection: "column", gap: 0,
});
const vizStatus = new TextRenderable(renderer, { content: "", fg: colors.blue, width: "100%", height: 1 });
const vizBars = new TextRenderable(renderer, { content: "", fg: colors.blue, width: "100%", height: 1 });
vizBox.add(vizStatus);
vizBox.add(vizBars);
col.add(vizBox);

const conversation = new ScrollBoxRenderable(renderer, {
  flexGrow: 1, width: "100%", scrollY: true, stickyScroll: true, stickyStart: "bottom",
});
const conversationText = new TextRenderable(renderer, { content: "Ready when you are.\n", fg: colors.ink, width: "100%", wrapMode: "word" });
conversation.add(conversationText);
col.add(conversation);

col.add(new TextRenderable(renderer, { content: SEP, fg: colors.track, height: 1 }));
const composer = new BoxRenderable(renderer, {
  width: "100%", height: 1, flexDirection: "row", alignItems: "center",
});
composer.add(new TextRenderable(renderer, { content: ">", fg: colors.blue, width: 2, height: 1 }));
const input = new InputRenderable(renderer, {
  flexGrow: 1, value: "", placeholder: "Type a task, question, or command...", placeholderColor: colors.muted,
  textColor: colors.ink, cursorColor: colors.blue, maxLength: 500,
});
composer.add(input);
col.add(composer);
col.add(new TextRenderable(renderer, { content: "enter send  •  ctrl+c quit", fg: colors.faint, height: 1 }));

const sidebar = new BoxRenderable(renderer, {
  width: 32, height: "100%", flexDirection: "column", gap: 1,
});
mainRow.add(sidebar);
sidebar.add(new TextRenderable(renderer, { content: "CONTEXT", fg: colors.blue, height: 1 }));
const contextText = new TextRenderable(renderer, { content: "0 / 1800 tokens", fg: colors.ink, width: "100%", height: 2, wrapMode: "word" });
sidebar.add(contextText);
const contextBar = new TextRenderable(renderer, { content: "", fg: colors.green, width: "100%", height: 1 });
sidebar.add(contextBar);
sidebar.add(new TextRenderable(renderer, { content: SEP, fg: colors.track, height: 1 }));
sidebar.add(new TextRenderable(renderer, { content: "ACTIVITY", fg: colors.blue, height: 1 }));
const activityText = new TextRenderable(renderer, { content: "Waiting for a task", fg: colors.muted, width: "100%", height: 4, wrapMode: "word" });
sidebar.add(activityText);
sidebar.add(new TextRenderable(renderer, { content: SEP, fg: colors.track, height: 1 }));
sidebar.add(new TextRenderable(renderer, { content: "SERVICES", fg: colors.blue, height: 1 }));
const servicesText = new TextRenderable(renderer, { content: "", fg: colors.muted, width: "100%", flexGrow: 1, wrapMode: "word" });
sidebar.add(servicesText);

function pillChunks(now: number): TextChunk[] {
  const chunks: TextChunk[] = [];
  for (const name of MODEL_NAMES) {
    const status = state.models[name];
    const elapsed = status === "loading"
      ? Math.max(0, now - state.loadingStartedAt)
      : state.modelElapsed[name];
    const mark = status === "done" ? "●" : status === "failed" ? "×" : status === "loading" ? "◉" : "○";
    const color = status === "done" ? green : status === "failed" ? red : status === "loading" ? blue : muted;
    chunks.push(chunk("[", muted), chunk(name, ink), chunk(mark, color));
    chunks.push(chunk(`]${elapsed === undefined ? "" : ` ${elapsed.toFixed(1)}s`}  `, muted));
  }
  return chunks;
}

function friendlyTool(name: string): string {
  const names: Record<string, string> = {
    "app.list_running": "Listed open apps", "browser.mcp": "Worked in Firefox",
    "terminal.run": "Ran terminal command", "python.kernel": "Ran local Python",
    "terminal.elevate": "Opened elevated terminal",
    "browser.search": "Searched the web", "screenshot.take": "Captured screen",
    "rlm.delegate": "Delegated task", "mcp.call": "Used connected service",
  };
  return names[name] || name.replaceAll(".", " ");
}

function markdownChunks(text: string, base: RGBA): TextChunk[] {
  const chunks: TextChunk[] = [];
  for (const line of text.split("\n")) {
    const heading = line.match(/^#{1,3}\s+(.+)/);
    const bullet = line.match(/^\s*[-*]\s+(.+)/);
    const body = heading?.[1] || bullet?.[1] || line;
    if (bullet) chunks.push(chunk("  • ", blue));
    const parts = body.split(/(`[^`]+`|\*\*[^*]+\*\*)/g);
    for (const part of parts) {
      if (part.startsWith("`") && part.endsWith("`")) chunks.push(chunk(part.slice(1, -1), green));
      else if (part.startsWith("**") && part.endsWith("**")) chunks.push(chunk(part.slice(2, -2), peach));
      else chunks.push(chunk(part, heading ? peach : base));
    }
    chunks.push(chunk("\n", base));
  }
  return chunks;
}

function eventLines(events: EngineEvent[]) {
  const chunks: TextChunk[] = [];
  for (const event of events) {
    if (event.kind === "heard" || event.kind === "user_msg") {
      chunks.push(chunk("YOU\n", blue), ...markdownChunks(textField(event, "text"), ink), chunk("\n", ink));
    } else if (event.kind === "done") {
      chunks.push(chunk("COZY\n", green), ...markdownChunks(textField(event, "text"), ink), chunk("\n", ink));
    } else if (event.kind === "tool_result") {
      chunks.push(chunk(`  ✓ ${friendlyTool(textField(event, "name"))}\n`, green));
    } else if (event.kind === "tool_fail") {
      chunks.push(chunk(`  × ${friendlyTool(textField(event, "name"))} failed\n`, red));
    } else if (event.kind === "rejected") {
      chunks.push(chunk(`  ! ${textField(event, "reason")}\n`, peach));
    } else if (event.kind === "error") {
      chunks.push(chunk(`  × ${textField(event, "msg")}\n`, red));
    } else if (event.kind === "backend_crash") {
      chunks.push(chunk(`  × engine: ${textField(event, "message")}\n`, red));
    }
  }
  return chunks.length ? new StyledText(chunks) : new StyledText([chunk("Ready when you are.\n", muted)]);
}

function updateUi() {
  const now = Date.now() / 1000;
  const total = Math.max(0, now - bootedAt);
  const failed = state.phase === "error";
  const active = MODEL_NAMES.find(name => state.models[name] === "loading");
  loadingHeadline.content = failed
    ? "startup could not finish"
    : active ? `warming ${active}… ${Math.max(0, now - state.loadingStartedAt).toFixed(1)}s · total ${total.toFixed(0)}s` : "preparing models…";
  for (let i = 0; i < MODEL_NAMES.length; i++) {
    const name = MODEL_NAMES[i];
    const status = state.models[name];
    const elapsed = status === "loading"
      ? Math.max(0, now - state.loadingStartedAt)
      : state.modelElapsed[name];
    const mark = status === "done" ? "●" : status === "failed" ? "×" : status === "loading" ? "◉" : "○";
    loadRows[i].content = `${mark} ${name.padEnd(8)}${elapsed === undefined ? "" : `${elapsed.toFixed(1)}s`}`;
    loadRows[i].fg = status === "done" ? colors.green : status === "failed" ? colors.red : status === "loading" ? colors.blue : colors.muted;
  }
  tipsEl.content = TIPS[Math.floor(total / 5) % TIPS.length];
  pillsLine.content = new StyledText(pillChunks(now));
  progressFill.width = `${Math.max(1, Math.round(state.startupProgress * 100))}%`;
  startupError.content = failed ? state.fatalError : "";

  const ready = state.hasStarted;
  loadingScreen.visible = !ready;
  workspace.visible = ready;
  if (ready) {
    const phaseColor = state.phase === "ready" ? colors.green
      : state.phase === "error" ? colors.red
      : state.phase === "thinking" || state.phase === "speaking" ? colors.peach
      : state.phase === "listening" || state.phase === "capturing" || state.phase === "transcribing" ? colors.blue
      : colors.muted;
    phaseText.content = `● ${state.phase}`;
    phaseText.fg = phaseColor;
    const live = state.phase === "listening" || state.phase === "capturing" || state.phase === "transcribing";
    vizBox.visible = live;
    if (live) {
      const transcribing = state.phase === "transcribing";
      const heard = state.transcript && state.transcript !== "Listening…" && state.transcript !== "Transcribing…"
        ? ` • > ${state.transcript}`
        : "";
      vizStatus.content = transcribing
        ? `◉ transcribing…${heard}`
        : `● LISTENING ${formatElapsed(state.listeningStartedAt || now, now)}${heard}`;
      const vizColor = transcribing ? colors.peach : colors.blue;
      vizStatus.fg = vizColor;
      vizBars.content = barsForLevels(state.audioHistory);
      vizBars.fg = vizColor;
    }
    conversationText.content = eventLines(state.events);
    const contextPct = Math.max(0, Math.min(1, state.contextUsed / Math.max(1, state.contextLimit)));
    const contextColumns = 28;
    contextText.content = `${state.contextUsed} / ${state.contextLimit} tokens\n${Math.round(contextPct * 100)}% active`;
    contextBar.content = "█".repeat(Math.round(contextPct * contextColumns)) + "░".repeat(contextColumns - Math.round(contextPct * contextColumns));
    const latestAction = [...state.events].reverse().find(event => event.kind === "llm");
    const latestResult = [...state.events].reverse().find(event => event.kind === "tool_result" || event.kind === "tool_fail");
    activityText.content = state.phase === "thinking" && latestAction
      ? `Working\n${friendlyTool(textField(latestAction, "tool"))}`
      : latestResult
        ? `${latestResult.kind === "tool_result" ? "Completed" : "Needs attention"}\n${friendlyTool(textField(latestResult, "name"))}`
        : state.phase === "listening" || state.phase === "capturing"
          ? "Listening for voice input"
          : "Waiting for a task";
    servicesText.content = MODEL_NAMES.map(name => {
      const status = state.models[name];
      const mark = status === "done" ? "●" : status === "failed" ? "×" : "○";
      return `${mark} ${name}`;
    }).join("\n");
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
const bootedAt = Date.now() / 1000;
const clock = setInterval(updateUi, 100);
updateUi();
supervisor.start();
