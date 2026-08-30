// Cozy TUI v2 - proper Stitch-style layout.
//
// Layout (no background, just text on the terminal):
//   +------------------------------------------------+
//   |  cat top-left    Cozy ui    v0.1              |  <- status
//   |  [wake●][stt●][llm●][tts●]  [ready]         |  <- progress
//   |                                                |
//   |  12:34  WAKE  score=0.62                        |  <- scrollable
//   |  12:34  heard "set volume to 30"               |     event area
//   |  12:34  llm  system.volume.set(level=30)      |
//   |  12:34  tts  "Done. Volume set."               |  <- icon only, no text
//   |  12:35  USER "open chrome"                     |
//   |  12:35  LLM  app.open(name=chrome)             |
//   |                                                |
//   +------------------------------------------------+
//   |  > _                                              |  <- sticky input
//   +------------------------------------------------+
//
// Protocol (NDJSON over stdio, one event per line):
//   {"kind": "warmup", "model": "wake", "state": "done"}
//   {"kind": "ready"}
//   {"kind": "wake", "score": 0.62}
//   {"kind": "wake_score", "score": 0.45}    <- drives visualizer
//   {"kind": "heard", "text": "set volume to 30"}
//   {"kind": "llm", "tool": "system.volume.set", "args": "..."}
//   {"kind": "tool_result", "name": "...", "out": "..."}
//   {"kind": "tts", "text": "Done. Volume set."}  <- text is in the event
//                                                log as a small icon
//   {"kind": "llm_text", "text": "..."}     <- bright white box
//   {"kind": "done", "text": "..."}          <- final answer, bright

import React, { useState, useEffect, useRef, useMemo } from "react";
import { render, Box, Text, useInput, useApp } from "ink";
import { spawn } from "node:child_process";
import { resolve as pathResolve } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = pathResolve(__filename, "..");
const ROOT = pathResolve(__dirname, "..", "..");
const PYTHON = process.env.COZY_PYTHON || pathResolve(ROOT, "assistant", ".venv", "bin", "python");
const RUNTIME = process.env.COZY_RUNTIME || pathResolve(ROOT, "assistant", "runtime.py");

// ── Cat ASCII frames (Hermes-style 6-frame cycle)
const CAT_FRAMES = [
  "      |\\__/,|   (`\\\n   _.|o o  |_   ) )\n -(((---(((--------",
  "      |\\__/,|   (-\\\n   _.|o -  |_   ) )\n -(((---(((--------",
  "      |\\__/,|   (`/\n   _.|o o  |_   ) )\n -(((---(((--------",
  "      |\\__/,|   (`\\\n   _.|o O  |_   ) )!\n -(((---(((--------",
  "      |\\__/,|   (`\\\n   _.|o o  |_   ) ?\n -(((---(((-------",
  "      |\\__/,|   (`)\n   _.|o o  |_   ) )\n -(((---(((--------",
];

const SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const SPINNER_DOTS = [".", "..", "...", "...."];

// ── Color tokens (Stitch-inspired)
const C = {
  muted: "gray",
  dim: "#5c6370",
  bright: "#d0d6e0",
  white: "white",
  blue: "#7aa2f7",
  green: "#c3e88d",
  pink: "#ff9b9b",
  red: "#ff6b6b",
  yellow: "#e0c07c",
  bg: "gray",
};

// ── Progress bar (4 segments) - separate component
function ProgressBar({ segments, state, spinnerIdx }) {
  return React.createElement(
    Box,
    { marginTop: 1, marginBottom: 1 },
    segments.map((seg, i) => {
      let mark, color;
      if (seg.state === "done") { mark = "●●●"; color = C.green; }
      else if (seg.state === "loading") {
        mark = ["◐◐◐", "●◐◐", "◐●◐", "◐◐●"][spinnerIdx % 4];
        color = C.yellow;
      }
      else if (seg.state === "failed") { mark = "✗"; color = C.red; }
      else { mark = "○○○"; color = C.muted; }
      return React.createElement(
        Text,
        { key: seg.name, color: seg.state === "pending" ? C.muted : color },
        `[${seg.name}]`,
        React.createElement(Text, { color }, mark),
        i < segments.length - 1 ? React.createElement(Text, null, "  ") : null
      );
    })
  );
}

// ── Status bar (top, always visible)
function StatusBar({ state, spinnerIdx }) {
  let label, color;
  if (state === "thinking" || state === "loading") {
    label = SPINNER_DOTS[spinnerIdx].padEnd(4, " ");
    color = C.yellow;
  } else if (state === "listening") {
    label = "listening";
    color = C.blue;
  } else if (state === "speaking") {
    label = "tts";
    color = C.pink;
  } else if (state === "done") {
    label = "ready";
    color = C.green;
  } else if (state === "error") {
    label = "error";
    color = C.red;
  } else {
    label = "ready";
    color = C.green;
  }
  return React.createElement(
    Box,
    null,
    React.createElement(Text, { bold: true, color: C.bright }, "Cozy ui"),
    React.createElement(Text, { color: C.dim }, "  v0.1  "),
    React.createElement(Text, { color }, `[${label}]`)
  );
}

// ── Audio visualizer - real-time, energy-driven
function AudioVisualizer({ level }) {
  const barLen = 40;
  const left = Math.floor(barLen / 2);
  const fill = Math.min(Math.floor(level * left), left);
  let bar = "";
  for (let i = 0; i < barLen; i++) bar += " ";
  for (let i = 0; i < fill; i++) {
    if (left - 1 - i >= 0) bar = bar.substring(0, left - 1 - i) + "▌" + bar.substring(left - i);
    if (left + i < barLen) bar = bar.substring(0, left + i) + "▐" + bar.substring(left + i + 1);
  }
  return React.createElement(
    Text,
    null,
    React.createElement(Text, { color: C.muted }, "> "),
    React.createElement(Text, { color: C.blue }, `[${bar}]`)
  );
}

// ── STT streaming box - solid box around streaming text
function SttBox({ text, spinnerIdx }) {
  if (!text) {
    return React.createElement(
      Text,
      null,
      React.createElement(Text, { color: C.muted }, "> "),
      React.createElement(Text, { color: C.dim }, "┌──────────────────────┐"),
      React.createElement(Text, { color: C.muted }, "│  "),
      React.createElement(Text, { color: C.yellow }, SPINNER_DOTS[spinnerIdx]),
      React.createElement(Text, { color: C.muted }, " listening     │"),
      React.createElement(Text, { color: C.dim }, "└──────────────────────┘")
    );
  }
  const lines = text.split("\n");
  const maxLen = Math.max(...lines.map(l => l.length));
  const top = "┌" + "─".repeat(maxLen + 2) + "┐";
  const bot = "└" + "─".repeat(maxLen + 2) + "┘";
  return React.createElement(
    Text,
    null,
    React.createElement(Text, { color: C.muted }, "> "),
    React.createElement(Text, { color: C.dim }, top),
    ...lines.flatMap((line, i) => [
      React.createElement(Text, { key: i, color: C.dim }, "│ "),
      React.createElement(Text, { key: i + "x", color: C.bright }, line),
      React.createElement(Text, { key: i + "y", color: C.dim }, " │"),
    ]),
    React.createElement(Text, { color: C.dim }, bot)
  );
}

// ── Done box - bright white, the LLM's final answer
function DoneBox({ text }) {
  if (!text) {
    return React.createElement(
      Text,
      null,
      React.createElement(Text, { color: C.muted }, "> _")
    );
  }
  const lines = text.split("\n");
  const maxLen = Math.max(...lines.map(l => l.length));
  const top = "┌" + "═".repeat(maxLen + 2) + "┐";
  const bot = "└" + "═".repeat(maxLen + 2) + "┘";
  return React.createElement(
    Text,
    null,
    React.createElement(Text, { color: C.muted }, "> "),
    React.createElement(Text, { color: C.white }, top),
    ...lines.flatMap((line, i) => [
      React.createElement(Text, { key: i, color: C.white }, "║ "),
      React.createElement(Text, { key: i + "x", bold: true, color: C.white }, line),
      React.createElement(Text, { key: i + "y", color: C.white }, " ║"),
    ]),
    React.createElement(Text, { color: C.white }, bot)
  );
}

// ── Event log entry renderer
function EventEntry({ event, pulse, spinnerIdx }) {
  const ts = new Date(event.ts * 1000).toLocaleTimeString("en-US", {
    hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit"
  });
  const timeStr = ts.slice(0, 8);  // HH:MM:SS

  if (event.kind === "wake") {
    return React.createElement(Text, null,
      React.createElement(Text, { color: C.dim }, timeStr + "  "),
      React.createElement(Text, { bold: true, color: C.blue }, "WAKE "),
      React.createElement(Text, null, `score=${event.score.toFixed(2)}`));
  }
  if (event.kind === "wake_score") {
    // Show a small audio bar in the log for wake_score events
    return React.createElement(Text, null,
      React.createElement(Text, { color: C.dim }, timeStr + "  "),
      React.createElement(Text, { color: C.blue }, "▌".repeat(Math.min(Math.floor(event.score * 20), 20))),
      React.createElement(Text, { color: C.dim }, ` ${event.score.toFixed(2)}`));
  }
  if (event.kind === "heard") {
    return React.createElement(Text, null,
      React.createElement(Text, { color: C.dim }, timeStr + "  "),
      React.createElement(Text, { color: C.bright }, "heard "),
      React.createElement(Text, null, `"${event.text}"`));
  }
  if (event.kind === "llm") {
    return React.createElement(Text, null,
      React.createElement(Text, { color: C.dim }, timeStr + "  "),
      React.createElement(Text, { color: C.green }, "llm   "),
      React.createElement(Text, { color: C.bright }, event.tool),
      event.args ? React.createElement(Text, { color: C.dim }, ` ${event.args}`) : null);
  }
  if (event.kind === "tool_result") {
    const ok = event.name && !event.name.startsWith("failed");
    return React.createElement(Text, null,
      React.createElement(Text, { color: C.dim }, timeStr + "  "),
      React.createElement(Text, { color: ok ? C.green : C.red }, "● "),
      React.createElement(Text, { color: C.dim }, event.name + " "),
      React.createElement(Text, { color: C.dim }, (event.out || "").slice(0, 60)));
  }
  if (event.kind === "tool_fail" || event.kind === "tool_error") {
    return React.createElement(Text, null,
      React.createElement(Text, { color: C.dim }, timeStr + "  "),
      React.createElement(Text, { color: C.red }, "✗ "),
      React.createElement(Text, { color: C.red }, `${event.name || "?"}: ${(event.out || event.msg || "").slice(0, 60)}`));
  }
  if (event.kind === "llm_text") {
    return React.createElement(Text, null,
      React.createElement(Text, { color: C.dim }, timeStr + "  "),
      React.createElement(Text, { color: C.green }, "llm   "),
      React.createElement(Text, { color: C.bright }, `"${(event.text || "").slice(0, 80)}"`));
  }
  if (event.kind === "tts") {
    // Don't show the TTS text - the user hears it. Just show "🔊 playing"
    return React.createElement(Text, null,
      React.createElement(Text, { color: C.dim }, timeStr + "  "),
      React.createElement(Text, { color: C.pink }, "🔊 playing"));
  }
  if (event.kind === "done") {
    return React.createElement(Text, null,
      React.createElement(Text, { color: C.dim }, timeStr + "  "),
      React.createElement(Text, { color: C.green, bold: true }, "done"),
      React.createElement(Text, null, ` "${(event.text || "").slice(0, 60)}"`));
  }
  if (event.kind === "warmup") {
    return null;  // Don't show in the log - the progress bar covers it
  }
  if (event.kind === "ready") {
    return null;
  }
  if (event.kind === "error") {
    return React.createElement(Text, null,
      React.createElement(Text, { color: C.dim }, timeStr + "  "),
      React.createElement(Text, { color: C.red }, "ERR  "),
      React.createElement(Text, { color: C.red }, event.msg || ""));
  }
  return null;
}

// ── Animated cat
function Cat({ state, frame }) {
  const speed = ["loading", "listening"].includes(state) ? 0.15 : 0.4;
  return React.createElement(
    Text, { color: C.bright },
    CAT_FRAMES[frame % CAT_FRAMES.length]
  );
}

// ── Sticky input bar (always at the bottom)
function InputBar({ mode, text, level, spinnerIdx, onSubmit, value, onChange }) {
  let inner;
  if (mode === "listening") {
    inner = React.createElement(AudioVisualizer, { level: level || 0 });
  } else if (mode === "thinking" || mode === "stt") {
    inner = React.createElement(SttBox, { text, spinnerIdx });
  } else if (mode === "done") {
    inner = React.createElement(DoneBox, { text });
  } else {
    // idle - show a plain prompt
    inner = React.createElement(
      Text,
      null,
      React.createElement(Text, { color: C.muted }, "> "),
      React.createElement(Text, { color: C.bright }, value || "_")
    );
  }
  return React.createElement(
    Box,
    { borderStyle: undefined, paddingX: 1, flexDirection: "row" },
    inner
  );
}

// ── Main app
function App({ engineStdout, engineStdin }) {
  const [segments, setSegments] = useState([
    { name: "wake", state: "pending" },
    { name: "stt", state: "pending" },
    { name: "llm", state: "pending" },
    { name: "tts", state: "pending" },
  ]);
  const [state, setState] = useState("loading");
  const [audioLevel, setAudioLevel] = useState(0);
  const [catFrame, setCatFrame] = useState(0);
  const [spinnerIdx, setSpinnerIdx] = useState(0);
  const [pulse, setPulse] = useState(0);
  const [inputMode, setInputMode] = useState("idle");
  const [inputText, setInputText] = useState("");
  const [typedBuf, setTypedBuf] = useState("");
  const [logLines, setLogLines] = useState([]); // {ts, kind, ...}
  const [sttBuffer, setSttBuffer] = useState("");

  // Cat animation
  useEffect(() => {
    const speed = ["loading", "listening"].includes(state) ? 150 : 400;
    const t = setInterval(() => setCatFrame(f => (f + 1) % CAT_FRAMES.length), speed);
    return () => clearInterval(t);
  }, [state]);

  // Spinner
  useEffect(() => {
    if (state === "thinking" || state === "loading") {
      const t = setInterval(() => setSpinnerIdx(i => (i + 1) % SPINNER_DOTS.length), 150);
      return () => clearInterval(t);
    }
  }, [state]);

  // Pulse for tool call indicators
  useEffect(() => {
    if (logLines.some(l => l.kind === "llm" && !l._done)) {
      const t = setInterval(() => setPulse(p => (p + 1) % 4), 200);
      return () => clearInterval(t);
    }
  }, [logLines]);

  // Consume events from engine
  useEffect(() => {
    if (!engineStdout) return;
    let buf = "";
    const onData = (chunk) => {
      buf += chunk;
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        try { handleEvent(JSON.parse(line)); } catch (e) {}
      }
    };
    engineStdout.on("data", onData);
    return () => engineStdout.removeListener("data", onData);
  }, [engineStdout]);

  const handleEvent = (ev) => {
    const { kind } = ev;
    if (kind === "wake_score") {
      setAudioLevel(ev.score);
    } else if (kind === "wake") {
      setState("listening");
      setInputMode("listening");
      addLog(ev);
    } else if (kind === "heard") {
      setState("thinking");
      setInputMode("thinking");
      setSttBuffer(ev.text || "");
      addLog(ev);
    } else if (kind === "llm") {
      setState("thinking");
      setInputMode("thinking");
      addLog({...ev, _done: false});
    } else if (kind === "llm_text") {
      // Show as the final answer in the input area (which becomes
      // the bright white box). Don't add to log - the log already
      // has the tool call entries.
      setSttBuffer(ev.text || "");
    } else if (kind === "tool_result" || kind === "tool_fail" || kind === "tool_error") {
      setState("thinking");
      addLog(ev);
      // Mark the last llm as done
      setLogLines(lines => {
        const next = [...lines];
        for (let i = next.length - 1; i >= 0; i--) {
          if (next[i].kind === "llm" && !next[i]._done) {
            next[i] = { ...next[i], _done: true };
            break;
          }
        }
        return next;
      });
    } else if (kind === "tts") {
      // Show "playing" in the log but DON'T show the text
      setInputMode("done");
      setState("speaking");
      addLog({...ev, text: ""});
    } else if (kind === "done") {
      // The final answer - show in the input area as a bright box
      setState("done");
      setInputMode("done");
      setSttBuffer(ev.text || "");
      addLog(ev);
    } else if (kind === "rejected") {
      // No state change
    } else if (kind === "warmup") {
      setSegments(prev => prev.map(s =>
        s.name === ev.model ? { ...s, state: ev.state || "done" } : s
      ));
    } else if (kind === "ready") {
      setState("idle");
      setInputMode("idle");
    } else if (kind === "error") {
      setState("error");
      addLog(ev);
    }
  };

  const addLog = (ev) => {
    setLogLines(lines => {
      const next = [...lines, ev];
      // Keep last 200 events
      return next.slice(-200);
    });
  };

  // Keyboard input - the input bar is always visible
  useInput((input, key) => {
    if (key.ctrl && input === "c") {
      process.exit(0);
    }
    if (key.return) {
      if (typedBuf.trim() && engineStdin) {
        // Send as a decide command
        engineStdin.write(JSON.stringify({ cmd: "decide", text: typedBuf }) + "\n");
        addLog({ kind: "user_msg", text: typedBuf, ts: Date.now() / 1000 });
        setTypedBuf("");
        setInputMode("thinking");
      }
      return;
    }
    if (key.backspace || key.delete) {
      setTypedBuf(s => s.slice(0, -1));
      return;
    }
    if (input && !key.ctrl && !key.meta) {
      setTypedBuf(s => s + input);
    }
  });

  return React.createElement(
    Box,
    { flexDirection: "column", paddingX: 1, height: "100%" },
    // Top: cat + status + version
    React.createElement(
      Box,
      { flexDirection: "row", flexShrink: 0 },
      React.createElement(Cat, { state, frame: catFrame }),
      React.createElement(Box, { width: 2 }),
      React.createElement(StatusBar, { state, spinnerIdx })
    ),
    // Progress bar
    React.createElement(ProgressBar, { segments, state, spinnerIdx }),
    // Spacer
    React.createElement(Box, { flexGrow: 1, flexDirection: "column" },
      // Scrollable event log - new events go at the bottom
      ...logLines.slice(-20).map((ev, i) =>
        React.createElement(EventEntry, {
          key: `${i}-${logLines.length - 20 + i}-${ev.ts}`,
          event: ev,
          pulse,
          spinnerIdx
        })
      )
    ),
    // Bottom: sticky input bar - never replaced
    React.createElement(InputBar, {
      mode: typedBuf ? "typing" : inputMode,
      text: sttBuffer,
      level: audioLevel,
      spinnerIdx,
      onSubmit: null,
      value: typedBuf,
      onChange: setTypedBuf
    })
  );
}

function main() {
  const py = process.env.COZY_PYTHON || PYTHON;
  const script = process.env.COZY_RUNTIME || RUNTIME;
  const child = spawn(py, [script, "--json-events", "--threshold", "0.5"], {
    stdio: ["pipe", "pipe", "inherit"],
    env: { ...process.env, COZY_TUI_MODE: "node" },
  });
  process.on("SIGINT", () => { child.kill("SIGINT"); process.exit(0); });
  child.on("exit", (code) => process.exit(code || 0));
  render(React.createElement(App, { engineStdout: child.stdout, engineStdin: child.stdin }));
}

main();
