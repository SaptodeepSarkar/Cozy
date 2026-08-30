// Cozy TUI v3 - fixes the user-reported bugs:
// 1. Input bar is STICKY to the bottom of the screen
// 2. Log area scrolls independently
// 3. TTS doesn't show its text in the input area (no duplicate)
// 4. LLM done event shows the answer in the input area
// 5. Audio visualizer responds to actual wake_score (10Hz from engine)
//
// Layout:
//   +--------------------------------+
//   |  cat  Cozy ui v0.1 [ready]    |  <- top: status (fixed)
//   |  [wake●][stt●][llm●][tts●]   |  <- progress
//   |                                |
//   |  12:34 WAKE score=0.62         |  <- log (scrollable, fills space)
//   |  12:34 heard "set volume 30"   |
//   |  12:34 llm  system.volume.set |
//   |  12:34 🔊 playing              |  <- TTS shown as icon only
//   |  12:34 ● system.volume.set done|
//   |                                |
//   |                                |  <- flexGrow space
//   |                                |
//   +--------------------------------+
//   |  > _                              |  <- input: STICKY to bottom
//   +--------------------------------+
//
// The key fix: use flexDirection="column" with the log area as
// flexGrow=1, and the input area as the LAST child with no flexGrow.
// When the log fills the screen, it overflows upward (new events at
// the bottom), and the input stays at the bottom of the terminal.

import React, { useState, useEffect, useMemo } from "react";
import { render, Box, Text, useInput, useApp } from "ink";
import { spawn } from "node:child_process";
import { resolve as pathResolve } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = pathResolve(__filename, "..");
const ROOT = pathResolve(__dirname, "..", "..");
const PYTHON = process.env.COZY_PYTHON || pathResolve(ROOT, "assistant", ".venv", "bin", "python");
const RUNTIME = process.env.COZY_RUNTIME || pathResolve(ROOT, "assistant", "runtime.py");

// ── Cat ASCII frames
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

const C = {
  muted: "gray", dim: "#5c6370", bright: "#d0d6e0", white: "white",
  blue: "#7aa2f7", green: "#c3e88d", pink: "#ff9b9b", red: "#ff6b6b",
  yellow: "#e0c07c", bg: "gray",
};

// ── Progress bar
function ProgressBar({ segments, spinnerIdx }) {
  return React.createElement(Box, { marginTop: 1, marginBottom: 1 },
    segments.map((seg, i) => {
      let mark, color;
      if (seg.state === "done") { mark = "●●●"; color = C.green; }
      else if (seg.state === "loading") {
        mark = ["◐◐◐", "●◐◐", "◐●◐", "◐◐●"][spinnerIdx % 4];
        color = C.yellow;
      }
      else if (seg.state === "failed") { mark = "✗"; color = C.red; }
      else { mark = "○○○"; color = C.muted; }
      return React.createElement(Text, { key: seg.name, color: seg.state === "pending" ? C.muted : color },
        `[${seg.name}]`, React.createElement(Text, { color }, mark),
        i < segments.length - 1 ? React.createElement(Text, null, "  ") : null
      );
    })
  );
}

// ── Status bar (top, fixed)
function StatusBar({ state, spinnerIdx }) {
  let label, color;
  if (state === "thinking" || state === "loading") {
    label = SPINNER_DOTS[spinnerIdx].padEnd(4, " "); color = C.yellow;
  } else if (state === "listening") { label = "listening"; color = C.blue; }
  else if (state === "speaking")  { label = "tts";        color = C.pink;  }
  else if (state === "done")      { label = "ready";      color = C.green; }
  else if (state === "error")     { label = "error";      color = C.red;   }
  else                              { label = "ready";      color = C.green; }
  return React.createElement(Box, null,
    React.createElement(Text, { bold: true, color: C.bright }, "Cozy ui"),
    React.createElement(Text, { color: C.dim }, "  v0.1  "),
    React.createElement(Text, { color }, `[${label}]`));
}

// ── Audio visualizer (real-time, energy-driven)
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
  return React.createElement(Text, null,
    React.createElement(Text, { color: C.muted }, "> "),
    React.createElement(Text, { color: C.blue }, `[${bar}]`));
}

// ── STT streaming box
function SttBox({ text, spinnerIdx }) {
  if (!text) {
    return React.createElement(Text, null,
      React.createElement(Text, { color: C.muted }, "> "),
      React.createElement(Text, { color: C.dim }, "┌──────────────────────┐"),
      React.createElement(Text, { color: C.muted }, "│  "),
      React.createElement(Text, { color: C.yellow }, SPINNER_DOTS[spinnerIdx]),
      React.createElement(Text, { color: C.muted }, " listening     │"),
      React.createElement(Text, { color: C.dim }, "└──────────────────────┘"));
  }
  const lines = text.split("\n");
  const maxLen = Math.max(...lines.map(l => l.length));
  const top = "┌" + "─".repeat(maxLen + 2) + "┐";
  const bot = "└" + "─".repeat(maxLen + 2) + "┘";
  return React.createElement(Text, null,
    React.createElement(Text, { color: C.muted }, "> "),
    React.createElement(Text, { color: C.dim }, top),
    ...lines.flatMap((line, i) => [
      React.createElement(Text, { key: i, color: C.dim }, "│ "),
      React.createElement(Text, { key: i + "x", color: C.bright }, line),
      React.createElement(Text, { key: i + "y", color: C.dim }, " │"),
    ]),
    React.createElement(Text, { color: C.dim }, bot));
}

// ── Done box (bright white, LLM's final answer)
function DoneBox({ text }) {
  if (!text) {
    return React.createElement(Text, null,
      React.createElement(Text, { color: C.muted }, "> _"));
  }
  const lines = text.split("\n");
  const maxLen = Math.max(...lines.map(l => l.length));
  const top = "┌" + "═".repeat(maxLen + 2) + "┐";
  const bot = "└" + "═".repeat(maxLen + 2) + "┘";
  return React.createElement(Text, null,
    React.createElement(Text, { color: C.muted }, "> "),
    React.createElement(Text, { color: C.white }, top),
    ...lines.flatMap((line, i) => [
      React.createElement(Text, { key: i, color: C.white }, "║ "),
      React.createElement(Text, { key: i + "x", bold: true, color: C.white }, line),
      React.createElement(Text, { key: i + "y", color: C.white }, " ║"),
    ]),
    React.createElement(Text, { color: C.white }, bot));
}

// ── Idle input
function IdleInput({ typed }) {
  return React.createElement(Text, null,
    React.createElement(Text, { color: C.muted }, "> "),
    React.createElement(Text, { color: C.bright }, typed || "_"));
}

// ── Event log entry
function EventEntry({ event, pulse }) {
  const ts = new Date(event.ts * 1000).toLocaleTimeString("en-US",
    { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const timeStr = ts.slice(0, 8);
  if (event.kind === "wake") {
    return React.createElement(Text, null,
      React.createElement(Text, { color: C.dim }, timeStr + "  "),
      React.createElement(Text, { bold: true, color: C.blue }, "WAKE "),
      React.createElement(Text, null, `score=${event.score.toFixed(2)}`));
  }
  if (event.kind === "wake_score") {
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
  if (event.kind === "user_msg") {
    return React.createElement(Text, null,
      React.createElement(Text, { color: C.dim }, timeStr + "  "),
      React.createElement(Text, { color: C.muted }, "USER "),
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
    // TTS event: just show the icon, NOT the text (no duplicate)
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
  if (event.kind === "warmup" || event.kind === "ready" || event.kind === "rejected") return null;
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
  return React.createElement(Text, { color: C.bright },
    CAT_FRAMES[frame % CAT_FRAMES.length]);
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
  const [logLines, setLogLines] = useState([]);

  useEffect(() => {
    const speed = ["loading", "listening"].includes(state) ? 150 : 400;
    const t = setInterval(() => setCatFrame(f => (f + 1) % CAT_FRAMES.length), speed);
    return () => clearInterval(t);
  }, [state]);

  useEffect(() => {
    if (state === "thinking" || state === "loading") {
      const t = setInterval(() => setSpinnerIdx(i => (i + 1) % SPINNER_DOTS.length), 150);
      return () => clearInterval(t);
    }
  }, [state]);

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
      setInputText(ev.text || "");
      addLog(ev);
    } else if (kind === "llm_text") {
      // STT streaming - update the text in the input box
      setInputText(ev.text || "");
    } else if (kind === "llm") {
      setState("thinking");
      setInputMode("thinking");
      addLog({...ev, _done: false});
    } else if (kind === "tool_result" || kind === "tool_fail" || kind === "tool_error") {
      setState("thinking");
      addLog(ev);
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
      // TTS event: show "🔊 playing" in log. DON'T change inputMode -
      // the TTS output is what the user hears, not what should be
      // displayed (no duplicate text).
      setState("speaking");
      addLog({...ev, text: ""});
    } else if (kind === "done") {
      // The LLM's final answer - show in the input area as a bright
      // box. This is the user's actual response.
      setState("done");
      setInputMode("done");
      setInputText(ev.text || "");
      addLog(ev);
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
      return next.slice(-200);
    });
  };

  useInput((input, key) => {
    if (key.ctrl && input === "c") {
      process.exit(0);
    }
    if (key.return) {
      if (typedBuf.trim() && engineStdin) {
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

  // The KEY fix: render in this order so the input is always LAST.
  // 1. Cat + status (top, fixed)
  // 2. Progress bar (fixed)
  // 3. SPACER (flexGrow=1, takes all remaining space, pushes log up)
  // 4. Log entries (shown inside the SPACER, but if log is too long
  //    it overflows - the SPACER absorbs the overflow)
  // 5. Input bar (fixed, always at the bottom)
  return React.createElement(Box, { flexDirection: "column", paddingX: 1, height: "100%" },
    // Top: all on one line - cat, status, progress
    React.createElement(Box, { flexShrink: 0, flexDirection: "row" },
      React.createElement(Cat, { state, frame: catFrame }),
      React.createElement(StatusBar, { state, spinnerIdx }),
      React.createElement(Box, { width: 2 }),
      React.createElement(ProgressBar, { segments, spinnerIdx })),
    // Log: takes all remaining space, scrolls up
    React.createElement(Box, { flexDirection: "column", flexGrow: 1, overflow: "hidden" },
      ...logLines.slice(-25).map((ev, i) =>
        React.createElement(EventEntry, { key: `${i}-${ev.ts}-${ev.kind}`,
          event: ev, pulse }))),
    // Input bar: STICKY to the bottom, never replaced
    React.createElement(Box, { flexShrink: 0 },
      inputMode === "listening"
        ? React.createElement(AudioVisualizer, { level: audioLevel })
        : inputMode === "thinking" || inputMode === "stt"
        ? React.createElement(SttBox, { text: inputText, spinnerIdx })
        : inputMode === "done"
        ? React.createElement(DoneBox, { text: inputText })
        : React.createElement(IdleInput, { typed: typedBuf }))
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
