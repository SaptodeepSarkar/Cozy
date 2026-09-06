/** Model-free interactive preview. No microphone capture or tool execution. */
import React from "react";
import { render } from "ink";
import { App } from "./app.js";
import type { EngineEvent } from "./protocol.js";

const listeners = new Set<(event: EngineEvent) => void>();
const timers = new Set<ReturnType<typeof setTimeout>>();
const emit = (kind: string, fields: Record<string, unknown> = {}) => {
  for (const listener of listeners) listener({ kind, ts: Date.now() / 1000, ...fields });
};
const later = (ms: number, action: () => void) => {
  const timer = setTimeout(() => { timers.delete(timer); action(); }, ms);
  timers.add(timer);
};
const stop = () => { for (const timer of timers) clearTimeout(timer); timers.clear(); };
const start = () => {
  stop(); emit("backend_start");
  ["wake", "stt", "llm", "tts"].forEach((model, index) => later(150 + index * 200, () => emit("warmup", { model, state: "done" })));
  later(1000, () => emit("ready"));
  later(2000, () => { emit("wake"); emit("stt_start"); });
  for (let i = 0; i < 45; i++) later(2000 + i * 80, () => emit("capture_level", { level: Math.abs(Math.sin(i * 0.38)) * 0.8 }));
  later(5700, () => emit("stt_infer"));
  later(6800, () => emit("heard", { text: "Give me a little room to think." }));
  later(7900, () => emit("tts", { text: "I'm here. What would you like to work on?" }));
  later(9300, () => emit("done", { text: "I'm here. What would you like to work on?" }));
};
render(<App eventSource={{ subscribe: listener => { listeners.add(listener); return () => { listeners.delete(listener); }; } }}
  send={text => { later(100, () => emit("heard", { text })); later(900, () => emit("done", { text: "Preview only — your request was not executed." })); return true; }}
  restart={start} stop={stop} />, { exitOnCtrlC: false });
// Wait for the React subscription before emitting preview events.
later(100, start);
