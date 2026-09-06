import wrapAnsi from "wrap-ansi";
import stringWidth from "string-width";
import React, { useEffect, useReducer, useState } from "react";
import { Box, Text, useApp, useInput, useStdout } from "ink";
import type { EngineEvent, ModelName } from "./protocol.js";
import { textField } from "./protocol.js";
import { initialState, reduceEvent, type CozyState, type Phase } from "./state.js";
import { SPINNER, theme } from "./theme.js";

interface AppProps {
  eventSource: { subscribe: (listener: (event: EngineEvent) => void) => () => void };
  send: (text: string) => boolean;
  restart: () => void;
  stop: () => void;
}

const labels: Record<Phase, string> = {
  starting: "Getting ready", ready: "Ready when you are", listening: "I'm listening",
  capturing: "I'm listening", transcribing: "Finding your words", thinking: "Working on it",
  speaking: "Speaking", error: "Needs attention",
};
const names: ModelName[] = ["wake", "stt", "llm", "tts"];
const clean = (value: string) => value.replace(/[\x00-\x1f\x7f-\x9f]/g, " ");

function useTerminalSize() {
  const { stdout } = useStdout();
  const read = () => ({ columns: stdout.columns || 80, rows: stdout.rows || 24 });
  const [size, setSize] = useState(read);
  useEffect(() => {
    const resize = () => setSize(read());
    stdout.on("resize", resize);
    return () => { stdout.off("resize", resize); };
  }, [stdout]);
  return size;
}

export function Presence({ state, frame, compact }: { state: CozyState; frame: number; compact: boolean }) {
  const recording = state.phase === "capturing" || state.phase === "listening";
  const busy = ["starting", "thinking", "transcribing", "speaking"].includes(state.phase);
  const color = state.phase === "error" ? theme.danger : theme.accent;
  const muted = state.voiceEnabled && state.micMuted === true && state.phase === "ready";
  const width = compact ? 25 : 41;
  const samples = state.audioHistory.slice(-width);
  const levels = [...Array(Math.max(0, width - samples.length)).fill(0), ...samples] as number[];
  const glyphs = "▁▂▃▄▅▆▇█";
  const wave = levels.map(level => glyphs[Math.round(level * 7)]).join("");
  return <Box flexDirection="column" alignItems="center" paddingY={compact ? 0 : 1}>
    <Text color={recording ? color : theme.dim}>{recording ? wave : busy ? `·  ${SPINNER[frame % SPINNER.length]}  ·` : "·     ◇     ·"}</Text>
    <Text bold color={state.phase === "error" ? theme.danger : theme.primary}>{muted ? "Microphone muted · typing still works" : labels[state.phase]}</Text>
    {!compact && <Text color={theme.muted}>{state.phase === "ready" ? (muted ? "Unmute your microphone in system sound settings" : state.voiceEnabled ? 'Say “Hey Cozy” or type below' : "Type a request below")
      : recording ? "Speak naturally. Pause when you're finished."
      : state.phase === "starting" ? "Preparing your local assistant"
      : state.phase === "error" ? "Details below · Ctrl+R to restart"
      : state.phase === "speaking" ? "Playing your response" : "You can keep typing while I work"}</Text>}
  </Box>;
}

export function eventSummary(event: EngineEvent): { label: string; body: string; color: string } | undefined {
  if (["heard", "user_msg"].includes(event.kind)) return { label: "you", body: textField(event, "text"), color: theme.primary };
  if (event.kind === "done") return { label: "cozy", body: textField(event, "text"), color: theme.accent };
  if (event.kind === "llm") return { label: "doing", body: textField(event, "tool"), color: theme.muted };
  if (event.kind === "tool_result") return { label: "✓", body: textField(event, "name"), color: theme.success };
  if (event.kind === "rejected") return { label: "notice", body: textField(event, "reason"), color: theme.warning };
  if (["error", "tool_error", "tool_fail", "backend_crash"].includes(event.kind)) return {
    label: "!", body: textField(event, "message") || textField(event, "msg") || textField(event, "out"), color: theme.danger,
  };
  return undefined;
}

export function App({ eventSource, send, restart, stop }: AppProps) {
  const [state, dispatch] = useReducer(reduceEvent, initialState);
  const [input, setInput] = useState("");
  const [cursor, setCursor] = useState(0);
  const [frame, setFrame] = useState(0);
  const [notice, setNotice] = useState("");
  const [details, setDetails] = useState(false);
  const [help, setHelp] = useState(false);
  const [offset, setOffset] = useState(0);
  const [reducedMotion, setReducedMotion] = useState(process.env.COZY_REDUCED_MOTION === "1");
  const { exit } = useApp();
  const { rows, columns } = useTerminalSize();
  const compact = rows < 28;

  useEffect(() => eventSource.subscribe(dispatch), [eventSource]);
  useEffect(() => {
    if (reducedMotion || !["starting", "thinking", "transcribing", "speaking"].includes(state.phase)) return;
    const timer = setInterval(() => setFrame(value => value + 1), 160);
    return () => clearInterval(timer);
  }, [state.phase, reducedMotion]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 4000);
    return () => clearTimeout(timer);
  }, [notice]);

  useInput((character, key) => {
    if (key.ctrl && character === "c") { stop(); exit(); return; }
    if (key.ctrl && character === "r") { setNotice("Restarting engine…"); restart(); return; }
    if (key.ctrl && character === "d") { setDetails(value => !value); return; }
    if (key.ctrl && character === "g") { setHelp(value => !value); return; }
    if (key.pageUp) { setOffset(value => Math.min(Math.max(0, entries.length - 1), value + 5)); return; }
    if (key.pageDown) { setOffset(value => Math.max(0, value - 5)); return; }
    if (key.escape) { setInput(""); setCursor(0); setHelp(false); setOffset(0); return; }
    if (key.leftArrow) { setCursor(value => Math.max(0, value - 1)); return; }
    if (key.rightArrow) { setCursor(value => Math.min(input.length, value + 1)); return; }
    if (key.ctrl && character === "a") { setCursor(0); return; }
    if (key.ctrl && character === "e") { setCursor(input.length); return; }
    if (key.return) {
      const command = input.trim();
      if (!command) return;
      if (command === "/help") setHelp(value => !value);
      else if (command === "/details") setDetails(value => !value);
      else if (command === "/motion") setReducedMotion(value => !value);
      else if (command === "/restart") restart();
      else if (send(command)) { dispatch({ kind: "user_msg", text: command, ts: Date.now() / 1000 }); setOffset(0); }
      else { setNotice("Engine unavailable · Ctrl+R to restart"); return; }
      setInput(""); setCursor(0); return;
    }
    if (key.backspace || key.delete) {
      if (cursor > 0) { setInput(value => value.slice(0, cursor - 1) + value.slice(cursor)); setCursor(value => value - 1); }
      return;
    }
    if (character && !key.ctrl && !key.meta && !key.upArrow && !key.downArrow) {
      const inserted = clean(character);
      setInput(value => value.slice(0, cursor) + inserted + value.slice(cursor));
      setCursor(value => value + inserted.length);
    }
  });

  const conversation = [...state.events];
  if (state.phase === "speaking" && state.response) {
    conversation.push({ kind: "done", text: state.response, ts: 0 });
  }
  const entries = conversation.map(eventSummary).flatMap(entry => {
    if (!entry) return [];
    return wrapAnsi(clean(entry.body), Math.max(1, columns - 11), { hard: true, trim: false })
      .split("\n").map((body, index) => ({ ...entry, label: index === 0 ? entry.label : "", body }));
  });
  const visibleRows = Math.max(1, rows - (compact ? 11 : 14) - ((details || state.phase === "starting") ? 2 : 0) - (help ? 3 : 0));
  const end = Math.max(0, entries.length - Math.min(offset, Math.max(0, entries.length - 1)));
  const visible = entries.slice(Math.max(0, end - visibleRows), end);
  const inputWidth = Math.max(4, columns - 12);
  const start = Math.max(0, cursor - inputWidth + 1);
  let shown = input.slice(start, start + inputWidth);
  while (stringWidth(shown) > inputWidth) shown = shown.slice(0, -1);
  const caret = Math.min(cursor - start, shown.length);

  if (rows < 12 || columns < 32) return <Box flexDirection="column"><Text color={theme.accent}>cozy · {labels[state.phase]}</Text><Text>Resize to at least 32 × 12</Text><Text>Ctrl+C quit</Text></Box>;

  return <Box flexDirection="column" height={rows - 1} paddingX={2}>
    <Box justifyContent="space-between" height={2}>
      <Text bold color={theme.primary}><Text color={theme.accent}>◇</Text> cozy</Text>
      <Text color={theme.muted}>{state.voiceEnabled ? "voice + keyboard" : "keyboard"} <Text color={theme.dim}> / local</Text></Text>
    </Box>
    <Presence state={state} frame={reducedMotion ? 0 : frame} compact={compact} />
    {(details || state.phase === "starting") && <Box justifyContent="center" gap={2} height={2}>
      {names.map(name => <Text key={name} color={state.models[name] === "failed" ? theme.danger : state.models[name] === "done" ? theme.success : theme.muted}>
        {state.models[name] === "done" ? "✓" : state.models[name] === "failed" ? "×" : "·"} {name}
      </Text>)}
    </Box>}
    <Box flexDirection="column" flexGrow={1} overflow="hidden" paddingTop={1}>
      {visible.length ? visible.map((entry, index) => <Box key={index} flexShrink={0}>
        <Box width={7}><Text bold color={entry.color}>{entry.label}</Text></Box>
        <Text color={entry.label === "doing" || entry.label === "✓" ? theme.muted : theme.primary} wrap="truncate-end">{clean(entry.body)}</Text>
      </Box>) : <Box flexDirection="column" alignItems="center" justifyContent="center" flexGrow={1}>
        <Text color={theme.primary}>A little space to think.</Text>
        <Text color={theme.muted}>Try “check memory usage”</Text>
      </Box>}
    </Box>
    {help && <Box flexDirection="column" height={3}>
      <Text color={theme.muted}>/details · /motion · /restart · /help</Text>
      <Text color={theme.muted}>← → edit · Ctrl+A/E start/end · PgUp/PgDn history</Text>
      <Text color={theme.muted}>Esc clear · Motion {reducedMotion ? "reduced" : "on"}</Text>
    </Box>}
    <Text color={notice ? theme.warning : theme.dim} wrap="truncate-end">{notice || (offset ? "Earlier activity · PgDn to return" : state.fatalError || "")}</Text>
    <Box borderStyle="single" borderTop borderBottom borderLeft={false} borderRight={false} borderColor={state.phase === "error" ? theme.danger : theme.accent} paddingX={1}>
      <Text color={theme.accent}>› </Text>
      {input ? <Text color={theme.primary}>{shown.slice(0, caret)}<Text inverse>{shown[caret] || " "}</Text>{shown.slice(caret + 1)}</Text>
        : <Text color={theme.dim}><Text inverse> </Text> Ask anything…</Text>}
    </Box>
    <Box justifyContent="space-between" height={1}><Text color={theme.muted}>↵ send  /help</Text><Text color={theme.dim}>{columns >= 64 ? "Ctrl+R restart  ·  " : ""}Ctrl+C quit</Text></Box>
  </Box>;
}
