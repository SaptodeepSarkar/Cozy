import React, { useEffect, useMemo, useReducer, useState } from "react";
import { Box, Text, useApp, useInput, useStdout } from "ink";
import type { EngineEvent, ModelName } from "./protocol.js";
import { numberField, textField } from "./protocol.js";
import { initialState, reduceEvent, type CozyState, type Phase } from "./state.js";
import { CAT_FRAMES, SPINNER, theme } from "./theme.js";

interface AppProps {
  eventSource: { subscribe: (listener: (event: EngineEvent) => void) => () => void };
  send: (text: string) => boolean;
  restart: () => void;
  stop: () => void;
}

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

const phaseLabels: Record<Phase, string> = {
  starting: "warming up", ready: "ready", listening: "listening", capturing: "capturing",
  thinking: "thinking", speaking: "speaking", error: "needs attention",
};

function Header({ state, frame }: { state: CozyState; frame: number }) {
  const active = state.phase === "starting" || state.phase === "thinking";
  const statusColor = state.phase === "error" ? theme.danger : state.phase === "ready" ? theme.success : theme.accent;
  return (
    <Box justifyContent="space-between" alignItems="center">
      <Box gap={2} alignItems="center">
        <Text color={theme.primary}>{CAT_FRAMES[frame % CAT_FRAMES.length]}</Text>
        <Box flexDirection="column">
          <Text bold color={theme.primary}>COZY</Text>
          <Text color={theme.dim}>local voice assistant</Text>
        </Box>
      </Box>
      <Box flexDirection="column" alignItems="flex-end">
        <Text color={statusColor} bold>{active ? `${SPINNER[frame % SPINNER.length]} ` : "● "}{phaseLabels[state.phase]}</Text>
        <Text color={theme.dim}>UI v2.0 · private & offline</Text>
      </Box>
    </Box>
  );
}

function Pipeline({ state }: { state: CozyState }) {
  const names: ModelName[] = ["wake", "stt", "llm", "tts"];
  return (
    <Box borderStyle="round" borderColor={theme.border} paddingX={1} gap={1}>
      <Text color={theme.dim}>PIPELINE</Text>
      {names.map((name, index) => {
        const modelState = state.models[name];
        const glyph = modelState === "done" ? "●" : modelState === "failed" ? "×" : modelState === "loading" ? "◐" : "○";
        const color = modelState === "done" ? theme.success : modelState === "failed" ? theme.danger : modelState === "loading" ? theme.warning : theme.dim;
        return <React.Fragment key={name}><Text color={color}>{glyph} {name}</Text>{index < names.length - 1 && <Text color={theme.border}>──</Text>}</React.Fragment>;
      })}
    </Box>
  );
}

function formatTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function EventLine({ event }: { event: EngineEvent }) {
  const prefix = <Text color={theme.dim}>{formatTime(event.ts)}  </Text>;
  if (event.kind === "heard" || event.kind === "user_msg") return <Text>{prefix}<Text color={theme.accent}>YOU  </Text><Text color={theme.primary}>{textField(event, "text")}</Text></Text>;
  if (event.kind === "llm") return <Text>{prefix}<Text color={theme.warning}>RUN  </Text><Text color={theme.primary}>{textField(event, "tool")}</Text><Text color={theme.dim}> {textField(event, "args")}</Text></Text>;
  if (event.kind === "tool_result") return <Text>{prefix}<Text color={theme.success}>DONE </Text><Text color={theme.muted}>{textField(event, "name")} · {textField(event, "out")}</Text></Text>;
  if (event.kind === "rejected") return <Text>{prefix}<Text color={theme.warning}>SKIP </Text><Text color={theme.muted}>{textField(event, "reason")}</Text></Text>;
  if (["error", "tool_error", "tool_fail", "backend_crash"].includes(event.kind)) return <Text>{prefix}<Text color={theme.danger}>ERR  </Text><Text color={theme.danger}>{textField(event, "message") || textField(event, "msg") || textField(event, "out")}</Text></Text>;
  if (event.kind === "backend_log") return <Text>{prefix}<Text color={theme.dim}>SYS  {textField(event, "message")}</Text></Text>;
  return null;
}

function Activity({ state, availableRows }: { state: CozyState; availableRows: number }) {
  const visible = state.events.slice(-Math.max(1, availableRows));
  return (
    <Box flexDirection="column" flexGrow={1} overflow="hidden" paddingX={1}>
      <Text color={theme.dim}>ACTIVITY</Text>
      {visible.length === 0
        ? <Text color={theme.dim}>No commands yet. Type below or say “Hey Cozy”.</Text>
        : visible.map((event, index) => <EventLine key={`${event.ts}-${event.kind}-${index}`} event={event} />)}
    </Box>
  );
}

function Meter({ level }: { level: number }) {
  const width = 18;
  const count = Math.round(level * width);
  return <Text color={theme.accent}>{"▰".repeat(count)}<Text color={theme.border}>{"▱".repeat(width - count)}</Text></Text>;
}

function FocusCard({ state }: { state: CozyState }) {
  let label = "TIP";
  let body: React.ReactNode = <>Type a command at any time. Press <Text bold>Ctrl+R</Text> to restart the engine.</>;
  let color: string = theme.muted;
  if (state.phase === "listening") { label = "LISTENING"; color = theme.accent; body = <><Meter level={state.audioLevel} />  say your command</>; }
  if (state.phase === "capturing") { label = "CAPTURING"; color = theme.accent; body = <><Meter level={state.audioLevel} />  transcribing your voice…</>; }
  else if (state.phase === "thinking") { label = "WORKING"; color = theme.warning; body = state.transcript || "Choosing the best action…"; }
  else if (state.phase === "speaking") { label = "COZY"; color = theme.accent; body = state.response || "Speaking…"; }
  else if (state.response) { label = "COZY"; color = theme.success; body = state.response; }
  else if (state.fatalError) { label = "ENGINE STOPPED"; color = theme.danger; body = <>{state.fatalError}  Press <Text bold>Ctrl+R</Text> to retry.</>; }
  return <Box borderStyle="round" borderColor={color} paddingX={1}><Text color={color} bold>{label}  </Text><Text color={theme.primary} wrap="truncate-end">{body}</Text></Box>;
}

function LoadingScreen({ state, frame }: { state: CozyState; frame: number }) {
  const names: ModelName[] = ["wake", "stt", "llm", "tts"];
  return <Box flexDirection="column" alignItems="center" justifyContent="center" flexGrow={1}>
    <Text color={theme.primary} bold>{CAT_FRAMES[frame % CAT_FRAMES.length]}</Text>
    <Text color={theme.accent} bold>{SPINNER[frame % SPINNER.length]}  Loading Cozy models…</Text>
    <Text color={theme.dim}>Please wait until the pipeline is ready.</Text>
    <Box marginTop={1} flexDirection="column">
      {names.map(name => <Text key={name} color={state.models[name] === "done" ? theme.success : state.models[name] === "failed" ? theme.danger : theme.warning}>
        {state.models[name] === "done" ? "●" : state.models[name] === "failed" ? "×" : "◐"} {name}
      </Text>)}
    </Box>
  </Box>;
}

export function App({ eventSource, send, restart, stop }: AppProps) {
  const [state, dispatch] = useReducer(reduceEvent, initialState);
  const [input, setInput] = useState("");
  const [frame, setFrame] = useState(0);
  const [notice, setNotice] = useState("");
  const { exit } = useApp();
  const { rows } = useTerminalSize();

  useEffect(() => eventSource.subscribe(dispatch), [eventSource]);
  useEffect(() => {
    const active = ["starting", "listening", "capturing", "thinking", "speaking"].includes(state.phase);
    const timer = setInterval(() => setFrame(value => value + 1), active ? 140 : 650);
    return () => clearInterval(timer);
  }, [state.phase]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 2500);
    return () => clearTimeout(timer);
  }, [notice]);

  useInput((character, key) => {
    if (key.ctrl && character === "c") { stop(); exit(); return; }
    if (key.ctrl && character === "r") { setNotice("Restarting engine…"); restart(); return; }
    if (key.escape) { setInput(""); return; }
    if (key.return) {
      const command = input.trim();
      if (!command) return;
      if (send(command)) { dispatch({ kind: "user_msg", text: command, ts: Date.now() / 1000 }); setInput(""); }
      else setNotice("Engine is unavailable — press Ctrl+R to retry");
      return;
    }
    if (key.backspace || key.delete) { setInput(value => value.slice(0, -1)); return; }
    if (character && !key.ctrl && !key.meta) setInput(value => value + character);
  });

  const activityRows = useMemo(() => Math.max(2, rows - 16), [rows]);
  if (state.phase === "starting") return <Box flexDirection="column" paddingX={1} height={rows}><LoadingScreen state={state} frame={frame} /></Box>;
  return (
    <Box flexDirection="column" paddingX={1} height={rows}>
      <Header state={state} frame={frame} />
      <Pipeline state={state} />
      <Activity state={state} availableRows={activityRows} />
      <FocusCard state={state} />
      <Box marginTop={1} borderStyle="round" borderColor={theme.accent} paddingX={1}>
        <Text color={theme.accent} bold>› </Text><Text color={theme.primary}>{input}</Text><Text color={theme.accent}>█</Text>
      </Box>
      <Box justifyContent="space-between"><Text color={notice ? theme.warning : theme.dim}>{notice || "Enter send · Esc clear"}</Text><Text color={theme.dim}>Ctrl+R restart · Ctrl+C quit</Text></Box>
    </Box>
  );
}
