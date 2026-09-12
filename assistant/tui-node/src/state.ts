import type { EngineEvent, ModelName, ModelState } from "./protocol.ts";
import { numberField, textField } from "./protocol.ts";

export type Phase = "starting" | "ready" | "listening" | "capturing" | "transcribing" | "thinking" | "speaking" | "error";

export interface CozyState {
  phase: Phase;
  models: Record<ModelName, ModelState>;
  events: EngineEvent[];
  audioLevel: number;
  audioHistory: number[];
  voiceEnabled: boolean;
  micMuted: boolean | null;
  transcript: string;
  response: string;
  fatalError: string;
  loadingModel: ModelName | "";
  loadingStartedAt: number;
  listeningStartedAt: number;
  modelElapsed: Partial<Record<ModelName, number>>;
  startupProgress: number;
  hasStarted: boolean;
}

export const initialState: CozyState = {
  phase: "starting",
  models: { wake: "pending", stt: "pending", llm: "pending", foxmcp: "pending", cleanup: "pending", tts: "pending" },
  events: [],
  audioLevel: 0,
  audioHistory: [],
  voiceEnabled: true,
  micMuted: null,
  transcript: "",
  response: "",
  fatalError: "",
  loadingModel: "",
  loadingStartedAt: 0,
  listeningStartedAt: 0,
  modelElapsed: {},
  startupProgress: 0,
  hasStarted: false,
};

const loggedKinds = new Set([
  "backend_crash", "error", "heard", "llm", "rejected",
  "tool_error", "tool_fail", "tool_result", "user_msg", "audio_profile", "done",
]);

const withEvent = (state: CozyState, event: EngineEvent): CozyState =>
  loggedKinds.has(event.kind)
    ? { ...state, events: [...state.events, event].slice(-250) }
    : state;

export function reduceEvent(state: CozyState, event: EngineEvent): CozyState {
  switch (event.kind) {
    case "backend_start":
      return initialState;
    case "warmup": {
      const model = textField(event, "model") as ModelName;
      const modelState = textField(event, "state") as ModelState;
      if (!(model in state.models) || !["pending", "loading", "done", "failed"].includes(modelState)) return state;
      const elapsed = numberField(event, "elapsed_s");
      const progress = numberField(event, "progress");
      return {
        ...state,
        models: { ...state.models, [model]: modelState },
        loadingModel: modelState === "loading" ? model : state.loadingModel === model ? "" : state.loadingModel,
        loadingStartedAt: modelState === "loading" ? event.ts : state.loadingStartedAt,
        modelElapsed: modelState === "done" || modelState === "failed"
          ? { ...state.modelElapsed, [model]: elapsed }
          : state.modelElapsed,
        startupProgress: Number.isFinite(progress) ? Math.max(0, Math.min(1, progress)) : state.startupProgress,
      };
    }
    case "ready":
      return { ...state, phase: "ready", fatalError: "", hasStarted: true, listeningStartedAt: 0, voiceEnabled: typeof event.voice === "boolean" ? event.voice : state.voiceEnabled };
    case "audio_status":
      return { ...state, micMuted: typeof event.muted === "boolean" ? event.muted : null };
    case "wake_score":
      return state; // Wake confidence is not microphone amplitude.
    case "wake":
      return withEvent({ ...state, phase: "listening", audioHistory: [], audioLevel: 0, transcript: "", listeningStartedAt: event.ts }, event);
    case "stt_start":
      return { ...state, phase: "capturing", transcript: "Listening…" };
    case "capture_level": {
      const raw = numberField(event, "level");
      const level = Number.isFinite(raw) ? Math.max(0, Math.min(1, raw)) : 0;
      return { ...state, audioLevel: level, audioHistory: [...state.audioHistory, level].slice(-41) };
    }
    case "stt_infer":
      return { ...state, phase: "transcribing", audioLevel: 0, transcript: "Transcribing…" };
    case "transcribed":
      return withEvent({ ...state, phase: "thinking", transcript: textField(event, "text") }, event);
    case "user_msg":
    case "heard":
      if (event.kind === "heard" && state.events.at(-1)?.kind === "user_msg" &&
          textField(state.events.at(-1)!, "text") === textField(event, "text")) return state;
      return withEvent({ ...state, phase: "thinking", transcript: textField(event, "text"), response: "" }, event);
    case "llm":
      return withEvent({ ...state, phase: "thinking" }, event);
    case "llm_text":
      return { ...state, phase: "thinking", response: textField(event, "text") };
    case "tts":
      return { ...state, phase: "speaking", response: textField(event, "text") || state.response };
    case "tts_start":
      return { ...state, phase: "speaking" };
    case "tts_done":
      return { ...state, phase: "ready", audioLevel: 0 };
    case "done":
      return withEvent({ ...state, phase: state.phase === "speaking" ? "speaking" : "ready", response: textField(event, "text"), audioLevel: 0 }, event);
    case "rejected":
      return withEvent({ ...state, phase: "ready", audioLevel: 0 }, event);
    case "tool_result":
    case "tool_fail":
    case "tool_error":
      return withEvent(state, event);
    case "error":
      // Runtime errors are recoverable and remain in the conversation. Only
      // a backend crash makes the whole interface enter the fatal state.
      return withEvent(state, event);
    case "startup_failed": {
      const message = textField(event, "msg") || "A required model could not be initialized.";
      return withEvent({ ...state, phase: "error", fatalError: message }, event);
    }
    case "backend_crash": {
      const message = textField(event, "message") || "The assistant engine stopped unexpectedly.";
      return withEvent({ ...state, phase: "error", fatalError: message }, event);
    }
    default:
      return state;
  }
}

const BARS = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];

/** Render mic levels (0..1) as a flat terminal waveform, e.g. `▁▃▅█▃▁`. */
export function barsForLevels(levels: number[], columns = 24): string {
  const tail = levels.slice(-columns);
  const pad = "▁".repeat(Math.max(0, columns - tail.length));
  return pad + tail.map(raw => {
    const level = Number.isFinite(raw) ? Math.max(0, Math.min(1, raw)) : 0;
    return BARS[Math.min(BARS.length - 1, Math.floor(level * BARS.length))];
  }).join("");
}

/** Format an elapsed duration as `MM:SS` for the listening status line. */
export function formatElapsed(startedAt: number, now: number): string {
  const secs = Math.max(0, Math.floor(now - startedAt));
  return `${String(Math.floor(secs / 60)).padStart(2, "0")}:${String(secs % 60).padStart(2, "0")}`;
}
