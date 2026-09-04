import type { EngineEvent, ModelName, ModelState } from "./protocol.js";
import { numberField, textField } from "./protocol.js";

export type Phase = "starting" | "ready" | "listening" | "capturing" | "thinking" | "speaking" | "error";

export interface CozyState {
  phase: Phase;
  models: Record<ModelName, ModelState>;
  events: EngineEvent[];
  audioLevel: number;
  transcript: string;
  response: string;
  fatalError: string;
}

export const initialState: CozyState = {
  phase: "starting",
  models: { wake: "pending", stt: "pending", llm: "pending", tts: "pending" },
  events: [],
  audioLevel: 0,
  transcript: "",
  response: "",
  fatalError: "",
};

const loggedKinds = new Set([
  "backend_crash", "error", "heard", "llm", "rejected",
  "tool_error", "tool_fail", "tool_result", "user_msg",
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
      return { ...state, models: { ...state.models, [model]: modelState } };
    }
    case "ready":
      return { ...state, phase: "ready", fatalError: "" };
    case "wake_score":
      return { ...state, audioLevel: Math.max(0, Math.min(1, numberField(event, "score"))) };
    case "wake":
      return withEvent({ ...state, phase: "listening", transcript: "", response: "" }, event);
    case "stt_start":
      return { ...state, phase: "capturing", transcript: "Listening…", response: "" };
    case "capture_level":
      return { ...state, audioLevel: Math.max(0, Math.min(1, numberField(event, "level"))) };
    case "transcribed":
      return withEvent({ ...state, phase: "thinking", transcript: textField(event, "text") }, event);
    case "heard":
      return withEvent({ ...state, phase: "thinking", transcript: textField(event, "text"), response: "" }, event);
    case "llm":
      return withEvent({ ...state, phase: "thinking" }, event);
    case "llm_text":
      return { ...state, phase: "thinking", response: textField(event, "text") };
    case "tts":
      return { ...state, phase: "speaking" };
    case "done":
      return { ...state, phase: "ready", response: textField(event, "text"), audioLevel: 0 };
    case "rejected":
      return withEvent({ ...state, phase: "ready", audioLevel: 0 }, event);
    case "tool_result":
    case "tool_fail":
    case "tool_error":
      return withEvent(state, event);
    case "error":
      return withEvent({ ...state, phase: "error" }, event);
    case "backend_crash": {
      const message = textField(event, "message") || "The assistant engine stopped unexpectedly.";
      return withEvent({ ...state, phase: "error", fatalError: message }, event);
    }
    default:
      return state;
  }
}
