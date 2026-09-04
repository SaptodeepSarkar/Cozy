export type ModelName = "wake" | "stt" | "llm" | "tts";
export type ModelState = "pending" | "loading" | "done" | "failed";

export type EngineEvent = {
  kind: string;
  ts: number;
  [key: string]: unknown;
};

export type UiEvent = EngineEvent & {
  kind:
    | "backend_crash"
    | "backend_log"
    | "backend_start"
    | "protocol_error";
};

export function parseEngineEvent(line: string): EngineEvent | undefined {
  try {
    const value: unknown = JSON.parse(line);
    if (
      typeof value === "object" &&
      value !== null &&
      typeof (value as EngineEvent).kind === "string"
    ) {
      const event = value as EngineEvent;
      return { ...event, ts: typeof event.ts === "number" ? event.ts : Date.now() / 1000 };
    }
  } catch {
    // The caller reports malformed protocol lines without taking down the UI.
  }
  return undefined;
}

export const textField = (event: EngineEvent, field: string): string =>
  typeof event[field] === "string" ? event[field] : "";

export const numberField = (event: EngineEvent, field: string): number =>
  typeof event[field] === "number" ? event[field] : 0;
