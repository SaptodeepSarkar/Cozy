import assert from "node:assert/strict";
import test from "node:test";
import { initialState, reduceEvent } from "./state.js";

test("warmup and ready events update the pipeline", () => {
  const loading = reduceEvent(initialState, { kind: "warmup", model: "llm", state: "loading", ts: 1 });
  assert.equal(loading.models.llm, "loading");
  const ready = reduceEvent(loading, { kind: "ready", ts: 2 });
  assert.equal(ready.phase, "ready");
});

test("a completed answer returns the UI to ready", () => {
  const thinking = reduceEvent(initialState, { kind: "heard", text: "hello", ts: 1 });
  const done = reduceEvent(thinking, { kind: "done", text: "Hi!", ts: 2 });
  assert.equal(done.phase, "ready");
  assert.equal(done.response, "Hi!");
});

test("STT inference remains visibly active after capture", () => {
  const capturing = reduceEvent(initialState, { kind: "stt_infer", ts: 1 });
  assert.equal(capturing.phase, "transcribing");
  assert.equal(capturing.transcript, "Transcribing…");
});

test("backend crashes remain visible and recover on restart", () => {
  const failed = reduceEvent(initialState, { kind: "backend_crash", message: "boom", ts: 1 });
  assert.equal(failed.phase, "error");
  assert.equal(failed.fatalError, "boom");
  assert.deepEqual(reduceEvent(failed, { kind: "backend_start", ts: 2 }), initialState);
});

test("waveform uses finite microphone samples, never wake confidence", () => {
  let state = reduceEvent(initialState, { kind: "wake_score", score: 0.9, ts: 1 });
  assert.equal(state.audioLevel, 0);
  for (let i = 0; i < 80; i++) state = reduceEvent(state, { kind: "capture_level", level: 0.5, ts: i });
  assert.equal(state.audioHistory.length, 41);
  assert.equal(state.audioLevel, 0.5);
  state = reduceEvent(state, { kind: "capture_level", level: NaN, ts: 81 });
  assert.equal(state.audioLevel, 0);
  assert.deepEqual(reduceEvent(state, { kind: "wake", ts: 82 }).audioHistory, []);
});

test("typed backend echo is deduplicated and responses persist in conversation", () => {
  const typed = reduceEvent(initialState, { kind: "user_msg", text: "hello", ts: 1 });
  const heard = reduceEvent(typed, { kind: "heard", text: "hello", ts: 2 });
  assert.equal(heard.events.length, 1);
  const done = reduceEvent(heard, { kind: "done", text: "Hi!", ts: 3 });
  const next = reduceEvent(done, { kind: "heard", text: "time?", ts: 4 });
  assert.equal(next.events[1].text, "Hi!");
});

test("muted microphones are visible without blocking typed commands", () => {
  const muted = reduceEvent(initialState, { kind: "audio_status", muted: true, ts: 1 });
  assert.equal(muted.micMuted, true);
  const typing = reduceEvent(muted, { kind: "user_msg", text: "hello", ts: 2 });
  assert.equal(typing.phase, "thinking");
  const textMode = reduceEvent(typing, { kind: "ready", voice: false, ts: 3 });
  assert.equal(textMode.voiceEnabled, false);
});
