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

test("backend crashes remain visible and recover on restart", () => {
  const failed = reduceEvent(initialState, { kind: "backend_crash", message: "boom", ts: 1 });
  assert.equal(failed.phase, "error");
  assert.equal(failed.fatalError, "boom");
  assert.deepEqual(reduceEvent(failed, { kind: "backend_start", ts: 2 }), initialState);
});
