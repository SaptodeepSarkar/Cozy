import React from "react";
import assert from "node:assert/strict";
import test from "node:test";
import { setTimeout as settle } from "node:timers/promises";
import { render } from "ink-testing-library";
import { App } from "./app.js";
import type { EngineEvent } from "./protocol.js";

// Exercise the actual Ink renderer and keyboard bindings without loading models.
test("conversation, voice state, editing, diagnostics and narrow layouts", async () => {
  let receive: (event: EngineEvent) => void = () => {};
  const sent: string[] = [];
  const ui = render(<App eventSource={{ subscribe: listener => { receive = listener; return () => {}; } }}
    send={text => { sent.push(text); return true; }} restart={() => {}} stop={() => {}} />);
  try {
    await settle(30);
    receive({ kind: "ready", ts: 1 });
    await settle(30);
    assert.match(ui.lastFrame()!, /Ready when you are/);
    ui.stdin.write("helo"); await settle(30);
    ui.stdin.write("\u001b[D"); await settle(30);
    ui.stdin.write("l"); await settle(30);
    ui.stdin.write("\r"); await settle(30);
    assert.deepEqual(sent, ["hello"]);
    receive({ kind: "heard", text: "hello", ts: 2 });
    receive({ kind: "done", text: "A response stays in your conversation.", ts: 3 });
    await settle(30);
    assert.match(ui.lastFrame()!, /A response stays/);
    receive({ kind: "wake", ts: 4 });
    receive({ kind: "stt_start", ts: 5 });
    receive({ kind: "capture_level", level: 1, ts: 6 });
    await settle(30);
    assert.match(ui.lastFrame()!, /I'm listening/);
    assert.match(ui.lastFrame()!, /█/);
    receive({ kind: "stt_infer", ts: 7 }); await settle(30);
    assert.match(ui.lastFrame()!, /Finding your words/);
    ui.stdin.write("/details"); await settle(30);
    ui.stdin.write("\r"); await settle(30);
    assert.match(ui.lastFrame()!, /wake/);
    Object.defineProperty(ui.stdout, "columns", { value: 40, configurable: true });
    ui.stdout.emit("resize"); await settle(30);
    assert.match(ui.lastFrame()!, /cozy/);
    assert.ok(ui.lastFrame()!.split("\n").length <= 24);
    receive({ kind: "backend_crash", message: "Engine offline", ts: 8 }); await settle(30);
    assert.match(ui.lastFrame()!, /Engine offline/);
  } finally { ui.unmount(); ui.cleanup(); }
});
