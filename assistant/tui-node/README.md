# Cozy terminal UI

The default Cozy interface is a React + Ink application written in strict
TypeScript/TSX. It supervises the Python voice runtime over an NDJSON protocol;
the UI therefore remains usable when the runtime reports an error or exits.

```bash
npm ci
npm run check
npm start -- --threshold 0.5
```

At runtime, press `Enter` to send a typed command, `Esc` to clear it, `Ctrl+R`
to restart the Python engine, or `Ctrl+C` to shut down both processes. Set
`COZY_PYTHON` or `COZY_RUNTIME` to override the backend paths for development.

The interface keeps one conversation and a persistent composer. During capture,
a rolling waveform uses real microphone RMS events, not wake confidence or
simulated audio. Transcription and processing use a separate activity indicator;
speaking displays the response. Results remain in the conversation. Long replies
wrap, and `PgUp` / `PgDn` browse the retained session history.

Use `/details` (or `Ctrl+D`) for engine status, `/help` (`Ctrl+G`) for shortcuts,
`/restart` to restart, and `/motion` to toggle reduced motion. Set
`COZY_REDUCED_MOTION=1` to disable decorative animation from launch; measured audio
feedback remains live. Edit text with arrow keys and `Ctrl+A` / `Ctrl+E`.
Layouts adapt to terminal size, with a minimum of 32 columns by 12 rows.

Preview the complete listening → transcription → response sequence without
loading models, opening the microphone, or executing tools:

```bash
npm run preview
```

Preview waveforms are synthetic demonstration data. The production UI renders
only measured microphone samples. Design references: [Prime Agent](https://github.com/PrimeIntellect-ai/prime-agent),
[OpenCode](https://opencode.ai/docs/tui/), and [Hermes](https://hermes-agent.nousresearch.com/docs/user-guide/cli).

Runtime diagnostics: a muted input now shows “Microphone muted” and typed requests
remain usable. Audio-stream failures surface as errors. `cozy --text --no-tts`
starts the same UI with only the language model; no wake/STT or audio capture is
loaded. In voice mode the wake detector keeps a rolling two-second window and
scores every 160 ms with one non-spinning ONNX worker per session.
