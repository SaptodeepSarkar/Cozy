# Cozy terminal UI

The default Cozy interface uses OpenTUI core with strict TypeScript. It
supervises the Python voice runtime over an NDJSON protocol and does not reveal
the main workspace until every required model is initialized.

```bash
npm ci
npm run check
npm start -- --threshold 0.5
```

At runtime, press `Enter` to send a typed command or `Ctrl+C` to shut down both
processes through the guarded single-signal path. Set
`COZY_PYTHON` or `COZY_RUNTIME` to override the backend paths for development.

The loading view shows the canonical SVG-derived logo, one model at a time,
elapsed seconds, and overall progress, including the input-polish model. The workspace keeps conversation output
stable across later wake events and provides a native focused input control.

Runtime diagnostics: a muted input now shows “Microphone muted”. Audio-stream failures
surface as errors. `cozy --no-tts` starts the same UI without spoken replies;
loaded. In voice mode the wake detector keeps a rolling two-second window,
scores every 160 ms, and confirms marginal activations across two scores.
