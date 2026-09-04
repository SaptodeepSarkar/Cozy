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
