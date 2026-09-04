# Cozy Linux audio routing

`cozy-audio-route` keeps the default sink/source aligned with a connected
Bluetooth headset. When no Bluetooth device is present it falls back to the
RNNoise filter output and the built-in analog sink.

The watcher intentionally polls every five seconds instead of using
`pactl subscribe`. PipeWire can emit a large burst of graph events while a
Bluetooth profile or filter node is negotiated; handling every event starts
many `pactl`/`awk` processes and can starve a desktop shell. Routing is
state-aware, so `pactl set-default-*` is only called when the desired default
actually differs.

Useful diagnostics:

```bash
systemctl --user status cozy-audio-route.service
systemctl --user show cozy-audio-route.service -p CPUUsageNSec
pactl get-default-sink
pactl get-default-source
```

To temporarily disable automatic routing without changing PipeWire itself:

```bash
systemctl --user disable --now cozy-audio-route.service
```

Run `bash audio-fix.sh` to reinstall the filter, gain service, and safe
polling router.
