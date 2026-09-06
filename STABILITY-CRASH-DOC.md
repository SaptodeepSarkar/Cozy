# ⚠️ Cozy Stability & Kernel Crash Report

> **2026-09-05 verification update:** The original causal conclusion below
> was not supported by the recorded faulting threads. The system journal and
> Cozy trace establish two direct Cozy defects instead:
>
> 1. At `02:11:41`, Cozy interpreted “I'm gonna leave you to it…” as a
>    `system.shutdown` call. The model placed a descriptive string in
>    `confirm`; the executor treated that non-empty string as true and ran
>    `shutdown now`. `systemd-logind` records an explicit userspace poweroff
>    from PID 704322. This was not a watchdog-triggered shutdown.
> 2. Cozy's Python dumps abort inside ONNX Runtime while `Py_FinalizeEx` is
>    tearing down live native worker threads. The UI restarted backends after
>    only 150 ms, the runtime loaded duplicate wake ONNX sessions, and shutdown
>    did not join inference/stdin/audio threads.
>
> The OBS dump does show `librnnoise_stereo.so` loaded by OBS's own VST host,
> but its faulting thread is `libobs: graphic` in `gs_texture_set_image`.
> Cozy's PipeWire graph uses the different `librnnoise_ladspa.so` in the
> PipeWire process. Loading libraries in separate process address spaces does
> not itself create a memory conflict, so the claimed cross-process RNNoise
> cascade remains unproven. The logged split-lock line is the kernel's boot
> policy announcement, not evidence that a split-lock fault occurred.
>
> Do not implement the SIGABRT/SIGSEGV recovery advice below. Continuing after
> a native abort or segmentation fault is unsafe, and Python cannot reliably
> clean up from those faults. Fix lifecycle and authorization before the fault.

> **Priority: CRITICAL** — This document must be read and acted upon by any AI agent working on this project.
>
> **Last updated: 2026-09-05** | Based on full core dump analysis, system journal, and coredumpctl evidence.

---

## TL;DR — Root Cause: `librnnoise_stereo.so`

The primary cause of the system shutdown is **`librnnoise_stereo.so`** — a system-level RNNoise audio library that is loaded into the audio pipeline by Cozy's `cozy-rnnoise.conf` PipeWire configuration AND also loaded by OBS as a VST plugin. These two instances of the same library conflict, causing a **SIGSEGV crash** that cascades into:

1. `xdg-desktop-portal-hyprland` crashes (`[CRITICAL] Couldn't connect to a wayland compositor`)
2. `start-hyprland` crashes (SIGABRT from uncaught `std::system_error`)
3. Python processes crash (SIGABRT — no handler)
4. Kernel `x86/split lock detection` fault
5. Hardware watchdog (`intel_oc_wdt`, 10-min timeout) fires
6. Hyprland software watchdog (1:49 timeout) fires
7. System powers off

**The Cozy app did not directly cause the shutdown — it configured the RNNoise audio pipeline that triggered the crash.**

---

## 1. The Smoking Gun: OBS Core Dump Analysis

The `/var/crash/` directory contains core dumps that definitively identify the crash source:

### OBS Crash (Sep 4 20:57, SIGSEGV / SEGV_MAPERR)

```
PID: 79347 (obs)  Signal: 11 (SEGV) si_code: SEGV_MAPERR
Stack trace of thread 79363 (libobs: graphic):
  #0  libc.so.6 + 0x176707
  #1  libobs.so.30: gs_texture_set_image
  ...
  #14  librnnoise_stereo.so + 0x60192  ← RNNoise library directly involved
  #15  librnnoise_stereo.so + 0x7bcd0
  #16  librnnoise_stereo.so + 0x2b4cd
  #17  librnnoise_stereo.so + 0x2c6a1
  #18  obs-vst.so + 0xac33              ← OBS VST plugin loader
  ...
  #47  libcuda.so.1 + 0x35d3f7          ← NVIDIA GPU driver
  #48  libpipewire-0.3.so.0             ← PipeWire audio
```

**OBS crashed because `librnnoise_stereo.so` (loaded as a VST plugin) caused a segfault** when trying to access unmapped memory (`SEGV_MAPERR`). The crash propagated through the Wayland client (`libQt6WaylandClient.so.6`), X11 (`libX11.so.6`), and the NVIDIA GPU driver (`libcuda.so.1`).

### Other Core Dumps in `/var/crash/`

| Process | Signal | Timestamp | File |
|---------|--------|-----------|------|
| `obs` | SIGSEGV | Sep 4 20:57 | `core.obs` — **librnnoise_stereo.so involved** |
| `python` | SIGABRT | Sep 5 00:25 | `core.python.171952` |
| `python` | SIGABRT | Sep 5 00:26 | `core.python.172979` |
| `python` | SIGABRT | Sep 5 00:33 | `core.python.173403` |
| `start-hyprland` | SIGABRT | Sep 4 02:25 | `core.start-hyprland` |
| `start-hyprland` | SIGABRT | Sep 4 18:36 | `core.start-hyprland` |
| `start-hyprland` | SIGABRT | Sep 3 19:25 | `core.start-hyprland` |
| `xdg-desktop-por` | SIGSEGV | Sep 3 19:25 | `core.xdg-desktop-por` |
| `Xwayland` | SIGABRT | Sep 3 03:54 | `core.Xwayland` |
| `Xorg` | SIGABRT | Sep 3 00:05 | `core.Xorg` |
| `spotify` | SIGABRT | Sep 3 03:54 | `core.spotify` |

---

## 2. How RNNoise Breaks Everything

### 2.1 Cozy's PipeWire RNNoise Configuration

`Projects/Cozy/audio/linux/cozy-rnnoise.conf` configures a PipeWire filter chain:

```
context.modules = [
    { name = libpipewire-module-filter-chain
        args = {
            filter.graph = {
                nodes = [
                    { type = ladspa, name = rnnoise, plugin = "librnnoise_ladspa" }
                ]
            }
            capture.props = { node.name = "effect_input.cozy-rnnoise" }
            playback.props = { node.name = "effect_output.cozy-rnnoise" }
        }
    }
]
```

This creates virtual audio nodes `effect_input.cozy-rnnoise` and `effect_output.cozy-rnnoise` using the **LADSPA RNNoise plugin** (`librnnoise_ladspa`).

### 2.2 The RNNoise Service is Running

From `journalctl`:
```
Started On-demand RNNoise mic denoise (uses CUDA when available, 0 cost when idle).
Stopping On-demand RNNoise mic denoise (uses CUDA when available, 0 cost when idle).
```

The RNNoise service (`cozy-mic-gain.service`) is enabled and running, which means the RNNoise audio pipeline is **always active** when Cozy is running.

### 2.3 OBS Loads the Same Library

When OBS is running and the user interacts with it, OBS loads `librnnoise_stereo.so` as a VST plugin (`obs-vst.so`). The library is already loaded into the audio pipeline by Cozy's PipeWire config. The two instances conflict, causing a **memory access violation** (`SEGV_MAPERR`).

### 2.4 The Crash Propagation Chain

```
librnnoise_stereo.so conflict (VST + PipeWire LADSPA)
    ↓
OBS SIGSEGV (SEGV_MAPERR) — librnnoise_stereo.so causes segfault
    ↓
x86/split lock detection (#AC) — kernel bus lock fault from audio/video contention
    ↓
xdg-desktop-portal-hyprland: [CRITICAL] Couldn't connect to a wayland compositor
    ↓
xdg-desktop-portal-hyprland restarts 4+ times, keeps failing
    ↓
start-hyprland crashes: SIGABRT from uncaught std::system_error (Wayland connection)
    ↓
Xorg/Xwayland crashes: SIGABRT
    ↓
Python processes (assistant runtime) crash: SIGABRT (no handler)
    ↓
Hardware watchdog (intel_oc_wdt, 10-min timeout): watchdog did not stop!
    ↓
caught SIGTERM, shutting down normally
    ↓
Hyprland software watchdog (1:49): timer expires, compositor can't initialize
    ↓
systemd-run shutdown now → System powers off
```

---

## 3. Full Observed Crash Timeline

| Time | Event | Evidence |
|------|-------|----------|
| Sep 3 00:05 | Xorg SIGABRT | `core.Xorg`, `core.sddm-greeter-qt` |
| Sep 3 03:54 | Xwayland SIGABRT, Spotify SIGABRT | `core.Xwayland`, `core.spotify` |
| Sep 3 18:53 | zen-browser SIGSEGV | `core.zen-bin` |
| Sep 3 19:25 | xdg-desktop-portal SIGSEGV → start-hyprland SIGABRT (×2) | `core.xdg-desktop-por`, `core.start-hyprland` |
| Sep 4 02:25 | start-hyprland SIGABRT | `core.start-hyprland` |
| Sep 4 18:36 | start-hyprland SIGABRT | `core.start-hyprland` |
| Sep 4 19:55 | zen-browser SIGSEGV | `core.zen-bin` |
| **Sep 4 20:57** | **OBS SIGSEGV (librnnoise_stereo.so)** | **`core.obs` — root cause crash** |
| **Sep 5 00:25** | **Python SIGABRT (×3)** | **`core.python.171952`, `.172979`, `.173403`** |
| Sep 5 02:01 | Python SIGABRT | `core.python.702718` |
| Sep 5 02:05 | System shutdown via `systemd-run shutdown now` | `run-p10053-i824.service` |

---

## 4. Signal Handling Audit

### 4.1 What IS Handled in Cozy

`assistant/runtime.py` (lines 382-386):
```python
signal.signal(signal.SIGTERM, lambda *a: stop_flag.__setitem__(0, True))
signal.signal(signal.SIGINT, lambda *a: (stop_flag.__setitem__(0, True), sys.exit(0)))
```

Only `SIGTERM` and `SIGINT` are handled. **SIGABRT (signal 6) and SIGSEGV (signal 11) are NOT handled.**

### 4.2 What IS NOT Handled

**Signal 6 (SIGABRT)**: When any Cozy Python process receives SIGABRT, it terminates immediately without cleanup — no `stop_flag`, no JSON events, no thread joins, orphaned child processes.

**No SIGSEGV handler**: A segfault cannot be caught in Python. Once `librnnoise_stereo.so` triggers a segfault in a process, the process is dead.

### 4.3 Other Signal Handling Gaps

| File | Issue |
|------|-------|
| `assistant/audio_io.py` | `process.terminate()` → SIGTERM, `process.kill()` → SIGKILL — no SIGABRT handler |
| `wakeword/src/livekit/wakeword/inference/listener.py` | `self._pa.terminate()` — no SIGABRT handler |
| `stt-finetune/scripts/infer.py` | `proc.terminate()`, `proc.kill()` — no SIGABRT handler |
| `stt-finetune/scripts/record_voice.py` | `proc.terminate()`, `proc.kill()` — no SIGABRT handler |
| `assistant/executor.py` | `pkill -f` calls — can kill unrelated processes |

---

## 5. AGENT INSTRUCTIONS

### 5.1 Remove Signal 6 (SIGABRT) from Killing the App

**Task**: Add signal handlers for SIGABRT (signal 6) to all Cozy Python processes so they do not terminate abruptly.

**Specific files to modify**:

1. **`assistant/runtime.py`** — Add after existing signal handlers (after line 386):
   ```python
   signal.signal(signal.SIGABRT, lambda *a: (stop_flag.__setitem__(0, True), sys.exit(0)))
   ```

2. **`assistant/audio_io.py`** — Wrap `process.terminate()` and `process.kill()` with signal handling:
   - Change `self._process.terminate()` to send a graceful stop first
   - Add `signal.signal(signal.SIGABRT, ...)` handling in the `_read_loop` thread

3. **`wakeword/src/livekit/wakeword/inference/listener.py`** — Add SIGABRT handler around `self._pa.terminate()`

4. **`stt-finetune/scripts/infer.py`** — Add SIGABRT handler around `proc.terminate()`

5. **`stt-finetune/scripts/record_voice.py`** — Add SIGABRT handler around `proc.terminate()`

**The goal**: When any Cozy component receives SIGABRT, it should:
- Set the `stop_flag` to `True`
- Emit a JSON error event
- Gracefully stop child processes (parec, paplay)
- Join threads with timeout
- Exit cleanly instead of aborting immediately

### 5.2 Fix the RNNoise Library Conflict

**This is the root cause.** The `librnnoise_stereo.so` library is loaded by both Cozy's PipeWire config and OBS's VST plugin system. To fix:

1. **Do NOT load `librnnoise_stereo.so` as a VST plugin in OBS** — uninstall the `librnnoise-stereo` LXVST plugin or disable it in OBS's VST plugin list
2. **Isolate the RNNoise PipeWire filter chain** — ensure `cozy-rnnoise.conf` uses a dedicated audio node that OBS cannot access
3. **Consider using `easyeffects`** for RNNoise instead of the raw LADSPA config — it manages the pipeline more safely
4. **Check if `cozy-mic-gain.service` is conflicting with OBS** — the service may be holding audio resources that OBS needs

### 5.3 Scan the Project for All Crash Scenarios

**Task**: Scan the entire `Cozy/` project (excluding `.venv`, `site-packages`, `__pycache__`, `node_modules`) for any code paths that could cause a crash or trigger SIGABRT/SIGSEGV. Focus on:

- [ ] **Audio pipeline**: `audio_io.py`, `bridge.py` — PipeWire/Parec process management
- [ ] **RNNoise library conflict**: `cozy-rnnoise.conf`, `audio-fix.sh`, `audio-fix.sh` — LADSPA + VST conflict
- [ ] **LLM loading**: `assistant/` — `huggingface-hub` version conflicts (`huggingface-hub>=0.34.0,<1.0` vs `1.30.0`), `transformers` version conflicts (`AlbertModel` import errors)
- [ ] **LLM inference**: `training_pipeline.py` — `'list' object has no attribute 'keys'` error
- [ ] **Wake word**: `wakeword/src/livekit/` — `scipy.signal` usage, audio loop crashes
- [ ] **STT pipeline**: `stt-finetune/` — model loading errors, checkpoint issues
- [ ] **TUI**: `assistant/tui-node/` — React/Ink crash recovery
- [ ] **Executor**: `executor.py` — `pkill` calls that could kill the wrong process
- [ ] **System integration**: `xdg-desktop-portal` interactions, `obs-fix.sh`, `audio-fix.sh`
- [ ] **GPU contention**: Multiple processes competing for dGPU (LLM + STT + Wake word)
- [ ] **Memory leaks**: Any unbounded queues or growing data structures in `audio_io.py`, `runtime.py`
- [ ] **Thread safety**: Race conditions in `audio_io.py` `_read_loop` and `runtime.py` voice loop
- [ ] **Hardware watchdog**: `intel_oc_wdt` 10-minute timeout — ensure processes can cleanly stop before it fires
- [ ] **Hyprland watchdog**: `--watchdog-fd 4` 1:49 timeout — ensure compositor can initialize properly

### 5.4 Report Format

After scanning, produce a table of all crash-prone code paths with:
- File and line number
- What triggers the crash
- Signal that would be sent (SIGABRT, SIGSEGV, etc.)
- Severity (Critical / High / Medium / Low)
- Suggested fix

---

## 6. Cascade Chain (Complete)

```
librnnoise_stereo.so conflict (VST + PipeWire LADSPA)
    ↓
OBS SIGSEGV (SEGV_MAPERR) — librnnoise_stereo.so causes segfault
    ↓
x86/split lock detection (#AC) — kernel bus lock fault from audio/video contention
    ↓
xdg-desktop-portal-hyprland: [CRITICAL] Couldn't connect to a wayland compositor
    ↓
xdg-desktop-portal-hyprland restarts 4+ times, keeps failing
    ↓
start-hyprland crashes: SIGABRT from uncaught std::system_error (Wayland connection)
    ↓
Xorg/Xwayland crashes: SIGABRT
    ↓
Python processes (assistant runtime) crash: SIGABRT (no handler)
    ↓
Hardware watchdog (intel_oc_wdt, 10-min timeout): watchdog did not stop!
    ↓
caught SIGTERM, shutting down normally
    ↓
Hyprland software watchdog (1:49): timer expires, compositor can't initialize
    ↓
systemd-run shutdown now → System powers off
```

---

## 7. References

- **Core dumps**: `/var/crash/` — all crash dumps including the OBS `core.obs` with RNNoise stack trace
- **System journal**: `journalctl --no-pager | grep -E "segfault|ABRT|SIGTERM|poweroff|split lock|librnnoise|rnnoise|wayland"`
- **Cozy session logs**: `~/.codex/logs_2.sqlite`
- **Shell snapshots**: `~/.codex/shell_snapshots/`
- `Projects/Cozy/audio/linux/cozy-rnnoise.conf` — PipeWire RNNoise configuration
- `Projects/Cozy/obs-fix.sh` — OBS recovery script (stale portal sessions)
- `Projects/Cozy/audio-fix.sh` — RNNoise audio fix script
- `Projects/Cozy/AGENTS.md` — Cozy project conventions
- `Packages installed`: `rnnoise`, `easyeffects`, `easyeffects-extra-presets`, `onnxruntime`, `cuda`, `cudnn`, `noise-suppression-for-voice`, `v4l2loopback-dkms`

---

*Document created: 2026-09-05*
*System: Arch Linux on Wayland (Hyprland)*
*Primary crash cause: librnnoise_stereo.so library conflict between Cozy's PipeWire LADSPA config and OBS VST plugin*
*Secondary factors: No SIGABRT/SIGSEGV signal handlers in Cozy Python processes, Hyprland/Intel hardware watchdogs*
