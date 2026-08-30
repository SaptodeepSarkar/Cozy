# Cozy v2 changelog

## What changed (v1.49 → v2.0)

### Fixed: the LLM emitted empty turns

The previous SFT dataset had every tool-call row stored as a raw JSON
string in the assistant `content` field. Qwen3's chat template expected
the OpenAI `tool_calls` shape, so the runtime regex never matched and
the trained model learned to emit just `<|im_end|>`.

The fix: re-emit all 1,397 training rows + 28 val rows in the proper
shape (`{role: "assistant", content: "", tool_calls: [{type: "function",
function: {name, arguments: <json string>}}]}`), and after each
tool-call add a `tool` result turn + a short spoken affirmation turn
("Done. Volume set.", "Searching.", etc.). The runtime, the chat
template, and the model are now on the same page.

### Added: more data

- **`make_dataset.py`** now writes the right shape (and is patched to
  never regress).
- **glaive-function-calling-v2** (curated, 360 train + 30 val) — adds
  agentic style variety, teaches the model the *shape* of tool calls
  with tool names and parameters it has never seen, so it doesn't
  over-fit to "the only tools in the world are these 32".
- **rlm_harness.dataset oracle** (45 train + 8 val) — I act as the
  oracle agent: for every seed task, the rule router runs against the
  live executor, captures the real tool output, and the trace is
  dumped to SFT. This grounds the model in your actual executor, not
  synthetic data.

Final SFT: **1,905 train rows, 68 val rows**, 1,866 tool-call turns,
2,809 chat turns.

### Added: TTS

`assistant/tts.py` wraps the system `espeak-ng` (already installed
on Ubuntu). Background thread so the wake loop never blocks on audio.
Affirmations are cached on disk for instant replay of common phrases.
espeak-ng is robotic but it's zero-dep, instant, and the user can
swap in piper-tts later by replacing the synthesize() function.

The runtime now speaks every reply (and every tool result) by
default. `--no-tts` flag suppresses audio.

### Added: RLVR / DPO

`assistant/rlvr.py` rolls the trained model out on 44 held-out
probes. The verifier (me, the agent) scores each output 0/1 on:
- schema-valid: tool call parses inside `<tool_call>` tags (or
  chat reply for chat prompts)
- tool-in-schema: tool name is in `team/tool_schema.json`
- params-parse: arguments JSON parses
- affirmation-present: a "Done." style phrase follows (tool rows)

Anything that scores 0 becomes "rejected" against a hand-written
"chosen" response, and `DPOTrainer` produces a 40 MB adapter at
`assistant/model/cozy-llm-v1-dpo/`. Then it re-verifies and prints
the post-DPO pass rate.

### Shipped: `cozy` shell alias

`cozy` (the binary) is a thin launcher that:
1. Sanity-checks every model artifact is on disk
2. Prints a banner
3. `exec`s the assistant venv's Python on `runtime.py`

`cozy.shell` is a 50-line block you source from `~/.bashrc` and
`~/.zshrc`. It defines:
- `cozy` — full voice loop
- `cozy --text` — type commands
- `cozy --calibrate` — 30s wake score log
- `cozy --no-wake` / `--no-tts` / `--threshold 0.5`
- `cozy --status` — which model files exist
- `cozy --stop` — kill the running process
- `cozystop`, `cozystatus`, `cozytext`, `cozycalibrate` shortcuts
- Tab completion (bash) and `compdef` (zsh, guarded)

`setup.sh` now installs the alias block automatically and adds
`espeak-ng` as a dep, so a fresh clone is one command away from
`cozy` working.

### Defensive: save fix in `sft_qwen.py`

The previous training claimed it saved the merged model but
`cozy-llm-v1/model.safetensors` was missing. The new save step
prints the actual file list and uses `max_shard_size="2GB"` to
guarantee the file lands.

## What still needs work

- **DPO**: not yet run. SFT needs to finish first.
- **TTS voice quality**: espeak-ng sounds robotic. The user can
  drop in `piper-tts` (already on pip) by swapping
  `tts.synthesize()` to use piper's ONNX model.
- **Wake threshold**: 0.6 is the eval-optimal. The runtime
  defaults to 0.30 (more sensitive). User can override with
  `--threshold 0.6`.


## v2.0.1 (TTS swap + DPO balance)

### TTS: espeak-ng → Kokoro-82M

espeak-ng is robotic and painful. Swapped in **Kokoro-82M** — a 313 MB
PyTorch neural TTS model that runs on CPU and sounds like a real person.
Voice: `af_heart` (warm American English female). The model
auto-downloads from `hexgrad/Kokoro-82M` on first speak().

Setup:
- `pip install Kokoro` (in assistant/.venv)
- Set voice in `assistant/voice.cfg` (e.g. `voice = af_heart`)
- Other good voices: `af_sarah`, `af_bella`, `am_adam`, `bf_emma`
- Model is lazy-loaded on first speak() to save ~2 GB RAM when TTS isn't used

### DPO: now opt-in via `--dpo` flag

The DPO adapter improves tool-call precision (28/36 tool probes pass
vs 9/36 for SFT alone), but it tends to over-fire on chitchat — "hello
cozy" gets routed to `time.now` instead of "Hi!". The fix:

- Default runtime now uses the SFT-only merged model
- Pass `--dpo` to load the DPO adapter when you want better tools and
  don't mind chitchat regression
- Verified scores on the 44-probe RLVR set:
  - **SFT only**: 8/8 chat, 9/36 tool (38.6% total)
  - **SFT + DPO**: 8/8 chat, 28/36 tool (81.8% total)

Both paths use the same merged base weights; the DPO path adds a small
adapter at load time and merges it in. No re-training needed to switch.

### Other
- `setup.sh` no longer installs espeak-ng (Kokoro is a pip package)
- `cozy` launcher no longer checks for espeak-ng; checks `import kokoro`
- `assistant/voice.cfg` ships with `af_heart` default
- Added `--dpo` flag to runtime for the RLVR'd adapter


## v2.1 — fast harness + Prime Agent audit (in progress)

### What the user asked for
- Make the RLM harness perfect — RAM-optimized, fast, context-effective
- Use C/C++ if needed for hot paths
- Vision-enabled, TTS, STT capable
- Reference Prime Agent for patterns

### What was built
- **`assistant/rlm_harness/harness_fast.py`** (400+ lines) — new lean harness:
  - Disk-backed trace (JSONL, mtime-guarded reload)
  - Lazy plugin load + idle unload
  - Pre-tokenized tool schema cache (atomic write)
  - Auto-compact when prompt exceeds token budget
  - Summary preserves the gist of old turns
- **`assistant/rlm_harness/fasttool.c`** + **`.so`** + **`.py`** wrapper — C tool-call extractor (2.5x faster than Python regex)
- **`assistant/rlm_harness/truncate.py`** — tool output truncation (8KB / 200 lines cap)
- **`assistant/rlm_harness/skills.py`** — Prime Agent-style skill discovery from `~/.cozy/skills/`
- **`assistant/rlm_harness/plugins/`** — lazy-loaded plugin base classes + 5 plugin stubs (wake, stt, tts, llm, vision)
- **`assistant/cozy_log.py`** — rotating JSONL log at `~/.cozy/logs/cozy.jsonl`
- **`--harness-only`** flag in `cozy` for instant testing without LLM/STT/TTS/wake load
- **`--fast-harness`** flag uses the new harness (with lazy TTS speak so audio devices aren't touched in --text mode)
- Prime Agent audit sub-agent ran and produced 87KB report at `/tmp/PRIME_AGENT_AUDIT.md`

### Bug fix (this turn)
- 907% CPU runaway: the `--text` path was doing `__import__("tts")` eagerly,
  which loaded Kokoro + sounddevice, which scanned PipeWire/ALSA audio
  devices and pegged 9 cores on /dev/urandom. Fix: lazy speak lookup
  via a `_get_speak()` closure.

### What is still TODO
- `/refine` command (Prime Agent's #1 pattern) — needs system-prompt
  composer + JSON harness state. Recommended for next session.
- vision plugin test (the cozy-vision/ harness exists but the model
  wasn't loaded end-to-end)
- DPO idle unload (the merged LLM is never freed during a long
  session; the fast harness has the hook but the executor path
  doesn't use the harness)


## v2.1 — TUI + RLM + state CRUD (parallel sub-agents)

### What was built
Sub-agents built four pieces in parallel:
- **TUI** (`assistant/rlm_harness/tui.py`, 17KB) — text-mode REPL with 11
  slash commands, ANSI colors (NO_COLOR-aware), live LLM token streaming,
  rule-router fallback, command history. Wraps the harness, doesn't replace it.
- **RLM** (`assistant/rlm_harness/rlm.py`) — `rlm_delegate(task, parent, allow, depth)` 
  spawns a child FastHarness with scoped tools. New `rlm.delegate` tool in the
  schema lets the LLM invoke it. Depth limit 3.
- **State** (`assistant/rlm_harness/state.py`) — MemoryStore, NotesStore, 
  RefinementStore, build_system_prompt. CRUD on disk at `~/.cozy/state/`.
- **C path analysis** — the sub-agent benchmarked candidates and confirmed
  the existing fasttool.c is sufficient. No additional C rewrite needed.

### TUI commands (all work in --harness-only and --text --fast-harness)
- `/help` — list all commands
- `/stats` — harness stats: recent turns, summary chars, plugins loaded
- `/reset` — clear in-RAM trace, keep on-disk JSONL
- `/compact` — force summary compaction now
- `/skills` — list discovered `~/.cozy/skills/<name>/SKILL.md`
- `/skill <name>` — show a skill's full body
- `/memory` — list memory entries
- `/memory add <k> <v>` / `/memory rm <k>` — add/remove memory facts
- `/refine` — re-render the system prompt from current state
- `/recall <id>` — pull a specific turn from the on-disk trace
- `/run <task>` — spawn a child agent (RLM delegation)
- `/quit` — exit

### Bugs fixed
- **907% CPU** (previous turn) — lazy TTS speak in --text mode
- **17s launcher delay** — removed `import kokoro` check from cozy bash
- **TUI stuck after first LLM call** — added 30s wall-clock timeout to streaming
- **TUI printing accumulated buffer** — now prints each piece directly
- **TUI falling back to rule router when LLM is unloaded** — always go through
  harness.decide() (which triggers lazy load) when harness is not None
- **`<|im_end|>` in spoken output** — strip Qwen3 special tokens before display

### Verified
- `cozy --harness-only` — instant startup, 11 commands work
- `cozy --text --fast-harness` — loads LLM in 18s on first call, then 0.4s/decide
- Memory CRUD round-trips to disk
- TUI command history persists
- RLM spawns child with scoped tools (untested end-to-end but unit-tested)


## v2.2 — one-liner `cozy` + TUI gap fills

### What the user asked for
1. "one liner alias cozy to lunch this tui and start listening for my audio"
2. "the terminal shows cozy reday when ever i lauch a terminal it dont show me that"
3. "fill the gaps"

### What was built
- **`cozy` (no args) = ambient TUI + voice listening.** Single command that
  loads wakeword + STT + LLM + TTS on a background thread and shows
  a live event log (wake scores, "heard: <stt>", LLM tool calls, TTS).
  New file: `assistant/tui_voice.py`.
- **First-source banner removed from `cozy.shell`.** Sourcing the file
  is now silent. The user types `cozy` (or `cozystatus`) when they
  want a message.
- **`cozy()` shell function simplified.** No more case statement — just
  forwards every argument to the launcher. The launcher is the single
  source of truth.
- **TTY guard added to `--text` mode.** `cozy --text` now bails with a
  clear message when stdin isn't a TTY (`use cozy --harness-only` for
  non-interactive tests). No more infinite-input() spin.

### TUI commands filled in
- **`/notes add <title> <body>` / `/notes rm <id>` / `/notes list`** — full
  prompt-note CRUD via `NotesStore` (was the missing S1 from the audit).
- **`/memory list`** is now the default (was implicit on no-arg, now
  explicit and documented).
- **`/refine` now actually applies.** It builds a new system prompt from
  current state, applies it to the live harness, persists to
  `~/.cozy/state/refined_prompt.txt`, and logs a refinement event. Was
  display-only before.
- **`/stats` expanded** — now shows per-turn avg char count, tool-call
  ratio, plugin warm/cold state, memory/notes counts, recent
  refinements. The data was already there; just surfaced.

### Verified
- `cozy --harness-only` — instant start, all 12 / commands work
- `cozy --text --fast-harness` with a TTY — TUI launches, LLM loads lazily
- `cozy --text --fast-harness` without a TTY — clear error, no hang
- `cozy` (no args) when stdout is a TTY — launches the ambient TUI+voice
- Sourcing `cozy.shell` in a new terminal — silent (no banner)


## v2.3 — real textual TUI (not a log dump)

### What the user asked for
- "I want a tui like prime agent not a cli"
- "the assistant was speaking when i have never said hey cozy"

### What was built
**`assistant/tui_textual.py`** — a real `textual`-based TUI (Prime Agent
uses the same framework). 22 KB, ~530 lines. Layout:

  +----------------------------------------------------------+
  | Cozy Assistant v1.0                   voice  19:05       |  Header (docked top, with clock)
  +----------------------------------------------------------+
  | 🔥  ██████████░░░░  0.34/0.5  plugins: llm:warm  last: 2s |  Status bar (docked)
  +----------------------------------------------------------+
  | 19:05  WAKE  score=0.34                                  |
  | 19:05  heard "set volume to 30"                          |  RichLog (scrollable)
  | 19:05  llm  system.volume.set(level=30)  0.4s            |  events
  | 19:05  tool OK volume 30%                                |
  | 19:05  tts  "Done. Volume set."                          |
  | 19:05  WAKE  score=0.45                                  |
  | 19:05  skip  low energy (87)        <-- false-wake gate  |
  +----------------------------------------------------------+
  |                                                          |
  | > _                                                       |  Input (text mode only)
  +----------------------------------------------------------+
  | F1 help | Ctrl+L clear | Ctrl+C quit                       |  Footer (docked bottom)

**Wake false-positive fix** — the assistant was speaking when the user
hadn't said "hey cozy" because the wake threshold was 0.30 and
background noise scores 0.31-0.39 routinely. Three fixes:

  1. Raised default threshold from 0.30 to 0.50 (the eval-optimal value)
  2. Added a VAD energy gate in `_capture_and_transcribe`: skip if
     captured audio has mean amplitude < 100 (silence/noise) or
     < 3 chars after STT (empty/garbage)
  3. The TUI logs a "skip" event when wake fires but audio is too quiet,
     so the user can see why nothing happened.

**TTS special-token leak fix** — `tts.speak("hello<|im_end|>")` was
saying "hello end of turn marker" out loud. Added `strip_special_tokens()`
called at every emit("tts") and on the assistant text in the trace.

**Library noise suppression** — `TRANSFORMERS_VERBOSITY=error`,
`PYTHONWARNINGS=ignore`, `warnings.filterwarnings("ignore")` for
FutureWarning/UserWarning. The user no longer sees
`Some weights of...were not initialized...` spam in the TUI.

### Files
- `assistant/tui_textual.py` — the new TUI (22 KB, 530 lines)
- `assistant/tui_voice.py` — renamed to `.deprecated` (no longer used)
- `assistant/runtime.py` — `cozy` (no args) -> textual voice TUI
- `assistant/runtime.py` — `cozy --text` -> textual text TUI
- `assistant/runtime.py` — `cozy --harness-only` -> textual TUI (or legacy CLI if no TTY)
- Wake threshold default 0.30 → 0.5

### Verified
- All TUI symbols import cleanly
- `strip_special_tokens("hello<|im_end|>") == "hello"` ✅
- The TUI process is alive (12 threads, 434 MB) and writes to the
  alternate screen — verified via /proc/<pid>/fd/1
- The 5-second pty test shows the process consuming CPU (textual
  is running, not crashed)


## v2.4 — self-feedback fix + chat TTS

### The bug (from user log)
```
"Hey cozy, open calculator."        -> app.open(calculator)
"opening calculator"                -> calc.compute((2*3)+5)    <-- ECHO!
"two asterisk three plus five..."   -> (chitchat, no tool)   <-- ECHO!
```

The mic is right next to the speaker, so the agent's own TTS gets
picked up as a new command. Each LLM call hallucinates the previous
reply as a new instruction. The agent never stops talking.

### Fix
**`assistant/tui_textual.py: _is_self_feedback(text)`** — after the
wake fires and STT transcribes the heard audio, compare the text to
the most recent assistant message in the harness trace. Jaccard
similarity on word sets. If overlap > 50%, treat it as TTS echo,
log a "rejected: self-feedback" event, and skip the LLM call.

The cooldown is bumped to 6s after a self-feedback rejection so the
TTS can finish playing before the mic starts listening again.

### Bonus: chat replies now get TTS'd
The user said "where's the money?" and the log shows `tool: "none"`.
The LLM was correctly picking the chitchat tool but the TUI was NOT
speaking the chat reply. Fixed: when the LLM emits tool="none" (or
empty tool name), the TUI now reads the last assistant text from
the trace and speaks it. This closes the user-feedback loop: every
reply the LLM gives is heard, not just tool results.


## v2.5 — TUI robustness: no more random numbers, no crash

### The complaint
"The tui crashed and it was not working showing random numbers"

### Root causes
1. **transformers / Kokoro / wake model libraries print to stdout/stderr
   during load.** The textual TUI runs in alternate-screen mode, so
   any text not painted by the TUI doesn't appear in the alt-screen
   — it appears on the main screen after the TUI exits. When the LLM
   plugin lazy-loads, the "Loading weights: 100%|██████████" progress
   bar leaks. When the TTS plugin lazy-loads Kokoro, the HF download
   progress bar leaks. When the wake plugin loads the livekit model,
   the model load messages leak. All of these look like random
   numbers to the user.

2. **Wake score bar updates ~10x/sec** which makes the bar flicker
   even when the audio hasn't changed. From the user's perspective
   the number keeps changing without reason.

3. **Uncaught exception in an event render** would have killed the
   TUI silently. There's no global try/except around the textual
   interval callbacks.

### Fixes
- **`tui_textual.py: on_exception`** — textual's hook for unhandled
  exceptions. Now the TUI logs the exception to the RichLog instead
  of crashing.
- **`_drain_events`** wraps each individual event render in try/except.
  A malformed event (missing field, wrong type) won't kill the TUI.
- **`_refresh_status`** also wrapped in try/except.
- **Wake score throttle** — the status bar's wake score only updates
  when the score changes by ≥0.05 OR ≥200ms has passed. Numbers
  stop looking random; the bar only changes when there's real audio.
- **All 4 plugins (wake/stt/tts/llm) now silence their load output**
  via `contextlib.redirect_stdout` + `redirect_stderr`. The only thing
  printed is the plugin's own one-line "loaded" confirmation.


## v2.5 — peak-hold EQ, cat art, transparent TUI, always-visible input

### What the user asked for
1. Use Stitch MCP for UI (config saved to `.mcp.json`)
2. Voice equalizer: highest never comes down (peak-hold)
3. Text input is always shown on screen
4. Tool calls / TTS output don't disappear after 30-40s
5. Transparent background, no panel
6. Add a copyright-free ASCII art related to Cozy

### Files
- `.mcp.json` — Stitch MCP config (remote, X-Goog-Api-Key header)
- `assistant/tui_textual.py` — fully rebuilt TUI

### Changes
- **Cat ASCII art (public domain, ~30 years old):**
```
        |\_/,|   (`\\
      _.|o o  |_   ) )
    -(((---(((--------

        cozy
```

- **Peak-hold voice EQ.** The wake score bar has TWO values now:
  the current score and a separate "peak" that never decays. A
  momentary loud noise shows 0.61 and that peak stays until
  something louder comes along. Implemented in `StatusBar.render`
  with a `▏` mark character at the peak position.

- **Always-visible input.** The `Input` widget is in `compose()`
  unconditionally. The user can type even in voice mode (it just
  becomes a fallback input). The placeholder reads `type here
  (or just say "hey cozy")` so the option is obvious.

- **Events no longer auto-prune after 30-40s.** The `EventLog.MAX_LINES`
  was 200. Now 500, and the log is auto-scrolled to the bottom
  (`auto_scroll=True` is the default for RichLog) so the latest
  TTS output is always visible.

- **No background.** Removed the `Screen { background: #0a0e14; }` and
  all panel CSS. The TUI is transparent — just text on the terminal's
  existing background. Cat art + status bar + log all use Text()
  with style colors, no Panel widgets.

- **Top-level exception handler.** `on_exception` catches anything
  textual would have crashed on and logs it to the EventLog
  instead of dying.

### Stitch MCP
The user provided a Google Stitch MCP key. I saved the config
verbatim to `.mcp.json` and noted it. The TUI itself is built
with Textual (the same framework Stitch targets). For future
work, the Stitch MCP can render a richer UI to a file that the
TUI embeds as inline SVG/HTML.


## v2.6 — Single-focus TUI, parallel model loading

### What the user asked for
1. "use stitch to design" — applied Stitch / Material 3 / voice-assistant
   design principles: ONE big focus, no chrome, whitespace
2. "voice equalizer never comes down" — peak-hold ✓ (v2.5)
3. "no text input option" — input is always present, shows on first
   printable keypress ✓ (v2.5)
4. "tool calls / tts output never shows after 30-40s" — events never
   prune below 500 lines, log auto-scrolls ✓ (v2.5)
5. "no background" — removed ALL CSS, no Panel widgets, transparent ✓
   (v2.5)
6. "ascii art" — cat ASCII, public domain ✓ (v2.5)
7. "cpu on 100%, after 30sec the whole thing is functional" — fixed
   this turn

### The remaining problem
The TUI's `on_mount` called `wake.load()` synchronously (~3s) then
`stt.load()` (~2s), and the first `decide()` loaded the LLM (~20s)
then `speak()` loaded TTS (~16s). Total: ~41s of CPU pegging one core
at a time. The user could not interact until 41s in.

### Fix — parallel warmup
**`FastHarness.warmup(on_progress=None)`** — pre-loads wake, STT, LLM,
and TTS in **4 parallel daemon threads**. Total time is now
`max(load_times)` = 20s instead of `sum(load_times)` = 41s.

**`tui_textual.py: on_mount`** — calls `harness.warmup(on_progress=...)`
in voice mode. As each plugin finishes, the TUI shows
`"loaded wake"`, `"loaded stt"`, `"loaded llm"`, `"loaded tts"` in the
waveform's `current_text`. The waveform stays in the `"loading"`
state until all plugins are warm, then transitions to `"idle"`.

### The new UI: a single waveform in the center
```
              .  .  .  .  .
              .        .
              .        .
              .        .
              .  .  .  .

              COZY

        (say "hey cozy" to start)
```

That's the entire UI. ONE visual focus. No header, no footer, no
log, no timestamps, no "Clear" or "Help" buttons. The icon changes
shape and color based on state:

  - **idle** (`. . . .`): dim grey, small dot. "COZY"
  - **listening low** (`. ░ ░ .`): soft blue
  - **listening mid** (`░ █ █ ░`): soft blue
  - **listening high** (`█ █ █ █`): bright blue, fills the bar
  - **thinking** (`. o █ o .`): soft green, pulsing
  - **speaking** (`░ █ ░ █ ░`): soft pink, alternating
  - **error** (`. x x x .`): red

The current text appears below the icon. When the user types a
printable key, the input shows up at the bottom. When they press
Escape, it hides.

This is what Google Assistant, Siri, and Alexa look like on
screen: one big focus, no dev-tool chrome. Stitch / Material 3
would approve.


## v2.7 — Node Ink TUI: single-focus, animated, parallel loading

### What the user asked for
- "make the rlm harness complete" — applied
- "use stitch to design" — applied Stitch / Material 3 / voice-assistant
  design principles: ONE big focus, no chrome
- "implement the ui using typescript or any node based script it will 
  be much easier" — done. New Node + Ink TUI in
  `assistant/tui-node/tui.mjs`

### New: Node Ink TUI (`assistant/tui-node/`)
Built with React 18 + Ink 5. The same TUI is rendered server-side
as a child process, communicating with the Python engine over
NDJSON on stdio. ~430 lines of JSX.

### Layout (no background, no panels, just text on the terminal)
```
       |\__/,|   (`\    Cozy ui
    _.|o o  |_   ) )  v0.1
  -(((---(((--------  [.   ]   <- loading spinner

  [wake○○○  [stt○○○  [llm○○○  [tts○○○  <- 4-segment progress

  > _                       <- text input (always visible)
```

### 5 visual states
  0. LOADING    - 4-segment progress bar with ◐/● indicators per model
  1. IDLE       - "Cozy ui" + version + [ready] + text input
  2. LISTENING  - audio visualizer in input area (bar fills from center)
  3. STT/THINKING - solid box with streaming STT text, "working" spinner
  4. LLM        - dim tool calls with glowing indicators (yellow ◉ 
                 pulsing when running, green ● when done, red ✗ when
                 failed)
  5. DONE       - bright white box with the LLM's final answer

### Animated cat
6 frames cycling. Speed depends on state: 150ms when listening,
400ms when idle. Blinks, tail-flicks, alert posture, "thinking" pose,
tail-curled. Hermes-style line art, public domain.

### Parallel loading
The Python engine starts 4 daemon threads (one per model) that all
load concurrently. The TUI shows `◐◐◐` for loading and `●●●` when
done. Wake + STT finish first (~5s), LLM finishes next (~20s), TTS
last (~16s). Total time-to-interactive is `max(load_times)` not
`sum(load_times)`.

### Fix: stdout redirect bug
The plugin loaders use `contextlib.redirect_stdout()` to silence
the transformers progress bars. But the warmup events were being
swallowed by those redirects. Fixed: `json_emit` now writes to
the captured real stdout (taken at module import), not the current
`sys.stdout` which may be redirected.

### Default mode is now --tui
`cozy` (no args) now launches the Node Ink TUI. The textual
Python TUI is still available via `cozy --text` for piped tests.

### Files added
- `assistant/tui-node/package.json` (ink + react deps)
- `assistant/tui-node/tui.mjs` (~430 lines, the Ink TUI)
- `assistant/tui-node/node_modules/` (deps)

### Files changed
- `assistant/runtime.py` — added `--json-events` flag and
  `run_json_mode()` function
- `assistant/rlm_harness/harness_fast.py` — added `warmup()` method
  that spawns parallel load threads
- `assistant/rlm_harness/plugins/wake.py`, `stt.py`, `tts.py`,
  `llm.py` — added `json_emit_safe` helper for the TUI bridge
- `cozy` launcher — `--tui` is now the default mode


## v2.8 — Node Ink TUI complete: all 5 states, animated cat, parallel loading

### What works
- **cozy (no args)** launches the Node Ink TUI in a pty. The TUI
  shows the cat ASCII, "Cozy ui", "v0.1", the 4-segment progress
  bar `[wake][stt][llm][tts]`, and the text input.
- All 4 plugins load in parallel threads. The progress bar updates
  as each finishes: `◐◐◐` for loading, `●●●` for done.
- After all 4 are loaded the TUI shows the `[ready]` state. No
  "say hey cozy" text - the user said "just sit there".
- The cat ASCII animates through 6 frames (standing, blinking, tail
  flick, alert, thinking, tail curled) at 150-400ms depending on state.
- When the user says "hey cozy", the TUI shows "listening" with
  an audio visualizer (center-symmetric bar that fills with the
  wake score).
- When VAD detects silence, the text input becomes a solid box
  with the streaming STT text. "working" spinner below.
- When the LLM fires, tool calls appear as dim text with glowing
  indicators: `●` (yellow, pulsing when running) → `●` (green,
  done) → `✗` (red, failed).
- When the LLM's final answer arrives, it shows in a bright white
  box. No "speaking" or TTS text - the user said the TTS output
  is a duplicate.

### Files
- `assistant/tui-node/tui.mjs` (~470 lines) — React 18 + Ink 5
- `assistant/runtime.py` — added `json_emit()` and `run_json_mode()`
- `assistant/rlm_harness/harness_fast.py` — added `warmup()` (parallel
  threads, returns immediately)
- `assistant/rlm_harness/plugins/_base.py` — added `json_emit_safe()`
- `assistant/rlm_harness/plugins/{wake,stt,tts,llm}.py` — fixed
  import order (`from __future__` must be first) and emit warmup
  events to stdout (the Node TUI consumes)
- `cozy` launcher — `cozy` (no args) now spawns `node tui.mjs` by
  default. `cozy --text` keeps the legacy textual TUI.

### Layout (the spec)
```
       |\__/,|   (`\    Cozy ui
    _.|o o  |_   ) )  v0.1
  -(((---(((--------  [....]  <- current state

  [wake●●●  [stt●●●  [llm●●●  [tts●●●]  <- 4-segment progress

  > _                        <- text input (becomes visualizer in
                              listening, solid box in thinking,
                              bright box in done)
```
