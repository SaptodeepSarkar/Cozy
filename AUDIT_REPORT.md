# Cozy product and code audit

Date: 2026-09-12

## Executive summary

The reported behavior was reproducible from code, saved runtime traces, model
benchmarks, and automated tests. The largest problems were orchestration bugs,
not just model quality: wake-trigger latency discarded the start of commands;
capture endpointing was too aggressive; TTS was only nominally warmed and cut
multi-part output; unfinished model reasoning polluted later prompts; and the
UI claimed playback was complete before audio actually ended.

This pass fixes those runtime paths, adds regression coverage, isolates unsafe
training smoke outputs, and hardens tool execution. The raw 0.6B LLM remains a
quality limitation and the wake-word training source is incomplete in this
checkout. Those two items require a separate model/source release rather than a
code-only claim of completion.

## Evidence collected

- The checked-in wake evaluation reports threshold `0.5`, optimal threshold
  `0.31`, recall `98.33%` and `0.0` FPPH at the optimal threshold.
- The saved local trace contains unfinished `<think>` outputs lasting roughly
  20–24 seconds, repeated generic replies, `none` replies, and unrelated tool
  calls for ordinary chat. These bad turns were replayed into later prompts.
- The repository benchmark documents v1.1 LLM tool-call accuracy of 63.2%.
  A fresh 8-probe GPU sample in this audit scored 3/8 (37.5%) in 9.884 seconds.
- The new deterministic runtime path routes those same 8 common probes 8/8
  without model generation.
- The harder 217-clip STT benchmark reports v1.2 overall WER 11.9% versus v1.1
  at 16.6%, but runtime was still pinned to v1.1.
- `cozy-vision/` is documented in `AGENTS.md` but absent from this checkout;
  `.gitignore` explicitly says it was deleted.
- The assistant suite passed after changes. The Node UI suite and TypeScript
  typecheck passed. The wake library suite cannot currently be collected in
  its own setup environment because `pytest` is absent and vendored training
  modules are stubs.

## Fixed in this pass

### A-01 — command audio lost after a late wake decision (high)

Cause: wake inference correctly needs a rolling two-second window, but capture
started strictly after inference. Natural speech such as “Hey Cozy, open
Firefox” could place the first command words in the already-consumed window.

Fix: retain a configurable 0.8-second command pre-roll and remove a recognized
leading wake phrase from the STT text. The two-second wake model contract is
unchanged.

Verification: regression tests cover pre-roll boundaries and wake-phrase
cleanup.

### A-02 — production wake threshold caused phantom activations (high)

Cause: the first audit incorrectly promoted the evaluation set's
`optimal_threshold: 0.31` to the live default. A balanced clip evaluation does
not represent hours of real-room negative audio, and this caused phantom wake
events in normal use.

Fix: restore the room-safe `0.50` default, retain the environment/CLI override,
and require two adjacent positive scores. Only a very strong score (`>=0.85`)
may trigger immediately. This adds at most one 160 ms scoring interval for a
marginal real wake while reducing one-frame false positives.

Verification: regression coverage requires the `0.50` default and checks the
explicit override. Live-room false-accept/false-reject testing is still a
release requirement.

### A-03 — STT stopped on natural pauses and clipped edge words (high)

Cause: endpointing stopped after 0.7 seconds of quiet, maximum capture was six
seconds, and faster-whisper ran a second VAD pass over already endpointed
audio. That second pass can remove quiet initial/final words.

Fix: endpoint silence is now 0.80 seconds, minimum capture is 0.55 seconds, and
the no-speech wait is 1.20 seconds instead of three. Pre-roll is retained, but
at least 160 ms of verified post-wake speech is required before STT runs.
Silence therefore never reaches Whisper and cannot become a fabricated
sentence.

Verification: CT2 call tests require `vad_filter=False`.

### A-04 — slower/older STT artifact selected (high)

Cause: complete local v1.2 artifacts existed, but constants and status output
always selected the v1.1 symlinks.

Fix: runtime and status prefer complete v1.2 CT2/HF artifacts and safely fall
back to the versioned v1.1 paths on a clean checkout. CT2 decoding changed
from beam 3 to greedy beam 1 for command latency.

Verification: artifact-selection and inference-option tests pass; `cozy
--status` reports the actual selected v1.2 files on this machine.

### A-05 — first TTS reply paid model-load latency (high)

Cause: the TTS warmup plugin imported the module but never initialized Kokoro.
The UI marked warmup complete and the first response did the expensive work.
The playback timeout path also referenced undefined variables, so successful
synthesis could fail before `paplay` was ever launched.

Fix: models now initialize sequentially to avoid native-library and memory
contention. Kokoro is pinned to CPU and its configured voice is loaded during
startup. The underlying error is surfaced instead of the generic “Kokoro could
not be initialized”. TTS is required when voice output is enabled; a failure
stays on the startup error screen rather than presenting a falsely ready app.
Playback duration is now read from the generated WAV before the bounded
`paplay` call.

### A-06 — TTS truncated multi-part replies and could hang (critical)

Cause: only the first item from Kokoro's sentence generator was returned.
Playback had no timeout. Cache names used only the first normalized 64
characters, allowing collisions.

Fix: concatenate every generated audio segment, use SHA-256-suffixed cache
keys, bound `paplay` by audio duration plus grace time, and correctly account
for pending audio before the worker starts.

Verification: tests cover multi-segment output and collision-resistant keys.

### A-07 — UI showed Ready while still speaking (high UX)

Cause: the backend emitted `done` immediately after enqueueing TTS and the
state reducer treated it as playback completion.

Fix: TTS now emits actual start/finish events. `done` preserves the Speaking
state until playback finishes. Recoverable component errors remain visible in
activity without turning the whole composer into a fatal screen.

Verification: reducer tests cover playback state, persistent replies, and the
absence of a synthetic second response.

### A-08 — LLM sometimes returned nothing or poisoned later turns (critical)

Cause: unfinished `<think>` output had no closing tag, so the old regular
expression did not strip it. The trace stored zero token counts, meaning the
configured context budget and compaction threshold were ineffective.

Fix: incomplete reasoning is stripped through end-of-output; empty output is
retried once with clean context; errors produce a visible, speakable fallback;
turn token estimates are populated; old reasoning is sanitized; summary size
is bounded; and newest turns are selected within a real conversation budget.
Generation caps were reduced from 512/256 to 160/96 tokens.

Verification: tests cover unfinished reasoning, clean-context retry, and
preservation of the newest request.

### A-09 — common actions depended on the regressed LLM (high)

Cause: an existing deterministic intent router was not used by the active fast
harness, so even obvious commands waited for a model with known tool-call
regressions.

Fix: high-confidence rules now bypass generation for time, numeric/spoken
volume, mute, brightness, app open/close, browser search, screenshot, media
play/pause, date, battery, and uptime. Unmatched requests still use the LLM.

Verification: all eight failed/passed GPU sample categories route correctly in
the active runtime path without invoking the LLM.

### A-10 — model output could reach hidden executor handlers (critical safety)

Cause: normalized names not present in the advertised tool schema were still
returned to the runtime. The executor contains handlers not exposed to the
model, so the schema was prompt decoration rather than an authority boundary.

Fix: model-selected actions are rejected unless they are in the tool schema.
Missing volume/brightness parameters now fail instead of silently applying
50%. Power actions retain their independent environment opt-in and strict
boolean confirmation.

Verification: a regression test attempts a hidden `system.shutdown` tool call
and confirms that no action is returned.

### A-11 — recursive delegation was nonfunctional and not truly scoped (high)

Cause: it imported a nonexistent relative logger, duplicated the task in the
child trace, never executed a selected child tool, and did not enforce the
allowed tool list after generation.

Fix: correct logging import, one task turn, actual tool execution, recursive
depth handling, schema validation, and post-selection scope enforcement.

Verification: tests cover successful delegated execution and blocked
out-of-scope actions.

### A-12 — “safe” smoke training could overwrite production models (critical)

Cause: the smoke profile passed two training steps but left default LLM, DPO,
HF and CT2 output paths pointing at active artifacts.

Fix: every smoke output is now rooted under that run's
`artifacts/training_runs/<run-id>/smoke_outputs/` directory. Standard/quality
profiles retain their intended production outputs.

Verification: dry-run output confirms all smoke model, adapter, DPO, STT and
export paths are isolated.

### A-13 — launcher, input, shutdown, and UI inconsistencies (high UX)

Cause: `bash run.sh` launched the legacy Python/Textual UI while `cozy`
launched Ink. Ink synthesized a second reply while the backend's real `done`
reply was already present, wake/STT state changes cleared prior output, and
shutdown paths could issue repeated native-process signals.

Fix: replace Ink/React with OpenTUI core. A dedicated loading surface is the
only screen until required models finish, with sequential model status,
elapsed seconds, a visual progress bar, and a solid SVG-derived logo. The main
workspace then provides native text input and persistent conversation cards.
Only backend `done` events render assistant replies. Ctrl+C uses one idempotent
shutdown path with one immediate SIGKILL for a child that may be inside
an uninterruptible native model load; ordinary shutdown uses one SIGTERM and,
only if still alive after two seconds, one SIGKILL. No signal is repeated.
`cozy --stop` matches only this checkout's absolute runtime/UI paths.

Verification: TypeScript typechecking and reducer tests pass, including a test
that a later wake cannot erase the prior answer.

### A-14 — executor false-success paths (high)

Cause: screenshot handlers returned `((ok, message), path)`, making the nested
tuple truthy even when capture failed. Missing apps fell back to opening their
name as a file and reported success. The calculator alias could launch Kate.

Fix: screenshots propagate the real result, missing applications fail clearly,
and the invalid calculator fallback was removed.

Verification: executor tests cover screenshot failure, missing apps, and
missing setting parameters.

### A-15 — Cozy omitted ArchFlow's input-quality layers (high)

Cause: ArchFlow and Cozy use the same v1.2 Whisper-small CT2 weights (matching
SHA-256), but Cozy returned every decoded segment verbatim. ArchFlow trims
padding, rejects segments with decoder `no_speech_prob > 0.7`, removes fillers
and accidental repetitions, and optionally applies a transcript-specific
Qwen3-0.6B LoRA. The perceived difference was orchestration, not STT weights.

Fix: port the same CT2 prompt/beam/independent-decoding settings, silence trim,
segment rejection, and deterministic polish. Voice startup now also loads the
same ArchFlow Qwen3-0.6B `dpo-sft` cleaner once and reuses it for transcripts of
ten or more words. Model/adapter paths and the word threshold are configurable.
If artifacts or CUDA are unavailable, input remains functional with the
deterministic pass. Cleanup output is rejected if it changes numbers or
negation, diverges excessively, emits control tokens, or changes length beyond
safe bounds.

Verification: tests cover no-speech segment rejection, prompt/decoding options,
filler and repetition handling, preservation of intentional emphasis, rewrite
safety, and model-unavailable fallback.

## Remaining product risks

### R-01 — raw LLM quality still requires a model release (high)

The rule path stabilizes common actions, but open-ended chat and long-tail tool
phrases still depend on a weak artifact. The new 8-probe raw sample is 37.5%;
the repository's broader prior result is 63.2%. Retrain/evaluate against the
correct chat template, promote only if held-out tool and chat gates pass, and
do not treat runtime parsing as a substitute for model quality.

Recommended release gates: at least 90% tool selection, at least 95% chat
non-overfire, zero hidden-tool execution, zero empty output, and p95 warm-model
decision latency measured on the RTX 3050.

### R-02 — wake training pipeline is not reproducible (release blocker)

The vendored `data/augment`, `data/features`, and `data/generate` packages are
explicit stubs, while README/AGENTS instructions claim the full retraining
command works. Tests import missing augmentation and VoxCPM modules. The setup
environment also omits pytest despite shipping a pytest suite.

Do not modify the production ONNX model until the exact upstream v0.2.0 source
plus Cozy's documented extraction-signature patch is restored and the full
evaluation passes. This pass intentionally did not fabricate a replacement
training pipeline or touch `hey_cozy.onnx`.

### R-03 — Cozy-Vision is absent (release blocker for GUI-agent claims)

There is no `cozy-vision/` implementation in this checkout, despite detailed
commands and model claims in `AGENTS.md`. Either restore and independently
test that product or remove the claims. The current terminal UI was audited;
a nonexistent desktop GUI could not be tested or fixed.

### R-04 — acoustic and perceived-latency validation needs a human session

Automated code tests cannot establish room-specific false wakes, microphone
gain, Bluetooth routing, word-edge recall, or perceived TTS onset. Run at least
50 wake attempts across distance/noise conditions, 100 spoken commands with
pause/overlap cases, and capture p50/p95 timings for wake, capture, STT, LLM,
first audio, and playback completion.

### R-05 — documentation/version drift (medium)

README, AGENTS, VERSIONS, changelog and package metadata mix v1.52, v1.53,
v2.0 and deleted-component claims. Documentation should be regenerated from a
single release manifest after the model/source decisions above.

### R-06 — setup hides system-package failures (medium)

Both package-manager branches end in `|| true`. A failed prerequisite install
therefore becomes a later, less actionable runtime/import error. Setup should
report optional versus required packages separately and fail when required
audio/build dependencies are unavailable.

## Verification record

- Assistant Python tests: 48 passing after the input-layer additions.
- OpenTUI reducer suite: 11 cases passing.
- TypeScript: `tsc --noEmit` passing.
- Python changed-file compilation: passing.
- Shell syntax for launch/setup scripts: passing.
- Diff whitespace validation: passing.
- Training smoke dry-run: passing; all generated model paths isolated.
- Actual LLM GPU sample: completed on NVIDIA RTX 3050 6GB Laptop GPU;
  3/8 raw model, 8/8 deterministic runtime routing for the same categories.
- ArchFlow-compatible input GPU smoke: Qwen3-0.6B `dpo-sft` loaded in 8.23
  seconds cold and cleaned a 17-word sample in 2.77 seconds while preserving
  both numbers, negation, and intentional emphasis. The complete wake + STT +
  assistant LLM + cleanup LLM + TTS stack loaded together in 20.98 seconds and
  freed cleanly, confirming it fits the target machine.
- End-to-end NDJSON backend smoke: LLM startup completed in about 5.64 seconds;
  `what time is it` routed, executed and emitted `done` in about 1.4 ms;
  graceful Ctrl+C emitted `shutdown` and exited with code 0.
- OpenTUI PTY smoke: loading screen transitioned to the workspace, native
  input submitted `hello`, exactly one backend reply remained visible, and
  Ctrl+C restored the terminal immediately.
- Wake runtime model construction: covered and passing in assistant tests.
- Full wake package tests: blocked by missing pytest and deliberately stubbed
  training modules; recorded as R-02 rather than reported as a pass.
