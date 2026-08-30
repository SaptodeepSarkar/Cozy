# Patterns Cozy should steal from Prime Agent

The Prime Agent repo is a much larger system (multi-user daemon, recursive subagents, MCP, OAuth) but it solves several problems Cozy also faces, and the patterns transfer cleanly. Below are the concrete, file-level patterns worth copying.

---

## 1. System-prompt composition (the one that grows via `/refine`)

**Source:** `/tmp/prime-agent-src/packages/coding-agent/src/core/system-prompt.ts` (lines 41-211) and `/tmp/prime-agent-src/packages/coding-agent/src/core/prompts/rlm.ts` (the `buildRlmPrompt` and `LONG_RUNNING_WORK_PROMPT` constants).

Prime Agent builds the system prompt by **composing pure function calls** in a fixed order: base prompt → harness-state overview → MCP guidance → extra guidelines → project context files → skills section → appended system prompt. Each piece is conditionally included and the order is deliberate (subagent guidance comes *after* the recursion doctrine so the model reads the when/why before the menu of subagents it can match against — see the inline comment at lines 121-123 of system-prompt.ts).

```ts
// system-prompt.ts, lines 95-211 (BuildSystemPromptOptions + buildSystemPrompt)
export interface BuildSystemPromptOptions {
    customPrompt?: string;
    selectedTools?: string[];
    promptGuidelines?: string[];
    appendSystemPrompt?: string;
    cwd: string;
    skills?: Skill[];
    harnessState?: HarnessState;
    allowRecursion?: boolean;
    rlmDepth?: number;
    ...
}

let prompt = buildRlmPrompt({ cwd, messagesPath, installedSkills, activeTools, allowRecursion, depth });
if (allowRecursion && hasIpython) prompt += `\n\n${buildSubagentGuidance({...})}`;
if (harnessState) prompt += `\n\n${formatHarnessStateForPrompt(harnessState, {...})}`;
const guidelines = formatPromptGuidelines(promptGuidelines);
if (guidelines) prompt += `\n\n# Additional Guidance\n\n${guidelines}`;
if (contextFiles.length > 0) { prompt += `\n\n# Project Context\n\n...`; }
if (hasFileAccess && skills.length > 0) prompt += formatSkillsForPrompt(skills);
if (appendSection) prompt += appendSection;
return prompt;
```

The `formatHarnessStateForPrompt` function (refinement.ts, lines 593-650) injects a **compact, capped overview** of all continual-harness entries (kind, title, path, version, content preview) with stable rules about when to refine. That overview is rebuilt every time the system prompt is rebuilt, so `/refine` becomes "edit a JSON store and trigger `rebuildSystemPrompt()`."

**How Cozy would adapt it.** Cozy's `assistant/runtime.py` (line ~1) currently has a hand-rolled prompt string. Replace it with a `build_system_prompt(state: CozyState) -> str` function in `assistant/prompt.py` that concatenates: (a) a small immutable base (your "You are Cozy, a voice assistant…" line), (b) a `format_harness_state(state, limit=6)` block that lists `.cozy/harness/memories/*.json` and `.cozy/harness/skills/*.json` with title + 180-char content preview + version (so the LLM has routing hints but not full text), (c) the discovered tool snippets, (d) the project AGENTS.md content, and (e) any user `appendSystemPrompt` from a config file. The `/refine` command becomes "edit one of those JSON files and call `runtime._rebuild_system_prompt()`." Mirror the `LOCAL` / `GLOBAL` scope split (refinement.ts lines 51-53) so Cozy can have per-session memories and a cross-session global memory file at `~/.cozy/harness/`.

---

## 2. Tool registry and the `ToolDefinition` interface

**Source:** `/tmp/prime-agent-src/packages/coding-agent/src/core/extensions/types.ts` (lines 80-160, the `ToolDefinition<TParams, TDetails, TState>` interface) and `/tmp/prime-agent-src/packages/coding-agent/src/core/tools/index.ts` (lines 47-53, the `createAllToolDefinitions` factory).

Tools are typed objects, not free functions. The contract is `name + label + description + promptSnippet + parameters (TypeBox schema) + executionMode + execute(toolCallId, params, signal, onUpdate, ctx) -> AgentToolResult<TDetails>`. The registry is a plain `Map<string, AgentTool>` built once per session and rebuilt only when tools change.

```ts
// tools/index.ts, lines 47-53
export type ToolName = "ipython";
export function createAllToolDefinitions(cwd: string, options?: ToolsOptions): Record<ToolName, ToolDef> {
    return {
        ipython: createIpythonToolDefinition(cwd, options?.ipython),
    };
}
```

```ts
// tools/bash.ts, lines 124-145 (ToolDefinition literal + pluggable BashOperations)
const definition: ToolDefinition<typeof bashSchema, BashToolDetails | undefined, BashRenderState> = {
    name: "bash", label: "bash",
    description: `Execute a bash command in the current working directory. ...`,
    promptSnippet: "Execute bash commands (ls, grep, find, etc.)",
    parameters: bashSchema,
    async execute(_toolCallId, { command, timeout }, signal?, onUpdate?, _ctx?) { ... },
};
```

The `BashOperations` interface (bash.ts, lines 28-58) is the **most reusable piece**: it abstracts the actual spawn behind `{ exec(command, cwd, { onData, signal, timeout, env }) }` so a tool definition can be reused for local bash, SSH, Docker, or a synthetic backend in tests. `createLocalBashOperations` is the default.

```ts
// tools/bash.ts, lines 32-58
export interface BashOperations {
    exec: (command: string, cwd: string, options: {
        onData: (data: Buffer) => void;
        signal?: AbortSignal; timeout?: number; env?: NodeJS.ProcessEnv;
    }) => Promise<{ exitCode: number | null }>;
}
export function createLocalBashOperations(options?: { shellPath?: string }): BashOperations { ... }
```

The build-time wiring is in `agent-session.ts` lines ~3590-3620: `_buildRuntime` calls `createAllToolDefinitions` and merges in extension-registered tools into `_toolRegistry: Map<string, AgentTool>`. Each tool also exposes `promptSnippet` (one-liner used inside a custom prompt) and `promptGuidelines` (a list of bullets appended to the system prompt when the tool is active — see `_rebuildSystemPrompt` in agent-session.ts, lines around 5750-5800, where the loop pulls snippets/guidelines from the registry).

**How Cozy would adapt it.** Build a `cozy/tools/` package with one `ToolDefinition` per voice-pipeline surface. The three Cozy would actually need: `play_tts(text: str, voice: str = "default")`, `transcribe_audio(path: str)` (in case the wake gate triggered a buffer capture), and `set_reminder(at: str, message: str)` (or similar LLM-initiated follow-up). Each tool's `execute` returns a `ToolResult` dataclass `{ content: str, details: dict, is_error: bool }` and is registered in a `TOOL_REGISTRY: dict[str, ToolDefinition]`. Pass the registry to the LLM as the `tools` field on every chat completion. Mirror `BashOperations` for `play_tts` — define a `TtsBackend` protocol with `synthesize(text, voice) -> Iterator[bytes] | coroutine` and ship `piper_backend` and `mock_backend` so tests don't need audio. The `promptSnippet` / `promptGuidelines` fields become a `@tool.snippet("…")` / `@tool.guideline("…")` decorator that registers metadata in the registry; the system-prompt builder reads the registry to compose the LLM-facing tool list.

---

## 3. Truncation policy for tool output

**Source:** `/tmp/prime-agent-src/packages/coding-agent/src/core/tools/truncate.ts` (full file, 254 lines) and `/tmp/prime-agent-src/packages/coding-agent/src/core/tools/output-accumulator.ts` (streaming accumulator).

The shared `truncateHead` / `truncateTail` functions are line- AND byte-bounded (whichever is hit first), return a `TruncationResult` that records which limit was hit, and `OutputAccumulator` writes the **full** output to a temp file so the LLM can read it back via a path if it needs more. From the bash tool (`tools/bash.ts` lines 99-117):

```ts
description: `Execute a bash command ... Output is truncated to last ${DEFAULT_MAX_LINES} lines or ${DEFAULT_MAX_BYTES / 1024}KB (whichever is hit first). If truncated, full output is saved to a temp file. ...`,
```

The truncation summary is appended *inline* to the LLM-visible result so the model knows where the cut happened and how to recover the rest. From `tools/bash.ts`, `formatOutput`:

```ts
if (truncation.truncatedBy === "lines") {
    text += `\n\n[Showing lines ${startLine}-${endLine} of ${truncation.totalLines}. Full output: ${snapshot.fullOutputPath}]`;
} else {
    text += `\n\n[Showing lines ${startLine}-${endLine} of ${truncation.totalLines} (${formatSize(DEFAULT_MAX_BYTES)} limit). Full output: ${snapshot.fullOutputPath}]`;
}
```

`DEFAULT_MAX_LINES = 2000` and `DEFAULT_MAX_BYTES = 50 * 1024` (truncate.ts, lines 15-16). UTF-8 is handled correctly in the `truncateStringToBytesFromEnd` helper (lines 199-219) so multi-byte chars never get split.

**How Cozy would adapt it.** The single biggest risk in Cozy is a long STT transcript or a noisy TTS round-tripping a wall of text back into the LLM context. Drop the 2000-line / 50-KB `truncateHead` + `truncateTail` pair (verbatim — no Cozy-specific redesign needed) into `assistant/truncate.py`. Wrap any tool result > 4 KB with `truncate_tail(text, max_lines=200, max_bytes=8_000)` and append a `[Showing last N of M lines. Full transcript: /tmp/cozy-tts-XXXX.log]` line. That single line tells the LLM it can read the file via the `read_file` tool if it actually needs the rest. The `OutputAccumulator` pattern (output-accumulator.ts) is the streaming analogue: if TTS is mid-utterance, write the full audio path to `/tmp/cozy-tts-$id.wav` as you go, then return the LLM just the transcript tail. This keeps the LLM context tiny and the audio filesystem-backed.

---

## 4. Skill on-disk format

**Source:** `/tmp/prime-agent-src/packages/coding-agent/src/core/skills.ts` (the `loadSkillsFromDir` at lines 158-245 and `formatSkillsForPrompt` at lines 273-310) plus `/tmp/prime-agent-src/packages/coding-agent/docs/skills.md` (the format spec).

Discovery rules (skills.ts lines 158-200): a directory containing `SKILL.md` is a skill root, no recursion below it. A skill is **a single markdown file with YAML frontmatter** that conforms to the [Agent Skills standard](https://agentskills.io/specification): `name` (lowercase a-z, 0-9, hyphen, must match parent dir name, max 64 chars), `description` (required, max 1024 chars), optional `disable-model-invocation: true`. At session start, only `name` + `description` + `type` + `filePath` are injected into the system prompt as an XML block (`<available_skills>...</available_skills>`). The full file loads on demand.

```ts
// skills.ts, lines 284-308
export function formatSkillsForPrompt(skills: Skill[]): string {
    const visibleSkills = skills.filter((s) => !s.disableModelInvocation);
    if (visibleSkills.length === 0) return "";
    const lines = [
        "\n\nThe following skills provide specialized instructions for specific tasks.",
        "Use ipython to inspect a skill's file when the task matches its description.",
        "Skills with a python_import are prepared in the persistent Python kernel ...",
        "When a skill file references a relative path, resolve it against the skill directory ...",
        "",
        "<available_skills>",
    ];
    for (const skill of visibleSkills) {
        lines.push("  <skill>");
        lines.push(`    <name>${escapeXml(skill.name)}</name>`);
        lines.push(`    <type>${skill.kind}</type>`);
        if (skill.kind === "python") {
            lines.push(`    <python_import>${escapeXml(skill.python.importName)}</python_import>`);
        }
        lines.push(`    <description>${escapeXml(skill.description)}</description>`);
        lines.push(`    <location>${escapeXml(skill.filePath)}</location>`);
        lines.push("  </skill>");
    }
    lines.push("</available_skills>");
    return lines.join("\n");
}
```

**How Cozy would adapt it.** Create `.cozy/skills/` in the Cozy home and one subdirectory per skill (`calendar/SKILL.md`, `music/SKILL.md`, `notes/SKILL.md`). Reuse the YAML frontmatter rule verbatim, including the strict name regex `/^[a-z0-9-]+$/` and the "name must match parent directory" rule (skills.ts lines 124-150, the `validateName` and `validateDescription` functions). At wake-gate, only emit the XML `<available_skills>` block to the LLM — full SKILL.md loads when the user invokes `/skill <name>` or when the description's keywords match. For a single-user laptop with no subagents, you can skip the Python-backed skill extension entirely; the markdown-only path is enough. If you do want callable skills (e.g. `await calendar.add_event(...)`), follow the Prime Agent pattern of adding a `pyproject.toml` + `src/<name>/__init__.py` per skill (see `packages/coding-agent/skills/refine/`) and resolve imports against the LLM-side venv that the Cozy `assistant/` project already maintains.

---

## 5. The Python REPL bootstrap (and why Cozy probably doesn't need it)

**Source:** `/tmp/prime-agent-src/packages/coding-agent/src/core/kernel/bootstrap.ts` (`bootstrapVenv` at lines 318-334, `acquireBootstrapLock` at lines 268-289), `/tmp/prime-agent-src/packages/coding-agent/src/core/kernel/repl-manager.ts` (the `ReplKernelManager` class at lines 60-220), and `/tmp/prime-agent-src/prime-agent-runtime/src/rlm/repl.py` (`main()` at lines 1121-1145).

The pattern is: TypeScript host owns a subprocess (Python 3.11 in a uv-managed venv at `~/.prime/agent/kernel-venv/`) that talks back over a newline-delimited JSON protocol on stdio. Bootstrap is gated by a `mkdir`-based lock so two parallel invocations don't double-install; the lock uses a `pid` file inside a lock dir and reaps stale locks (lines 270-289). The kernel ships with a `dill` snapshot so the user-namespace survives session resume.

```ts
// bootstrap.ts, lines 318-334
async function bootstrapVenv(venv, pythonSkills, options) {
    await mkdir(path.dirname(venv), { recursive: true });
    const uv = await ensureUv(options);
    const python = path.join(venv, "bin", "python");
    const sourceDir = await resolveRuntimeSourceDir();
    const runtimeRequirement = sourceDir ?? RUNTIME_REQUIREMENT;
    const runtimeIdentity = await resolveRuntimeIdentity();
    await run(uv, ["python", "install", PYTHON_VERSION]);
    await run(uv, ["venv", venv, "--python", PYTHON_VERSION, "--seed"]);
    await run(uv, ["pip", "install", "--python", python, runtimeRequirement, STATE_SNAPSHOT_REQUIREMENT, ...DEFAULT_RLM_EXTRA_UV_ARGS]);
    await syncPythonSkills(uv, venv, python, runtimeIdentity, pythonSkills, options);
}
```

```python
# repl.py, lines 1121-1145
def main() -> None:
    global _loop, _serve_task
    stdin_fd = _setup_fds()
    _start_owner_watchdog()
    sys.modules.setdefault("rlm.repl", sys.modules[__name__])
    user_module = types.ModuleType("__main__")
    user_module.__dict__["__builtins__"] = __builtins__
    sys.modules["__main__"] = user_module
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    signal.signal(signal.SIGINT, _sigint_handler)
    threading.Thread(target=_read_requests, args=(stdin_fd, queue), daemon=True).start()
    _send({"event": "ready", "protocol": PROTOCOL_VERSION, "python": platform.python_version()})
    _serve_task = _loop.create_task(_serve(queue, user_module.__dict__))
    while not _serve_task.done():
        try: _loop.run_until_complete(_serve_task)
        except KeyboardInterrupt: continue
    _loop.close()
```

**How Cozy would adapt it.** Cozy already has a per-subsystem venv (assistant, wakeword, stt-finetune per `AGENTS.md`), so a full-blown REPL kernel is overkill. But three concrete sub-patterns transfer: (a) the **bootstrap lock** — if Cozy ever does auto-install on first run, copy the `acquireBootstrapLock` `mkdir`-with-pid-file pattern instead of touching `fcntl`/flock; it is cross-platform and recovers from crashed predecessors via the `lockMissingPidIsStale` check. (b) the **`mkdirSync({ recursive: true })` then write** pattern (config.ts `appendRotatingLog`, lines around 30-60): always create the directory immediately before the first write, never as a separate "init" step. (c) The "version file next to the venv" check (`BOOTSTRAP_VERSION_FILE` + `bootstrapVersionCurrent`) lets you skip re-bootstrap when nothing has changed — apply the same idea to Cozy's model files: stamp `assistant/model/.cozy-version` on first use and re-resolve only when the file is missing or the user changes `voice.cfg`.

---

## 6. Naming conventions, error handling, observability

**Sources:**
- Logging: `/tmp/prime-agent-src/packages/coding-agent/src/core/logging.ts` (23 lines, full file) and `appendRotatingLog` in `config.ts` around lines 30-60.
- Structured errors: `/tmp/prime-agent-src/packages/coding-agent/src/core/session-cwd.ts` (`MissingSessionCwdError` with attached `issue` data), `/tmp/prime-agent-src/packages/coding-agent/src/core/kernel/shared.ts` (`KernelBusyAfterInterruptError`).
- Module pattern: every file has a one-line `getLogger("coding-agent.<area>")` at the top (e.g. skills.ts line 9: `const log = getLogger("coding-agent.skills");`) and logs as `log.warn("message", { key: value })` not as a stringified blob.

```ts
// logging.ts, full file
import { type LogEntry, setLogSink, stringifyLogEntry } from "@earendil-works/pi-ai";
import { appendRotatingLog, getAgentLogPath } from "../config.js";
const AGENT_LOG_MAX_BYTES = 20 * 1024 * 1024;
let context: Record<string, unknown> = {};
export function setLogContext(fields: Record<string, unknown>): void {
    Object.assign(context, fields);
}
export function installFileLogSink(fields?: Record<string, unknown>): void {
    context = { pid: process.pid, ...fields };
    setLogSink((entry: LogEntry) => {
        appendRotatingLog(getAgentLogPath(), stringifyLogEntry({ ...entry, ...context }), AGENT_LOG_MAX_BYTES);
    });
}
```

```ts
// config.ts appendRotatingLog excerpt
export function appendRotatingLog(logPath: string, message: string, maxBytes: number = MAX_LOG_BYTES): void {
    try {
        mkdirSync(dirname(logPath), { recursive: true });
        try {
            if (existsSync(logPath) && statSync(logPath).size > maxBytes) {
                rmSync(`${logPath}.old`, { force: true });
                renameSync(logPath, `${logPath}.old`);
            }
        } catch { /* Keep appending rather than dropping the log on a rotation failure. */ }
        appendFileSync(logPath, `${message}\n`);
    } catch { /* A read-only or missing log dir must never break the caller. */ }
}
```

The discipline: every `try/catch` is paired with a comment explaining **why** the failure is being swallowed (rotation is best-effort, log writes must not crash the agent). Domain errors are subclasses of `Error` with a `name` and structured payload (`this.issue = ...; this.name = "MissingSessionCwdError"`).

**How Cozy would adapt it.** Centralize logging in `cozy/log.py`: a `get_logger(name: str) -> logging.Logger` that prefixes every record with `pid=… session_id=…` (mirror `setLogContext` in `logging.ts`). One rotating JSONL file at `~/.cozy/logs/cozy.jsonl` (mirror the `appendRotatingLog` function verbatim — 20 MB cap, `.old` rollover, `try/except` at every step so a logging failure never breaks the assistant). Adopt the `Error` subclass pattern: `WakeWordError`, `SttTimeoutError`, `LlmContextOverflowError` each carry a structured `.details` dict. The `getLogger("coding-agent.<area>")` namespacing maps cleanly to Python's `logging.getLogger("cozy.<area>")`. Most importantly: every catch site gets a one-line comment about what was swallowed and why. This is the single highest-leverage style fix you can copy.

---

## 7. Model fallback / restore

**Source:** `/tmp/prime-agent-src/packages/coding-agent/src/core/model-resolver.ts` (`restoreModelFromSession` at the bottom of the file, lines ~700-770) and `/tmp/prime-agent-src/packages/coding-agent/src/core/agent-session.ts` (`setModel` and `cycleModel` near the top).

Prime Agent's approach is: every session stores the chosen `provider + modelId` in the transcript header. On resume, `restoreModelFromSession` looks for the saved model in the *current* available set; if not found, it tries (a) the *currently active* model (if any), (b) `findPreferredDefaultModel` (Prime Inference's pinned default → provider-level default → first available), and attaches a one-time `fallbackMessage` string the user sees exactly once.

```ts
// model-resolver.ts, lines ~700-770 (restoreModelFromSession)
if (restoredModel) return { model: restoredModel, fallbackMessage: undefined };
const registeredModel = modelRegistry.find(savedProvider, savedModelId);
const reason = !registeredModel ? "model no longer exists"
    : !modelRegistry.hasConfiguredAuth(registeredModel) ? "no auth configured"
    : "model is not available";
const fallbackCurrentModel = currentModel && (!isPrivatePrimeInferenceModel(currentModel) || availableCurrentModel)
    ? (availableCurrentModel ?? currentModel) : undefined;
if (fallbackCurrentModel) return { model: fallbackCurrentModel,
    fallbackMessage: `Could not restore model ${savedProvider}/${savedModelId} (${reason}). Using ${fallbackCurrentModel.provider}/${fallbackCurrentModel.id}.` };
if (availableModels.length > 0) {
    const fallbackModel = findPreferredDefaultModel(availableModels) ?? availableModels[0];
    return { model: fallbackModel, fallbackMessage: `... Using ${fallbackModel.provider}/${fallbackModel.id}.` };
}
```

The `setModel` flow (agent-session.ts near the top) persists the change via `sessionManager.appendModelChange(provider, modelId)` and updates the default in `settingsManager.setDefaultModelAndProvider`. The `cycleModel` flow walks `availableModels` and replaces; it never silently drops the model.

Companion logic in `agent-session-runtime.ts` lines around the `modelFallbackMessage` getter:
```ts
get modelFallbackMessage(): string | undefined {
    if (isNoModelsAvailableMessage(this._modelFallbackMessage) && this._session.model) {
        return undefined;
    }
    return this._modelFallbackMessage;
}
```
The "no models available" warning is a **stateful flag** that is automatically cleared once a model is actually selected — same logic a single-user Cozy needs when, e.g., the user pulls their ANTHROPIC key and the next run finds a fallback.

**How Cozy would adapt it.** Cozy's `voice.cfg` already names the LLM; copy this pattern into `cozy/models.py`. On every wake→LLM invocation, try the configured model; on `AuthenticationError` or `NotFoundError`, fall back through a list in `voice.cfg` (e.g. `model: claude-sonnet-4.5` then `fallbacks: [claude-haiku-4-5, gpt-5-mini]`) and say a one-time TTS message: *"Sorry, that model isn't available right now — switching to Haiku."* Persist the new choice back to `voice.cfg` only on explicit user confirmation, otherwise re-prompt next session. The detect-context-overflow helper at `/tmp/prime-agent-src/packages/ai/src/utils/overflow.ts` (full file, 144 lines) is a goldmine — it has regex patterns for Anthropic, OpenAI, Google, Mistral, OpenRouter, llama.cpp, LM Studio, etc. Lift the regex list into a `is_context_overflow(error_msg: str) -> bool` and use it to trigger a context-summary compaction pass before retrying.

---

## 8. Session / checkpoint / cron / recovery

**Source:** `/tmp/prime-agent-src/packages/coding-agent/src/core/session-manager.ts` (`SessionManager._persist` at line ~1060 and the entry types at lines 80-200) and `/tmp/prime-agent-src/packages/coding-agent/src/core/cron-jobs.ts` (`AgentCronJobStore` at lines ~270-500).

The session store is a single JSONL file at `~/.prime/agent/sessions/<session-id>.jsonl` where each line is a typed entry (`{ type, id, parentId, timestamp, ... }`). The header is a `SessionHeader` with `id`, `cwd`, `git`, `parentSession`, `rlmDepth` (session-manager.ts lines 86-94). Persistence is append-only, with the first write being a full rewrite:

```ts
// session-manager.ts, SessionManager._persist
_persist(entry: SessionEntry): void {
    if (!this.persist || !this.sessionFile) return;
    const hasAssistant = this.fileEntries.some((e) => e.type === "message" && e.message.role === "assistant");
    const shouldPersistWithoutAssistant = entry.type === "session_state" || entry.type === "session_info";
    if (!hasAssistant && !shouldPersistWithoutAssistant) { this.flushed = false; return; }
    if (!this.flushed || !existsSync(this.sessionFile)) {
        this._rewriteFile();
        this.flushed = true;
    } else {
        mkdirSync(dirname(this.sessionFile), { recursive: true });
        appendFileSync(this.sessionFile, `${JSON.stringify(entry)}\n`);
        this._notifyPersistListeners();
    }
}
```

Entry types include `message`, `compaction`, `branch_summary`, `model_change`, `thinking_level_change`, `service_tier_change`, `custom`, `session_state`. The "tree" structure (`id`/`parentId`) means you can fork a session mid-stream without copying. Compaction is itself an entry: `{ type: "compaction", summary, firstKeptEntryId, tokensBefore }` and the on-load code skips everything before `firstKeptEntryId` (compaction.ts around lines 60-95).

`cron-jobs.ts` uses a `proper-lockfile`-guarded JSON file at `~/.prime/agent/cron-jobs.json` for schedule records. Each job has `{ id, schedule: { kind, expression, intervalMs }, sessionId, sessionFile, prompt, nextRunAt, lastRunAt, lastError, runCount }` (lines 38-60). On schedule tick the store re-reads the file, finds due jobs, marks `lastRunAt`, runs the prompt through the session, records the result.

**How Cozy would adapt it.** Even though Cozy is single-session and CLI-only, a JSONL transcript per day is the right pattern. Create `assistant/sessions/<YYYY-MM-DD>.jsonl` (one file per day, easier to grep/rotate) with three entry types: `{ type: "turn", id, parent_id, ts, user_text, assistant_text, stt_ms, llm_ms, tts_ms }`, `{ type: "tool_call", id, parent_id, ts, tool_name, args, result, is_error }`, and `{ type: "state_change", id, parent_id, ts, field, value }` (covers model swap, volume, wake threshold). Append-only with the first entry being a full rewrite of the day's file (mirror the `flushed` flag). This gives you: (1) instant resume on `cozy --continue` last-day, (2) the data your SFT pipeline in `assistant/rlm_harness/` already ingests, and (3) a natural place to record token / latency / error stats for the dashboard. The `cron-jobs.ts` pattern transfers directly to "remind me at 3pm" intents: a `~/.cozy/reminders.json` with `[{ id, fire_at, prompt, status, run_count }]`, read at wake-time, prompts re-enqueued. The `captureGitContext` pattern in `utils/git.ts` is also reusable — stamp the Cozy git SHA into the session header so you can diff quality across versions.

---

## 9. Two more small things worth stealing

**a) `OutputAccumulator` and temp-file pointer.** `tools/output-accumulator.ts` writes tool output to `/tmp/<prefix>-<rand>.log` and returns the path inline. Adopt the same `cozy/cache/<uuid>.txt` pattern for any LLM-bound payload over 4 KB (STT transcripts, calendar JSON, list of files). The inline message says "full content at $path" and the LLM uses the existing `read_file` tool if it actually needs more.

**b) `find-skills`-style discovery at wake time.** The discovery rule in `skills.ts` (lines 158-200) — "if a directory contains `SKILL.md`, treat it as a skill root and do not recurse further" — is exactly the rule Cozy needs to support a `~/.cozy/skills/` directory without any config. Steal the YAML frontmatter parser wholesale from `utils/frontmatter.ts` rather than hand-rolling one; the regex for the name (`/^[a-z0-9-]+$/`) catches 95% of mistakes before runtime.

---

## What's NOT worth copying

- The full daemon supervisor / RPC protocol (`docs/agent-connection.md`, `modes/daemon/`). Cozy is a CLI; you don't need cross-process recovery or socket-based worker dispatch.
- Recursive subagents via `rlm("...")` and the `RLMSpawnHandle` flow (`prime-agent-runtime/src/rlm/__init__.py`). Cozy has a single user, not a tree of delegating agents.
- OAuth / `auth-storage.ts` (~900 lines). Cozy uses env-var API keys, not subscription logins.
- The TUI components and keybinding system (`packages/tui/`, ~hundreds of files). Cozy's `runtime.py` is a CLI; reuse your existing prompt UX.
- The model-registry's `refreshAvailableModels` + private model cache. With a single env-var API key, you don't need the network-side freshness layer.

---

## Suggested Cozy integration order

1. Drop `truncate.py` and `format_harness_state_for_prompt` (item 3 and item 1) into `assistant/`. Build `.cozy/harness/` and `/refine` flow on day one — this is the single biggest UX win (it makes Cozy learn user preferences).
2. Create `assistant/skills.py` (item 4) and ship two example skills (`calendar`, `notes`) so the format has proof.
3. Build the `ToolDefinition`-style registry (item 2) around TTS, STT, and a `set_reminder` tool; pass it to the LLM on every turn.
4. Adopt the logging and error-subclass conventions (item 6) in a single cleanup pass — cheap, high consistency payoff.
5. Add the model-fallback chain (item 7) and the overflow regex list so Cozy doesn't crash mid-conversation.
6. JSONL session log + reminders (item 8) — but only after the SFT pipeline needs the data, not before.
