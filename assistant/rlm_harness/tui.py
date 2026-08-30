"""Cozy text-mode REPL with / commands. Stdlib only.

The TUI replaces the bare input() loop in --text mode and adds
/ commands for harness introspection and control:

  /help            this list
  /stats           harness stats (turns, summary, plugins loaded, RAM hint)
  /reset           clear in-RAM trace (keeps the on-disk JSONL)
  /compact         force summary compaction now
  /skills          list discovered skills
  /skill <name>    show a skill's full body
  /memory          list memory entries
  /memory add <k> <v>  add a fact
  /memory rm <k>       remove a fact
  /refine          re-render the system prompt from current state
  /recall <id>     pull a specific turn back from the on-disk trace
  /run <task>      spawn a child agent (RLM)
  /quit            exit

The TUI prints colored output when stdout is a TTY (and NO_COLOR is
not set). It does NOT use prompt_toolkit or textual - just readline
and ANSI codes.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# ---------- color helpers (NO_COLOR support)
USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

def _ansi(code: str) -> str:
    return f"\033[{code}m" if USE_COLOR else ""

def dim(s: str) -> str:    return _ansi("2") + s + _ansi("0")
def green(s: str) -> str:  return _ansi("32") + s + _ansi("0")
def red(s: str) -> str:    return _ansi("31") + s + _ansi("0")
def yellow(s: str) -> str: return _ansi("33") + s + _ansi("0")
def cyan(s: str) -> str:   return _ansi("36") + s + _ansi("0")
def bold(s: str) -> str:   return _ansi("1") + s + _ansi("0")

# ---------- TUI state
class TUI:
    def __init__(self, harness, executor, run_llm):
        self.h = harness
        self.exec = executor
        self.run_llm = run_llm
        self.history: list[str] = []
        self.h_idx = -1
        self.last_action: tuple[str, dict] | None = None
        self.commands = self._collect_commands()

    def _collect_commands(self) -> dict[str, str]:
        return {
            "/help":    "list all / commands",
            "/stats":   "show harness stats (turns, summary, plugins)",
            "/reset":   "clear in-RAM trace (keeps disk)",
            "/compact": "force summary compaction now",
            "/skills":  "list discovered skills",
            "/skill":   "/skill <name>   show a skill's full body",
            "/memory":  "/memory [add <k> <v> | rm <k> | list]   CRUD facts about the user",
            "/notes":   "/notes [add <title> <body> | rm <id> | list]   persistent rules the LLM reads every turn",
            "/refine":  "re-build and apply the system prompt from current state",
            "/refine":  "re-render system prompt from current state",
            "/recall":  "/recall <id>     pull a turn back from on-disk trace",
            "/run":     "/run <task>      spawn a child agent (RLM)",
            "/quit":    "exit (also: /exit, /q, EOF)",
        }

    def banner(self) -> None:
        if self.h is None:
            print()
            print(bold("╭─ Cozy (rule router only) ─╮"))
            print(dim("  no LLM loaded; using intents.py regex router"))
            print(dim("  type /help for commands"))
            print(bold("╰───────────────────────────────╮"))
            print()
            return
        s = self.h.stats()
        plugins = s.get("plugins_loaded") or ["(none)"]
        print()
        print(bold("╭─ Cozy " + ("(fast harness) " if self.h.cfg.use_llm else "") + "─╮"))
        print(f"  plugins : " + cyan(", ".join(plugins)))
        print(f"  turns   : {s.get('recent_turns', 0)} recent / "
              f"~{s.get('approx_tokens', 0)} tokens")
        print(f"  summary : {s.get('summary_chars', 0)} chars")
        print(f"  history : {len(self.history)} commands")
        print(dim("  type /help for commands, or just talk"))
        print(bold("╰───────────────────────────────╮"))
        print()

    def prompt(self) -> str:
        # Readline-style command history (without the readline module's
        # setup complexity - just a basic up-arrow style for now).
        # We use Python's built-in input() with a small history buffer
        # the user navigates with /recall or just by retyping.
        try:
            line = input(cyan("cozy> ") if USE_COLOR else "cozy> ")
        except (EOFError, KeyboardInterrupt):
            return "/quit"
        line = line.strip()
        if line:
            self.history.append(line)
            self.h_idx = -1
        return line

    def run_forever(self) -> None:
        self.banner()
        while True:
            try:
                line = self.prompt()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not line:
                continue
            if line.startswith("/"):
                if self._handle_slash(line):
                    return  # /quit
                continue
            self._handle_user(line)

    def _handle_slash(self, line: str) -> bool:
        """Return True if TUI should exit."""
        parts = line.split(maxsplit=1)
        cmd = parts[0]
        arg = parts[1] if len(parts) > 1 else ""
        if cmd in ("/quit", "/exit", "/q"):
            print(dim("bye."))
            return True
        if cmd == "/help":
            self._cmd_help()
        elif cmd == "/stats":
            self._cmd_stats()
        elif cmd == "/reset":
            self._cmd_reset()
        elif cmd == "/compact":
            self._cmd_compact()
        elif cmd == "/skills":
            self._cmd_skills()
        elif cmd == "/skill":
            self._cmd_skill(arg)
        elif cmd == "/memory":
            self._cmd_memory(arg)
        elif cmd == "/notes":
            self._cmd_notes(arg)
        elif cmd == "/refine":
            self._cmd_refine()
        elif cmd == "/recall":
            self._cmd_recall(arg)
        elif cmd == "/run":
            self._cmd_run(arg)
        else:
            print(red(f"unknown command: {cmd}"))
            print(dim("  type /help for the list"))
        return False

    def _cmd_help(self) -> None:
        print(bold("/ commands:"))
        for cmd, desc in self.commands.items():
            print(f"  {cmd:20s}  {dim(desc)}")

    def _cmd_stats(self) -> None:
        if self.h is None:
            print(yellow("no harness loaded (rule-router mode)"))
            return
        s = self.h.stats()
        print(bold("harness stats:"))
        for k, v in s.items():
            print(f"  {k:18s}  {v}")
        # RAM: chars in recent turns (rough proxy)
        ram_kb = sum(len(t.content or "") + len(str(t.tool_calls)) for t in self.h.trace.recent) // 1024
        print(f"  ram_recent_kb     {ram_kb}")
        # Per-turn metrics from the trace
        recent = self.h.trace.recent
        if recent:
            # Average turn length, tool-call ratio
            tool_calls = sum(1 for t in recent if t.tool_calls)
            chat_turns = sum(1 for t in recent if t.role == "assistant" and t.content)
            avg_len = sum(len(t.content or "") for t in recent) / max(1, len(recent))
            print(dim(f"  per-turn: avg={avg_len:.0f}ch, tool_calls={tool_calls}, chat_turns={chat_turns}"))
        # Plugin warm/cold
        if self.h.plugins:
            print(dim("  plugins:"))
            for name, p in self.h.plugins.items():
                state = "warm" if getattr(p, "_loaded", False) else "cold"
                print(dim(f"    {name:8s}  {state}"))
        # Memory + notes + skills counts
        from .state import MemoryStore, NotesStore, RefinementStore
        m = MemoryStore(); ns = NotesStore()
        print(dim(f"  memory: {len(m.list())} facts   notes: {len(ns.list())} rules"))
        # Refinement log
        recent_refs = RefinementStore().tail(5)
        if recent_refs:
            print(dim("  recent refinements:"))
            for r in recent_refs:
                print(dim(f"    - {r.get('ts', 0):.0f} {r.get('kind')} {r.get('ref')}"))

    def _cmd_reset(self) -> None:
        if self.h is None:
            print(red("no harness to reset"))
            return
        n = len(self.h.trace.recent)
        self.h.reset()
        print(yellow(f"cleared {n} in-RAM turns (disk trace kept at "
                     f"{self.h.cfg.trace_file})"))

    def _cmd_compact(self) -> None:
        if self.h is None:
            print(red("no harness to compact"))
            return
        before = len(self.h.trace.recent)
        self.h.trace.compact()
        after = len(self.h.trace.recent)
        print(green(f"compacted: {before} -> {after} recent turns, "
                    f"summary now {len(self.h.trace.summary)} chars"))

    def _cmd_skills(self) -> None:
        from .skills import discover_skills
        skills = discover_skills()
        if not skills:
            print(dim("no skills installed in ~/.cozy/skills/"))
            return
        for s in skills:
            print(f"  {cyan(s.get('name', '?')):30s}  {s.get('description', '')}")

    def _cmd_skill(self, name: str) -> None:
        if not name:
            print(red("usage: /skill <name>"))
            return
        from .skills import _valid_name
        if not _valid_name(name):
            print(red(f"invalid skill name: {name!r}"))
            return
        p = Path.home() / ".cozy" / "skills" / name / "SKILL.md"
        if not p.exists():
            print(red(f"no skill at {p}"))
            return
        print(p.read_text())

    def _cmd_memory(self, arg: str) -> None:
        from .state import MemoryStore
        m = MemoryStore()
        if not arg or arg.strip() == "list":
            rows = m.list()
            if not rows:
                print(dim("  (no memory yet — try /memory add <key> <value>)"))
            for k, v in rows:
                print(f"  {cyan(k):24s}  {v}")
            return
        parts = arg.split(maxsplit=1)
        if parts[0] == "add":
            kv = (parts[1] if len(parts) > 1 else "").split(maxsplit=1)
            if len(kv) != 2:
                print(red("usage: /memory add <key> <value>"))
                return
            m.add(kv[0], kv[1])
            print(green(f"added memory {kv[0]!r}"))
        elif parts[0] == "rm":
            if m.remove(parts[1]):
                print(green(f"removed memory {parts[1]!r}"))
            else:
                print(red(f"no memory {parts[1]!r}"))
        else:
            print(red(f"unknown: {arg!r} (try /memory add|rm|list)"))

    def _cmd_notes(self, arg: str) -> None:
        from .state import NotesStore
        n = NotesStore()
        if not arg or arg.strip() == "list":
            notes = n.list()
            if not notes:
                print(dim("  (no notes — try /notes add <title> <body>)"))
            for note in notes:
                first_line = (note.content or "").strip().splitlines()[0] if note.content else ""
                print(f"  {cyan(note.id):14s}  {note.title}: {first_line[:80]}")
            return
        parts = arg.split(maxsplit=2)
        if parts[0] == "add":
            if len(parts) < 3:
                print(red("usage: /notes add <title> <body>"))
                return
            note = n.add(parts[1], parts[2])
            print(green(f"added note {note.id!r}: {parts[1]!r}"))
        elif parts[0] == "rm":
            if n.remove(parts[1]):
                print(green(f"removed note {parts[1]!r}"))
            else:
                print(red(f"no note {parts[1]!r}"))
        else:
            print(red(f"unknown: {arg!r} (try /notes add|rm|list)"))

    def _cmd_refine(self) -> None:
        """Re-build the system prompt from current state (memory, notes, skills)
        and apply it to the live harness. Persists to ~/.cozy/state/refined_prompt.txt
        so the change survives a restart.
        """
        if self.h is None:
            print(red("no harness to refine"))
            return
        from .state import build_system_prompt, RefinementStore
        from cozy_log import log_event  # sibling module, not in this package
        from pathlib import Path
        new_prompt = build_system_prompt(self.h)
        # Persist to disk so a restart picks it up
        ref_path = Path.home() / ".cozy" / "state" / "refined_prompt.txt"
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        from .harness_fast import _atomic_write_text
        _atomic_write_text(ref_path, new_prompt)
        # Apply to live harness
        old_prompt = self.h.system
        self.h.system = new_prompt
        # Log the refinement
        RefinementStore().record(
            "system_prompt.update", "main",
            before=old_prompt[:200], after=new_prompt[:200],
            note="applied via /refine")
        log_event("refine", old_len=len(old_prompt), new_len=len(new_prompt))
        print(green("system prompt refined and applied:"))
        print(dim("─" * 60))
        print(new_prompt)
        print(dim("─" * 60))
        print(dim(f"persisted to {ref_path}"))

    def _cmd_recall(self, arg: str) -> None:
        if self.h is None:
            print(red("no harness trace to recall from"))
            return
        if not arg or not arg.isdigit():
            print(red("usage: /recall <line_number>"))
            return
        n = int(arg)
        if not self.h.cfg.trace_file.exists():
            print(red("no trace file"))
            return
        with open(self.h.cfg.trace_file) as f:
            lines = f.readlines()
        if n < 1 or n > len(lines):
            print(red(f"line {n} out of range (1..{len(lines)})"))
            return
        print(green(f"turn {n}/{len(lines)}:"))
        print(dim("─" * 60))
        print(lines[n-1].rstrip())
        print(dim("─" * 60))

    def _cmd_run(self, task: str) -> None:
        if not task:
            print(red("usage: /run <task>"))
            return
        if self.h is None:
            print(red("no harness to spawn a child from"))
            return
        from .rlm import rlm_delegate
        result = rlm_delegate(task, self.h, allow=None, depth=1)
        print(green(f"child result ({len(result)} chars):"))
        print(f"  {result[:300]}")

    def _handle_user(self, text: str) -> None:
        # The user typed something.
        t0 = time.time()
        if self.h is None:
            # No harness at all - rule router only.
            self._rule_router_dispatch(text)
            return

        # With a harness, ALWAYS go through decide() - it lazily
        # loads the LLM plugin on first call. The "is loaded" checks
        # I had before were wrong: the plugin is _loaded after decide().
        llm_plugin = self.h.plugins.get("llm")
        if llm_plugin and llm_plugin._loaded:
            # Already loaded - use live streaming for nice UX
            self._handle_user_streaming(text)
        else:
            # First call - sync decide triggers lazy load
            name, args = self.h.decide(text)
            dt = time.time() - t0
            self._render_decision(name, args, dt)

    def _rule_router_dispatch(self, text: str) -> None:
        """Last-resort rule router when no harness is available."""
        from intents import route as intent_route
        res = intent_route(text)
        tool = res.get("tool")
        if tool in ("unhandled", "none", None):
            print(f"  {dim('(no rule matched, type /help to see tools)')}")
            return
        from executor import execute
        mapping = {
            "set_volume":   ("system.volume.set",
                                {"level": res["args"].get("level",
                                   max(0, min(100, 50 + res["args"].get("delta", 0))))}),
            "open_app":     ("app.open", {"name": res["args"].get("app", "")}),
            "browser_search":("browser.search",
                                {"query": res["args"].get("q", "")}),
            "screenshot":   ("screenshot.take", {}),
            "query_time":   ("time.now", {}),
        }
        if tool in mapping:
            name, args = mapping[tool]
            r = self.exec(name, args)
            print(f"  {green('OK') if r['ok'] else red('FAIL')} {r['output'][:200]}")

    def _handle_user_streaming(self, text: str) -> None:
        llm = self.h.plugins["llm"]
        # Inject a user turn
        from .harness_fast import Turn
        self.h.trace.append(Turn(role="user", content=text, producer="user"))
        msgs, _ = self.h.trace.build_prompt(self.h.system, self.h.tools.repr)
        # Stream to a local buffer + on-screen
        buf = []
        printed_so_far = 0
        print(f"  {cyan('cozy>')} ", end="", flush=True)
        def on_token(piece: str) -> None:
            # The piece is ALREADY a text chunk. Just print it.
            if not piece:
                return
            buf.append(piece)
            try:
                print(piece, end="", flush=True)
            except UnicodeEncodeError:
                # Handle non-ASCII pieces on non-UTF-8 terminals
                print(piece.encode("ascii", "replace").decode(), end="", flush=True)
        try:
            # Wall-clock cap: 30s. Prevents a hung streamer from
            # blocking the TUI forever.
            import signal
            class _StreamTimeout(Exception): pass
            def _alarm(signum, frame):
                raise _StreamTimeout("stream timed out")
            old_handler = signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(30)
            try:
                raw = llm.generate(msgs, on_token=on_token)
            except _StreamTimeout:
                print(f"\n  {yellow('(stream timed out, showing partial result)')}")
                raw = "".join(buf)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except Exception as exc:
            print(f"\n  {red('LLM error:')} {exc}")
            return
        print()  # newline after streaming
        # Extract tool call
        from .fasttool import extract_tool_call
        name, args = extract_tool_call(raw)
        if name:
            print(f"  {dim('->')} {yellow(name)} {dim(str(args)[:80])}")
            try:
                result = self.exec(name, args or {})
                if result.get("ok"):
                    out = result.get("output", "")
                    print(f"  {green('OK')} {out[:200]}")
                else:
                    print(f"  {red('FAIL')} {result.get('output', '')[:200]}")
                from rlm_harness.truncate import truncate_tail
                spoken = truncate_tail(result.get("output", ""))
                self.last_action = (name, args or {})
            except Exception as exc:
                print(f"  {red('ERROR')} {exc}")
        else:
            # Record the streamed reply in the trace
            from .harness_fast import Turn
            text = "".join(buf).strip()
            import re as _re
            text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.S).strip()
            # Strip Qwen3 special tokens (they shouldn't be spoken)
            text = _re.sub(r"<\|im_end\|>", "", text)
            text = _re.sub(r"<\|im_start\|>", "", text)
            text = _re.sub(r"<\|endoftext\|>", "", text)
            text = text.strip()
            self.h.trace.append(Turn(role="assistant", content=text, producer="model"))
            self.last_action = None

    def _render_decision(self, name: str, args: dict, dt: float) -> None:
        if name:
            print(f"  {dim('->')} {yellow(name)} {dim(str(args)[:80])} ({dt:.2f}s)")
            try:
                result = self.exec(name, args or {})
                if result.get("ok"):
                    out = result.get("output", "")
                    print(f"  {green('OK')} {out[:200]}")
                else:
                    print(f"  {red('FAIL')} {result.get('output', '')[:200]}")
                from rlm_harness.truncate import truncate_tail
                spoken = truncate_tail(result.get("output", ""))
                self.last_action = (name, args or {})
            except Exception as exc:
                print(f"  {red('ERROR')} {exc}")
        else:
            for t in reversed(self.h.trace.recent):
                if t.role == "assistant" and t.content:
                    print(f"  {cyan('cozy>')} {t.content}")
                    self.last_action = None
                    break


def run_tui(harness, executor, run_llm) -> None:
    """Entry point: start the TUI loop with the given harness."""
    TUI(harness, executor, run_llm).run_forever()
