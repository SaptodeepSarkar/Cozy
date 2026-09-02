"""Main agent loop: VLM planner + VLA grounder + Pop!_OS driver.

A :class:`VisionRunner` is the entry point for end-to-end desktop
control. The flow:

  1. **Snapshot** the live OS context (active window, open windows,
     focused element, working area) via :class:`OSContextCollector`.
  2. **Plan** with the VLM: hand it the goal + screenshot + OS
     context, get back a precise, ordered, *checkable* todo list.
  3. For each todo item, the VLA (UI-TARS-2B) executes a tight loop
     of `step()` calls until it emits ``finished()`` or the step
     budget is exhausted.
  4. After every todo item, the same VLM verifies with ``verify()``
     that the todo is done. If yes, move to the next. If no, retry
     or re-plan. **No model swap** — both vision models live on
     the dGPU simultaneously (NF4 4-bit, ~3.9 GB of 6 GB).
  5. The full trace (plan, actions, verifications, contexts) is
     written to JSONL for the SFT trainer to mine preference pairs.

A :class:`TaskSession` wraps :class:`VisionRunner` with a
human-friendly streaming progress log and graceful Ctrl-C handling.
"""
from __future__ import annotations
import os

import json
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from .context import OSContext, OSContextCollector
from .driver import PopOSDriver, WindowInfo
from .grounder import (
    Action,
    ClickAction,
    FinishedAction,
    Grounder,
    HotkeyAction,
    ScrollAction,
    TypeAction,
    WaitAction,
)
from .planner import Planner, Plan, TodoItem, Answer, VerifyResult
from .reward import reward, RewardInfo

try:
    from .cozy_swap import swap_out, swap_in
    _HAVE_COZY_SWAP = True
except Exception:
    _HAVE_COZY_SWAP = False

HERE = Path(__file__).resolve().parent
TRACE_DIR = HERE.parent / "data" / "traces"
TRACE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class VisionResult:
    success: bool
    score: float
    goal: str
    todo: list[dict]
    actions: list[dict]
    contexts: list[dict]
    reward: RewardInfo
    duration_s: float
    error: Optional[str] = None


# Progress callback type
ProgressFn = Callable[[str, dict], None]


def _default_progress(event: str, data: dict) -> None:
    """Print a one-line progress update to stdout."""
    ts = time.strftime("%H:%M:%S")
    if event == "start":
        print(f"[{ts}] ▶ start: {data['goal']!r}")
    elif event == "context":
        ctx = data["context"]
        win = ctx.get("active_window", "(none)")
        n = len(ctx.get("open_windows", []))
        print(f"[{ts}]   screen: active={win!r}, {n} windows open, size={ctx.get('screen_size')}")
    elif event == "plan":
        n = len(data["todo"])
        print(f"[{ts}]   plan: {n} todo items")
        for i, t in enumerate(data["todo"], 1):
            print(f"[{ts}]     {i}. {t['action']}  target={t.get('target','')!r}")
    elif event == "todo_start":
        print(f"[{ts}]   todo {data['idx']+1}/{data['total']}: {data['todo']['action']!r}")
    elif event == "action":
        a = data["action"]
        ms = data.get("latency_ms")
        ms_str = f" ({ms:.0f}ms)" if ms else ""
        print(f"[{ts}]     step {data['step']}: {a}{ms_str}")
    elif event == "verify":
        mark = "✓" if data["done"] else "✗"
        print(f"[{ts}]   verify: {mark} {data.get('reason','')[:100]}")
    elif event == "todo_done":
        print(f"[{ts}]   todo {data['idx']+1} done")
    elif event == "todo_failed":
        print(f"[{ts}]   todo {data['idx']+1} FAILED after retries: {data.get('reason')}")
    elif event == "replan":
        print(f"[{ts}]   replan: {len(data['todo'])} new todos")
    elif event == "done":
        ok = "✓" if data["success"] else "✗"
        print(f"[{ts}] ◀ done ({ok}): {data['reason'][:100]} in {data['duration_s']:.1f}s")
    elif event == "error":
        print(f"[{ts}] ✗ error: {str(data['error'])[:200]}")
    elif event == "interrupt":
        print(f"[{ts}] ! interrupted by user")


class VisionRunner:
    def __init__(
        self,
        planner: Optional[Planner] = None,
        grounder: Optional[Grounder] = None,
        driver: Optional[PopOSDriver] = None,
        collector: Optional[OSContextCollector] = None,
        max_todo_items: int = 12,
        max_steps_per_todo: int = 8,
        max_retries: int = 2,
        step_delay_s: float = 0.4,
        save_screenshots: bool = True,
        trace_name: Optional[str] = None,
        on_progress: Optional[ProgressFn] = None,
    ) -> None:
        self.planner = planner or Planner()
        self.grounder = grounder or Grounder()
        self.driver = driver or PopOSDriver()
        self.collector = collector or OSContextCollector()
        self.max_todo_items = max_todo_items
        self.max_steps_per_todo = max_steps_per_todo
        self.max_retries = max_retries
        self.step_delay_s = step_delay_s
        self.save_screenshots = save_screenshots
        self.trace_name = trace_name or f"trace_{int(time.time())}.jsonl"
        self.on_progress = on_progress or _default_progress
        self._interrupted = False

    def warmup(self) -> None:
        import torch
        try:
            self.planner.load()
        except torch.OutOfMemoryError as e:
            free_gb = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0
            used_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            raise RuntimeError(
                f"VLM failed to load: {e}.\n"
                f"  GPU: {used_gb:.2f} GB used, {free_gb:.2f} GB free.\n"
                f"  The dGPU may be full. Try:\n"
                f"    1. COZY_VISION_SWAP_COZY=0 bash run.sh task ...  (don't swap cosy)\n"
                f"    2. Manually find and stop the cozy watchdog:\n"
                f"         pgrep -af assistant/runtime.py\n"
                f"       then kill the parent (probably the dsh-qa-box sidecar or prime agent)\n"
                f"    3. Run cozy-vision task before starting cozy voice"
            ) from e
        try:
            self.grounder.load()
        except torch.OutOfMemoryError as e:
            free_gb = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0
            used_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            raise RuntimeError(
                f"VLA failed to load after VLM was on GPU: {e}.\n"
                f"  GPU: {used_gb:.2f} GB used, {free_gb:.2f} GB free.\n"
                f"  Need ~1.5 GB more for the VLA. Try COZY_VISION_SWAP_COZY=0 or stop the cozy watchdog."
            ) from e

    def shutdown(self) -> None:
        self.planner.free()
        self.grounder.free()

    def interrupt(self) -> None:
        """Signal a graceful stop after the current step."""
        self._interrupted = True

    # ------------------------------------------------- main entry
    def run(self, goal: str, task: Optional[dict] = None,
            max_replans: int = 1) -> VisionResult:
        """Run ``goal`` end-to-end and return a :class:`VisionResult`.

        ``max_replans`` caps the number of times we ask the VLM to
        re-plan after a todo fails. Default 1. Set to 0 to never
        re-plan (just give up after one pass).
        """
        task = task or {"text": goal, "verifier": "noop"}
        trace_path = TRACE_DIR / self.trace_name
        t0 = time.perf_counter()
        actions: list[dict] = []
        contexts: list[dict] = []
        todo_dump: list[dict] = []
        self._interrupted = False
        self._replan_count = 0
        self._last_screen_hash: Optional[str] = None
        self._no_change_count = 0
        try:
            self.on_progress("start", {"goal": goal})

            # 1. Initial context + screenshot
            context = self.collector.snapshot(screen_size=self.driver.screen_size)
            contexts.append(context.to_dict())
            shot0 = self.driver.screenshot(save=self.save_screenshots)
            self._log(trace_path, {"ts": time.time(), "kind": "context", "context": context.to_dict()})
            self.on_progress("context", {"context": context.to_dict()})

            # 2. Plan
            plan = self.planner.plan(goal, shot0, context)
            todo = plan.todo[: self.max_todo_items]
            todo_dump = [t.to_dict() for t in todo]
            self._log(trace_path, {
                "ts": time.time(), "kind": "plan",
                "raw": plan.raw, "todo": todo_dump,
            })
            self.on_progress("plan", {"todo": todo_dump, "raw": plan.raw})

            if not todo:
                self.on_progress("error", {"error": "VLM returned an empty plan"})
                return VisionResult(
                    success=False, score=0.0, goal=goal, todo=[],
                    actions=[], contexts=contexts,
                    reward=RewardInfo(score=0.0, reason="empty plan"),
                    duration_s=time.perf_counter() - t0,
                    error="VLM returned no todo items",
                )

            # 3. Execute each todo item. VLM verifies between items.
            for tg_idx, todo_item in enumerate(todo):
                if self._interrupted:
                    self.on_progress("interrupt", {})
                    break
                self.on_progress("todo_start", {
                    "idx": tg_idx, "total": len(todo), "todo": todo_item.to_dict(),
                })
                todo_done = self._execute_todo(
                    tg_idx, todo_item, todo[:tg_idx], actions, contexts, trace_path,
                )
                if not todo_done:
                    self.on_progress("todo_failed", {
                        "idx": tg_idx, "reason": "VLA couldn't finish after retries",
                    })
                    if self._replan_count >= max_replans:
                        self.on_progress("error", {
                            "error": f"replan cap reached ({max_replans}); giving up"
                        })
                        break
                    # Re-plan from the new screen
                    try:
                        ctx3 = self.collector.snapshot(screen_size=self.driver.screen_size)
                        shot3 = self.driver.screenshot(save=self.save_screenshots)
                        replan = self.planner.plan(goal, shot3, ctx3, history=todo[:tg_idx + 1])
                        if replan.todo:
                            self._replan_count += 1
                            self.on_progress("replan", {
                                "todo": [t.to_dict() for t in replan.todo],
                                "count": self._replan_count,
                            })
                            actions.append({"replan": [t.to_dict() for t in replan.todo]})
                            # replace remaining todos with the replanned ones
                            todo = todo[:tg_idx + 1] + replan.todo
                        else:
                            todo_done = True
                    except Exception:
                        pass
                if todo_done:
                    self.on_progress("todo_done", {"idx": tg_idx})

            # 4. Reward on the final state
            after = self._snapshot_state()
            after.update({"contexts": contexts})
            before = {"actions": []}
            r = reward(task, before, after, {"actions": actions, "todo": todo_dump, "contexts": contexts})
            result = VisionResult(
                success=r.score >= 1.0,
                score=r.score,
                goal=goal,
                todo=todo_dump,
                actions=actions,
                contexts=contexts,
                reward=r,
                duration_s=time.perf_counter() - t0,
            )
            self.on_progress("done", {
                "success": result.success, "reason": r.reason,
                "duration_s": result.duration_s,
            })
            return result
        except Exception as e:
            err = f"{e}\n{traceback.format_exc()}"
            self.on_progress("error", {"error": err})
            return VisionResult(
                success=False, score=-1.0, goal=goal, todo=todo_dump,
                actions=actions, contexts=contexts,
                reward=RewardInfo(score=-1.0, reason=str(e)),
                duration_s=time.perf_counter() - t0,
                error=err,
            )

    def _execute_todo(
        self, tg_idx: int, todo_item: TodoItem, history: list[TodoItem],
        actions: list[dict], contexts: list[dict], trace_path: Path,
    ) -> bool:
        """Execute a single todo. Returns True if verified complete."""
        for retry in range(self.max_retries + 1):
            if self._interrupted:
                return False
            ctx2 = self.collector.snapshot(screen_size=self.driver.screen_size)
            contexts.append(ctx2.to_dict())
            shot = self.driver.screenshot(save=self.save_screenshots)
            self._log(trace_path, {"ts": time.time(), "kind": "context", "context": ctx2.to_dict()})
            # Bail if the screen is not changing across retries — VLA
            # is likely stuck on wrong coordinates or the desktop
            # has not responded to the actions we sent.
            screen_hash = self._hash_screen(shot)
            if self._last_screen_hash == screen_hash and retry > 0:
                self._no_change_count += 1
                if self._no_change_count >= 2:
                    self.on_progress("todo_failed", {
                        "idx": tg_idx,
                        "reason": "screen not changing; VLA likely stuck on wrong coordinates",
                    })
                    return False
            else:
                self._no_change_count = 0
            self._last_screen_hash = screen_hash
            sub_history = []
            finished_by_vla = False
            for step_idx in range(self.max_steps_per_todo):
                if self._interrupted:
                    return False
                try:
                    action = self.grounder.step(todo_item, shot, history=sub_history)
                except Exception as e:
                    actions.append({"todo_idx": tg_idx, "step": step_idx, "error": str(e)})
                    break
                payload = self._dispatch(action)
                entry = {
                    "todo_idx": tg_idx,
                    "todo": todo_item.to_dict(),
                    "step": step_idx,
                    "retry": retry,
                    "action": str(action),
                    "latency_ms": getattr(action, "latency_ms", None),
                    "payload": payload,
                }
                actions.append(entry)
                self._log(trace_path, {"ts": time.time(), "kind": "action", **entry})
                self.on_progress("action", entry)
                if isinstance(action, FinishedAction):
                    finished_by_vla = True
                    break
                sub_history.append(str(action))
                time.sleep(self.step_delay_s)
            # VLM verifies (same model, no swap)
            try:
                vshot = self.driver.screenshot(save=self.save_screenshots)
                vctx = self.collector.snapshot(screen_size=self.driver.screen_size)
                verify = self.planner.verify(todo_item, vshot, vctx)
                self._log(trace_path, {
                    "ts": time.time(), "kind": "verify",
                    "todo_idx": tg_idx, "retry": retry,
                    "done": verify.done, "reason": verify.reason,
                })
                self.on_progress("verify", {
                    "done": verify.done, "reason": verify.reason,
                })
                if verify.done:
                    return True
                if finished_by_vla:
                    finished_by_vla = False
                    continue
            except Exception as e:
                actions.append({"todo_idx": tg_idx, "verify_error": str(e)})
                if finished_by_vla:
                    return True
        return False

    @staticmethod
    def _hash_screen(img) -> str:
        """Stable 64-bit hash of a downsized grayscale image."""
        import hashlib
        g = img.convert("L").resize((64, 64))
        return hashlib.md5(g.tobytes()).hexdigest()
    def _dispatch(self, action: Action) -> dict:
        try:
            if isinstance(action, ClickAction):
                self.driver.click(action.x, action.y, button=action.button, count=action.count)
            elif isinstance(action, TypeAction):
                self.driver.type_text(action.text)
            elif isinstance(action, HotkeyAction):
                self.driver.hotkey(action.keys)
            elif isinstance(action, ScrollAction):
                self.driver.scroll(action.dx, action.dy)
            elif isinstance(action, WaitAction):
                time.sleep(action.seconds)
            elif isinstance(action, FinishedAction):
                pass
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _snapshot_state(self) -> dict:
        try:
            win = self.driver.get_active_window()
        except Exception:
            win = WindowInfo(title="", pid=None, app=None)
        return {
            "active_window": win.title,
            "active_pid": win.pid,
            "active_app": win.app,
            "ts": time.time(),
        }

    def _log(self, path: Path, entry: dict) -> None:
        with path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


class TaskSession:
    """High-level "give me a task" wrapper around :class:`VisionRunner`.

    Handles Ctrl-C gracefully, streams progress, and keeps the models
    loaded between calls (so the second task is fast).
    """

    def __init__(
        self,
        planner: Optional[Planner] = None,
        grounder: Optional[Grounder] = None,
        driver: Optional[PopOSDriver] = None,
        collector: Optional[OSContextCollector] = None,
        on_progress: Optional[ProgressFn] = None,
    ) -> None:
        self.runner = VisionRunner(
            planner=planner, grounder=grounder, driver=driver,
            collector=collector, on_progress=on_progress,
        )
        self._installed_sigint = False

    def __enter__(self):
        # Swap cosy-llm-v1 off the dGPU so the VLM+VLA can both fit.
        # SIGSTOP alone does NOT free CUDA memory (the CUDA context
        # is still held), so we kill the cosy processes and respawn
        # after. Honour the COZY_VISION_SWAP_COZY env var (set by run.sh).
        self._killed_cozy: list[int] = []
        if _HAVE_COZY_SWAP and os.environ.get("COZY_VISION_SWAP_COZY", "1") == "1":
            self._killed_cozy = swap_out()
            if self._killed_cozy:
                print(f"[task] killed cozy-llm-v1 (pids={self._killed_cozy}) to free {len(self._killed_cozy) * 1700} MB of GPU VRAM")
        try:
            self.runner.warmup()
        except Exception:
            if self._killed_cozy and os.environ.get("COZY_VISION_RESPAWN_COZY", "1") == "1":
                swap_in()
            raise
        self._install_sigint()
        return self

    def __exit__(self, *args):
        try:
            self.runner.shutdown()
        finally:
            # Always respawn cosy so the voice pipeline keeps working
            if self._killed_cozy and _HAVE_COZY_SWAP and os.environ.get("COZY_VISION_RESPAWN_COZY", "1") == "1":
                new_pid = swap_in()
                print(f"[task] respawned cozy-llm-v1 (new pid={new_pid})")

    def _install_sigint(self) -> None:
        def handler(signum, frame):
            print("\n[ctrl-c] stopping after current step...")
            self.runner.interrupt()
        if not self._installed_sigint:
            signal.signal(signal.SIGINT, handler)
            self._installed_sigint = True

    def ask(self, question: str) -> Answer:
        """Visual Q&A: answer a question about the current screen."""
        context = self.collector.snapshot(screen_size=self.driver.driver.screen_size)
        shot = self.runner.driver.screenshot()
        return self.runner.planner.answer(question, shot, context)

    def task(self, goal: str) -> VisionResult:
        """Run a single desktop task and stream progress."""
        return self.runner.run(goal)
