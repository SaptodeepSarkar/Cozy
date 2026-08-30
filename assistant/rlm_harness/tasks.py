"""Task definitions for the Cozy RLM harness.

A task is one user utterance (or a multi-turn goal) that the assistant is
expected to handle. Tasks can come from three places:

1. The bundled ``data/tasks_seed.jsonl`` (curated starter set).
2. ``assistant/data/sft_train.jsonl`` - reuse existing SFT prompts.
3. ``team/data/stt_command_seeds.jsonl`` - real STT transcripts.

A task is just a dict::

    {"id": "v1", "text": "set volume to 30", "category": "volume", ...}

The harness treats ``text`` as the first user turn. The "oracle" (human
or another AI) then decides the correct assistant response (text reply
or tool call) and the trace is saved.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

HERE = Path(__file__).resolve().parent
ASSISTANT = HERE.parent  # assistant/
TEAM = ASSISTANT.parent / "team"

SEED_TASKS_PATH = HERE / "data" / "tasks_seed.jsonl"


@dataclass
class Task:
    id: str
    text: str
    category: str = "general"
    # optional expected tool (used for hints in dataset mode; not enforced)
    hint: dict | None = None
    # difficulty tag (used to mix easy/medium/hard in a session)
    difficulty: str = "easy"
    # extra metadata
    meta: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.category}/{self.difficulty}] {self.text}"


# ---------------------------------------------------------- seed corpus
SEED_TASKS: list[dict] = [
    # ----- volume -----
    {"category": "volume", "difficulty": "easy", "text": "set volume to 50"},
    {"category": "volume", "difficulty": "easy", "text": "volume 80 percent"},
    {"category": "volume", "difficulty": "easy", "text": "lower the volume to 20"},
    {"category": "volume", "difficulty": "medium", "text": "make it louder"},
    {"category": "volume", "difficulty": "medium", "text": "too quiet, turn it up"},
    {"category": "volume", "difficulty": "medium", "text": "mute the sound"},
    {"category": "volume", "difficulty": "hard",
     "text": "awaz 70 percent kar do"},
    {"category": "volume", "difficulty": "hard",
     "text": "volume thoda kam karo bhai"},

    # ----- brightness -----
    {"category": "brightness", "difficulty": "easy",
     "text": "set brightness to 40"},
    {"category": "brightness", "difficulty": "easy",
     "text": "brightness 100"},
    {"category": "brightness", "difficulty": "medium",
     "text": "screen is too bright"},
    {"category": "brightness", "difficulty": "medium",
     "text": "dim the display a bit"},
    {"category": "brightness", "difficulty": "hard",
     "text": "roshni 30 kar do"},

    # ----- apps -----
    {"category": "app", "difficulty": "easy", "text": "open chrome"},
    {"category": "app", "difficulty": "easy", "text": "launch terminal"},
    {"category": "app", "difficulty": "easy", "text": "start spotify"},
    {"category": "app", "difficulty": "medium", "text": "open the calculator"},
    {"category": "app", "difficulty": "medium", "text": "I need the file manager"},
    {"category": "app", "difficulty": "hard", "text": "vs code kholo"},
    {"category": "app", "difficulty": "hard", "text": "settings app chalu karo"},
    {"category": "app", "difficulty": "hard", "text": "close chrome please"},

    # ----- browser -----
    {"category": "browser", "difficulty": "easy",
     "text": "search for python tutorials"},
    {"category": "browser", "difficulty": "easy",
     "text": "google biryani recipe"},
    {"category": "browser", "difficulty": "medium",
     "text": "look up the weather in kolkata"},
    {"category": "browser", "difficulty": "medium",
     "text": "open github.com"},
    {"category": "browser", "difficulty": "hard",
     "text": "youtube kholo"},

    # ----- screenshot -----
    {"category": "screenshot", "difficulty": "easy",
     "text": "take a screenshot"},
    {"category": "screenshot", "difficulty": "easy",
     "text": "capture my screen"},
    {"category": "screenshot", "difficulty": "medium",
     "text": "screenshot lo please"},

    # ----- media -----
    {"category": "media", "difficulty": "easy", "text": "play music"},
    {"category": "media", "difficulty": "easy", "text": "pause the music"},
    {"category": "media", "difficulty": "easy", "text": "next track"},
    {"category": "media", "difficulty": "easy", "text": "previous song"},
    {"category": "media", "difficulty": "hard", "text": "gaana chalu karo"},
    {"category": "media", "difficulty": "hard", "text": "music roko"},

    # ----- time/date -----
    {"category": "time", "difficulty": "easy", "text": "what time is it"},
    {"category": "time", "difficulty": "easy",
     "text": "tell me the time"},
    {"category": "time", "difficulty": "medium",
     "text": "what's today's date"},
    {"category": "time", "difficulty": "hard", "text": "time batao"},

    # ----- chat (no tool) -----
    {"category": "chat", "difficulty": "easy", "text": "hi"},
    {"category": "chat", "difficulty": "easy", "text": "hello cozy"},
    {"category": "chat", "difficulty": "easy", "text": "who are you"},
    {"category": "chat", "difficulty": "easy",
     "text": "what can you do"},
    {"category": "chat", "difficulty": "easy", "text": "thank you"},
    {"category": "chat", "difficulty": "medium",
     "text": "tell me a joke"},
    {"category": "chat", "difficulty": "medium",
     "text": "how are you doing today"},
    {"category": "chat", "difficulty": "hard", "text": "kaise ho"},
    {"category": "chat", "difficulty": "hard",
     "text": "tum kaun ho"},

    # ----- settings -----
    {"category": "settings", "difficulty": "easy",
     "text": "open wifi settings"},
    {"category": "settings", "difficulty": "easy",
     "text": "show me bluetooth settings"},
    {"category": "settings", "difficulty": "medium",
     "text": "go to display settings"},

    # ----- window / desktop -----
    {"category": "window", "difficulty": "easy",
     "text": "minimize everything"},
    {"category": "window", "difficulty": "medium",
     "text": "show me the desktop"},
]


def _ensure_seed_file() -> None:
    """Write the seed JSONL if it doesn't exist yet."""
    if SEED_TASKS_PATH.exists():
        return
    SEED_TASKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    rng.shuffle(SEED_TASKS)
    with SEED_TASKS_PATH.open("w", encoding="utf-8") as f:
        for i, row in enumerate(SEED_TASKS):
            row = {"id": f"seed-{i:04d}", **row}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tasks(source: str = "seed",
               limit: int | None = None,
               shuffle: bool = True,
               categories: list[str] | None = None,
               difficulties: list[str] | None = None) -> list[Task]:
    """Load tasks.

    ``source`` is one of:
        "seed"           -> bundled curated set
        "sft"            -> assistant/data/sft_train.jsonl user turns
        "stt"            -> team/data/stt_command_seeds.jsonl
        "<path.jsonl>"   -> any jsonl file with a "text" field per row
    """
    _ensure_seed_file()
    if source == "seed":
        path = SEED_TASKS_PATH
        rows: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
    elif source == "sft":
        path = ASSISTANT / "data" / "sft_train.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            msgs = json.loads(line).get("messages", [])
            for m in msgs:
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    rows.append({"text": m["content"], "category": "sft",
                                 "difficulty": "medium"})
                    break
    elif source == "stt":
        path = TEAM / "data" / "stt_command_seeds.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({"text": r["text"], "category": "stt",
                         "difficulty": "medium",
                         "meta": {"tool": r.get("tool")}})
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(source)
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({"text": r.get("text", r.get("utterance", "")),
                         "category": r.get("category", "custom"),
                         "difficulty": r.get("difficulty", "medium"),
                         "hint": r.get("hint"),
                         "meta": r.get("meta", {})})

    if categories:
        rows = [r for r in rows if r.get("category") in categories]
    if difficulties:
        rows = [r for r in rows if r.get("difficulty") in difficulties]

    if shuffle:
        random.Random(0).shuffle(rows)

    if limit:
        rows = rows[:limit]

    return [
        Task(
            id=str(r.get("id", f"row-{i:04d}")),
            text=r["text"],
            category=r.get("category", "general"),
            difficulty=r.get("difficulty", "easy"),
            hint=r.get("hint"),
            meta={k: v for k, v in r.items()
                  if k not in {"id", "text", "category", "difficulty",
                               "hint"}},
        )
        for i, r in enumerate(rows)
        if r.get("text")
    ]


def iter_tasks(*args, **kwargs) -> Iterator[Task]:
    yield from load_tasks(*args, **kwargs)
