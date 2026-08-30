"""Cozy RLM harness.

Two modes for a fine-tuned tool-calling LLM:

* ``dataset``  - a human (or another AI) plays the oracle for every task.
                 Each task -> user turn -> (optional) tool result -> assistant
                 turn. The complete trace is dumped to JSONL in the same
                 schema as ``assistant/data/sft_train.jsonl`` so it drops
                 straight into the next SFT run.

* ``play``     - run the current ``cozy-llm-v1`` on every task and record
                 its decisions. Useful for review and for mining DPO
                 preferences.

Run from the repo root:

    python -m assistant.rlm_harness dataset --out assistant/data/sft_extra.jsonl
    python -m assistant.rlm_harness play    --out assistant/rlm_harness/traces/play.jsonl
"""
from .trace import Trace, Turn, Role
from .tasks import Task, load_tasks
from .harness import Harness, ModelBackend, RuleBackend
from . import dataset_mode, play_mode

__all__ = [
    "Trace",
    "Turn",
    "Role",
    "Task",
    "load_tasks",
    "Harness",
    "ModelBackend",
    "RuleBackend",
    "dataset_mode",
    "play_mode",
]
