"""Cozy-Vision harness.

Local desktop agent that takes a plain-English goal, plans sub-tasks with
a VLM, and drives the mouse + keyboard with a VLA.
"""
from .runner import VisionRunner, VisionResult
from .planner import Planner
from .grounder import Grounder
from .driver import PopOSDriver
from .reward import reward
from . import tasks

__all__ = [
    "VisionRunner",
    "VisionResult",
    "Planner",
    "Grounder",
    "PopOSDriver",
    "reward",
    "tasks",
]
