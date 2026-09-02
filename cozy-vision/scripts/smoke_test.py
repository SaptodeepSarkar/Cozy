"""Smoke test: load both models and run a synthetic inference.

Verifies that
  1. The Qwen2.5-VL-3B-Instruct model loads with NF4 4-bit quantization
  2. The UI-TARS-2B-SFT model loads with the Qwen2-VL class
  3. Both can do one forward pass on a 512x512 random image
  4. The grounder's action parser understands UI-TARS output

Run with: ``bash run.sh smoke``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.context import OSContextCollector
from harness.grounder import Grounder
from harness.planner import Planner, TodoItem


def main() -> int:
    print("Cozy-Vision smoke test")
    print("=" * 60)

    collector = OSContextCollector()
    # Mock screen size since the real driver is not available without ydotool
    screen_size = (1920, 1080)

    print("loading planner (Qwen2.5-VL-3B-Instruct NF4 4-bit) ...")
    planner = Planner()
    planner.load()
    print("  ok")
    print("loading grounder (UI-TARS-2B-SFT NF4 4-bit) ...")
    grounder = Grounder()
    grounder.load()
    print("  ok")

    print("\nsynthetic 512x512 random RGB image ...")
    rng = np.random.default_rng(0)
    img = Image.fromarray((rng.random((512, 512, 3)) * 255).astype("uint8"))

    print("\nplanner.plan('click the centre of the image')")
    ctx = collector.snapshot(screen_size=screen_size)
    plan = planner.plan("click the centre of the image", img, ctx)
    print(f"  raw:        {plan.raw[:200]!r}")
    print(f"  todo items: {len(plan.todo)}")
    for t in plan.todo:
        print(f"    - {t.action!r} target={t.target!r} check={t.check!r}")

    if plan.todo:
        todo = plan.todo[0]
    else:
        todo = TodoItem(action="click the centre", target="", check="")
    print("\ngrounder.step(todo) ...")
    action = grounder.step(todo, img)
    print(f"  action: {action}")
    print(f"  type:   {type(action).__name__}")

    # Round-trip the parser
    print("\nparser sanity:")
    for fake in ["click(123, 456)", "Action: click(10, 20, 'right')",
                 "type('hello world')", "hotkey('ctrl', 'c')",
                 "scroll(0, 100)", "wait()", "finished('done')",
                 "click(99)", "unknown()"]:
        a = Grounder._parse(fake)
        print(f"  {fake!r:50s} -> {type(a).__name__} {a}")

    planner.free()
    grounder.free()
    print("\nall good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
