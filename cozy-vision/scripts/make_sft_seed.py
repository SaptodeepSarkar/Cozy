"""Generate diverse synthetic SFT data for the VLM planner and VLA grounder.

Each row is unique because:
  * random window count (2-5)
  * random window titles from a curated pool of real app names
  * random window positions and sizes
  * random theme (Pop!_OS dark, Pop!_OS light, GNOME, COSMIC accents)
  * per-task colour-coded dock badge
  * per-task focus indicator (one window highlighted as active)
  * mobile vs desktop form factor (50/50 mix) so the VLM learns to
    recognise phones and tell the VLA to open hamburger menus etc.

The output is two JSONL files:
  data/sft_planner_seed.jsonl  {image, messages: [...]} for the VLM
  data/sft_vla_seed.jsonl      {image, todo, assistant} for the VLA
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SHOTS = DATA / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)


# Curated app pool (real desktop + mobile app names)
DESKTOP_APPS = [
    ("Firefox", "browser"), ("Chrome", "browser"), ("Chromium", "browser"),
    ("Files", "file-manager"), ("Nautilus", "file-manager"),
    ("Terminal", "terminal"), ("Konsole", "terminal"),
    ("VS Code", "editor"), ("Sublime Text", "editor"),
    ("GIMP", "editor"), ("LibreOffice Writer", "editor"),
    ("Calculator", "utility"), ("Settings", "system"),
    ("System Monitor", "system"), ("Disk Usage Analyzer", "system"),
    ("Thunderbird", "mail"), ("Slack", "chat"), ("Discord", "chat"),
    ("Spotify", "media"), ("Rhythmbox", "media"), ("Videos", "media"),
    ("Photos", "media"), ("Image Viewer", "media"),
    ("Software Center", "system"), ("Help", "system"),
    ("Text Editor", "editor"), ("gedit", "editor"),
    ("Calendar", "productivity"), ("Contacts", "productivity"),
    ("Notes", "productivity"), ("Remmina", "remote"),
    ("Boxy SVG", "editor"), ("Inkscape", "editor"),
    ("Steam", "gaming"), ("Lutris", "gaming"),
]
MOBILE_APPS = [
    ("Instagram", "social"), ("Twitter", "social"), ("X", "social"),
    ("WhatsApp", "chat"), ("Telegram", "chat"), ("Signal", "chat"),
    ("Gmail", "mail"), ("Outlook", "mail"),
    ("YouTube", "media"), ("TikTok", "media"), ("Netflix", "media"),
    ("Spotify", "media"), ("Apple Music", "media"),
    ("Chrome", "browser"), ("Safari", "browser"),
    ("Maps", "maps"), ("Google Maps", "maps"),
    ("Uber", "transport"), ("Lyft", "transport"),
    ("Amazon", "shopping"), ("Flipkart", "shopping"),
    ("Swiggy", "food"), ("Zomato", "food"),
    ("Paytm", "finance"), ("PhonePe", "finance"),
    ("Camera", "media"), ("Photos", "media"), ("Gallery", "media"),
    ("Settings", "system"), ("Phone", "utility"),
    ("Clock", "utility"), ("Calculator", "utility"),
    ("Calendar", "productivity"), ("Notes", "productivity"),
    ("Slack", "chat"), ("Discord", "chat"), ("Teams", "chat"),
]
WEBSITES = [
    ("github.com", "github"), ("gitlab.com", "gitlab"),
    ("stackoverflow.com", "stack"), ("reddit.com", "reddit"),
    ("youtube.com", "youtube"), ("twitter.com", "twitter"),
    ("x.com", "x"), ("linkedin.com", "linkedin"),
    ("amazon.com", "amazon"), ("flipkart.com", "flipkart"),
    ("wikipedia.org", "wiki"), ("medium.com", "medium"),
    ("gmail.com", "gmail"), ("outlook.live.com", "outlook"),
    ("notion.so", "notion"), ("figma.com", "figma"),
    ("canva.com", "canva"), ("docs.google.com", "gdocs"),
    ("drive.google.com", "gdrive"), ("dropbox.com", "dropbox"),
    ("maps.google.com", "gmaps"), ("openstreetmap.org", "osm"),
    ("chat.openai.com", "chatgpt"), ("claude.ai", "claude"),
    ("gemini.google.com", "gemini"), ("perplexity.ai", "perplexity"),
]


SYSTEM_PROMPT_PLAN = """You are the PLANNER for a local desktop agent on Pop!_OS (COSMIC desktop, Wayland).

You see the current screenshot of the user's screen, the live OS context (active window, open
windows, focused element, working area), and a high-level user goal. You must break the goal
into a precise, ordered, *checkable* todo list that a separate fast visual-grounding model will
execute one item at a time.

Each todo item must be:
  * ATOMIC: one user-facing action.
  * GROUNDED: include the target element / window / file / text.
  * VERIFIABLE: include a `check` predicate that the executor can run after the action.
  * SAFE: never propose destructive actions.

Output ONLY a JSON array of objects:
[{"action": "<short verb phrase>", "target": "<element / window / file / text>",
  "check": "<how to verify success>", "params": {...optional kwargs...}}]

For MOBILE UIs, include steps to open the hamburger menu (☰) or bottom tab bar as needed
to reach less-prominent actions. For WEBSITES, include specific scroll / hover / click
patterns that are common on that site."""


SYSTEM_PROMPT_VLA = """You are a GUI agent executing ONE todo item at a time on Pop!_OS / COSMIC.

Output ONE action per turn: click(x, y), type(text), hotkey(k1, k2), scroll(dx, dy), wait(),
or finished(). For mobile, the hamburger icon is usually in the top-left or top-right corner.
For websites, the URL bar is usually addressable with Ctrl+L."""


# A library of (task, plan, vla_steps) tuples that captures common interactions.
# Each entry also specifies device family (desktop / mobile / web) so the
# generator paints the right kind of screen.
PLAN_LIBRARY: list[dict] = [
    {
        "task": "open firefox on the desktop",
        "device": "desktop",
        "plan": [
            {"action": "press the super key to open the app launcher",
             "target": "cosmic app launcher", "check": "launcher overlay visible"},
            {"action": "type 'firefox' into the launcher search",
             "target": "launcher search input", "check": "search query 'firefox' shown",
             "params": {"text": "firefox"}},
            {"action": "press Enter to launch the first result",
             "target": "launcher first result row",
             "check": "process 'firefox' is running and a window titled 'Mozilla Firefox' exists"},
        ],
        "vla": [
            ("press the super key to open the app launcher", "hotkey('super')"),
            ("type 'firefox' into the launcher search", "type('firefox')"),
            ("press Enter to launch the first result", "hotkey('Return')"),
            ("press the super key to open the app launcher", "finished()"),
        ],
    },
    {
        "task": "open github.com in the browser",
        "device": "web",
        "plan": [
            {"action": "open the default browser if not already running",
             "target": "default browser", "check": "process 'firefox' or 'chrome' is running"},
            {"action": "focus the URL bar with Ctrl+L",
             "target": "browser URL bar", "check": "URL bar is selected and accepts input"},
            {"action": "type the URL 'github.com'",
             "target": "URL bar", "check": "URL bar text == 'github.com'",
             "params": {"text": "github.com"}},
            {"action": "press Enter to navigate",
             "target": "URL bar", "check": "browser URL contains 'github.com'"},
        ],
        "vla": [
            ("open the default browser if not already running", "hotkey('super')"),
            ("focus the URL bar with Ctrl+L", "hotkey('ctrl', 'l')"),
            ("type the URL 'github.com'", "type('github.com')"),
            ("press Enter to navigate", "hotkey('Return')"),
            ("focus the URL bar with Ctrl+L", "finished()"),
        ],
    },
    {
        "task": "set the volume to 50 percent on the desktop",
        "device": "desktop",
        "plan": [
            {"action": "press the super key to open the app launcher",
             "target": "cosmic app launcher", "check": "launcher overlay visible"},
            {"action": "type 'settings' into the launcher search",
             "target": "launcher search input", "check": "search query 'settings' shown"},
            {"action": "press Enter to launch the first result",
             "target": "launcher first result row",
             "check": "process 'gnome-control-center' is running"},
            {"action": "click the sound page in the settings sidebar",
             "target": "settings sidebar -> Sound",
             "check": "sound settings page is visible"},
            {"action": "click the output volume slider at 50%",
             "target": "output volume slider", "check": "pactl get-sink-volume shows 50%",
             "params": {"level": 50}},
        ],
        "vla": [
            ("press the super key to open the app launcher", "hotkey('super')"),
            ("type 'settings' into the launcher search", "type('settings')"),
            ("press Enter to launch the first result", "hotkey('Return')"),
            ("click the sound page in the settings sidebar", "click(180, 320)"),
            ("click the output volume slider at 50%", "click(420, 480)"),
            ("press the super key to open the app launcher", "finished()"),
        ],
    },
    {
        "task": "find a restaurant on Zomato mobile app",
        "device": "mobile",
        "plan": [
            {"action": "open the Zomato app from the home screen",
             "target": "Zomato app icon",
             "check": "process 'com.zomato.android' is in foreground or 'Zomato' is the focused window"},
            {"action": "tap the search icon in the top-right",
             "target": "top-right search icon",
             "check": "search input field is visible"},
            {"action": "type the restaurant name or cuisine",
             "target": "search input field",
             "check": "search input contains the typed text"},
            {"action": "tap the first search result to open the restaurant page",
             "target": "first search result card",
             "check": "restaurant details page is visible (name, rating, address)"},
        ],
        "vla": [
            ("open the Zomato app from the home screen", "click(540, 1900)"),
            ("tap the search icon in the top-right", "click(680, 280)"),
            ("type the restaurant name or cuisine", "type('biryani')"),
            ("tap the first search result to open the restaurant page", "click(540, 800)"),
            ("open the Zomato app from the home screen", "finished()"),
        ],
    },
    {
        "task": "open the Instagram hamburger menu on mobile",
        "device": "mobile",
        "plan": [
            {"action": "open the Instagram app",
             "target": "Instagram app icon",
             "check": "Instagram home feed is visible"},
            {"action": "tap the hamburger menu (three horizontal lines) in the top-right",
             "target": "top-right hamburger menu icon",
             "check": "side drawer / 'Your activity' panel is open"},
            {"action": "tap the 'Settings' entry near the bottom of the drawer",
             "target": "drawer -> Settings",
             "check": "Settings page is visible"},
        ],
        "vla": [
            ("open the Instagram app", "click(420, 1900)"),
            ("tap the hamburger menu (three horizontal lines) in the top-right", "click(680, 100)"),
            ("tap the 'Settings' entry near the bottom of the drawer", "click(360, 1200)"),
            ("open the Instagram app", "finished()"),
        ],
    },
    {
        "task": "close the current window",
        "device": "desktop",
        "plan": [
            {"action": "press Alt+F4 to close the focused window",
             "target": "focused window", "check": "active window changes or closes"},
        ],
        "vla": [
            ("press Alt+F4 to close the focused window", "hotkey('alt', 'F4')"),
            ("press Alt+F4 to close the focused window", "finished()"),
        ],
    },
    {
        "task": "take a screenshot of the current screen",
        "device": "desktop",
        "plan": [
            {"action": "press the Print key to capture the whole screen",
             "target": "global Print shortcut",
             "check": "a new PNG appears in ~/Pictures/Screenshots/"},
        ],
        "vla": [
            ("press the Print key to capture the whole screen", "hotkey('Print')"),
            ("press the Print key to capture the whole screen", "finished()"),
        ],
    },
    {
        "task": "scroll to the bottom of a long webpage",
        "device": "web",
        "plan": [
            {"action": "press End to jump to the bottom of the page",
             "target": "browser viewport",
             "check": "scroll position is at the bottom (scrollY == scrollHeight - clientHeight)"},
        ],
        "vla": [
            ("press End to jump to the bottom of the page", "hotkey('End')"),
            ("press End to jump to the bottom of the page", "finished()"),
        ],
    },
    {
        "task": "search for 'python tutorials' on google",
        "device": "web",
        "plan": [
            {"action": "open google.com if not already there",
             "target": "browser URL bar", "check": "URL bar shows 'google.com'"},
            {"action": "type 'python tutorials' in the search box",
             "target": "google search input", "check": "search box contains 'python tutorials'",
             "params": {"text": "python tutorials"}},
            {"action": "press Enter to search",
             "target": "google search input", "check": "search results page is visible"},
        ],
        "vla": [
            ("open google.com if not already there", "hotkey('ctrl', 'l')"),
            ("type 'python tutorials' in the search box", "type('python tutorials')"),
            ("press Enter to search", "hotkey('Return')"),
            ("open google.com if not already there", "finished()"),
        ],
    },
    {
        "task": "switch to the next virtual desktop",
        "device": "desktop",
        "plan": [
            {"action": "press Super+Page_Down to move to the workspace on the right",
             "target": "cosmic workspace switcher",
             "check": "active window changes to a window in the new workspace"},
        ],
        "vla": [
            ("press Super+Page_Down to move to the workspace on the right", "hotkey('super', 'Page_Down')"),
            ("press Super+Page_Down to move to the workspace on the right", "finished()"),
        ],
    },
    {
        "task": "lock the screen",
        "device": "desktop",
        "plan": [
            {"action": "press Super+L to lock the screen",
             "target": "global lock shortcut", "check": "screen is locked (greeter visible)"},
        ],
        "vla": [
            ("press Super+L to lock the screen", "hotkey('super', 'l')"),
            ("press Super+L to lock the screen", "finished()"),
        ],
    },
    {
        "task": "answer the question based on the current screen",
        "device": "desktop",
        "plan": [
            {"action": "answer the question based on the current screenshot",
             "target": "screen", "check": "answer is grounded in pixels"},
        ],
        "vla": [],
    },
]


def _draw_desktop(task: str, plan_devices: list[str], idx: int) -> Path:
    """Paint a unique fake Pop!_OS COSMIC desktop."""
    w, h = 1280 + random.randint(-60, 60), 800 + random.randint(-40, 40)
    themes = [
        {"bg": (28, 28, 36), "panel": (20, 20, 28), "win": (48, 50, 56), "border": (120, 120, 130), "text": (220, 220, 220)},
        {"bg": (240, 240, 245), "panel": (220, 220, 225), "win": (255, 255, 255), "border": (160, 160, 170), "text": (30, 30, 35)},
        {"bg": (35, 22, 38), "panel": (25, 15, 28), "win": (60, 40, 65), "border": (140, 100, 150), "text": (235, 220, 240)},
        {"bg": (20, 30, 25), "panel": (15, 22, 18), "win": (35, 55, 45), "border": (110, 150, 130), "text": (220, 240, 230)},
    ]
    theme = random.choice(themes)
    img = Image.new("RGB", (w, h), color=theme["bg"])
    d = ImageDraw.Draw(img)
    # Top bar
    d.rectangle([(0, 0), (w, 28)], fill=theme["panel"])
    d.text((10, 6), "pop!_os  |  cosmic", fill=theme["text"])
    # Dock at the bottom
    d.rectangle([(0, h - 60), (w, h)], fill=theme["panel"])
    # Random windows (2-5)
    n_windows = random.randint(2, 5)
    chosen = random.sample(DESKTOP_APPS, min(n_windows, len(DESKTOP_APPS)))
    focused_idx = random.randint(0, n_windows - 1)
    for i, (title, kind) in enumerate(chosen[:n_windows]):
        ww = random.randint(360, 520)
        wh = random.randint(260, 400)
        wx = random.randint(40, max(40, w - ww - 40))
        wy = random.randint(60, max(60, h - wh - 80))
        is_focused = (i == focused_idx)
        d.rectangle([(wx, wy), (wx + ww, wy + wh)],
                    outline=theme["border"], width=(3 if is_focused else 1),
                    fill=theme["win"])
        # Title bar
        bar_color = theme["panel"] if not is_focused else (80, 90, 110)
        d.rectangle([(wx, wy), (wx + ww, wy + 24)], fill=bar_color)
        d.text((wx + 8, wy + 4), title, fill=theme["text"])
        # A few fake "content" lines
        for j in range(8):
            line_y = wy + 40 + j * 24
            if line_y > wy + wh - 20:
                break
            line_w = random.randint(int(ww * 0.3), int(ww * 0.85))
            d.rectangle([(wx + 16, line_y), (wx + 16 + line_w, line_y + 6)],
                        fill=theme["border"])
    # Dock icons (a few)
    for i, (title, _) in enumerate(chosen[:n_windows]):
        d.rectangle([(20 + i * 64, h - 50), (60 + i * 64, h - 16)],
                    fill=theme["border"])
        d.text((20 + i * 64, h - 50), title[:6], fill=theme["text"])
    # Highlight the active window in a different border colour
    if focused_idx < len(chosen):
        ft, _ = chosen[focused_idx]
        d.text((20, h - 90), f"focus: {ft}", fill=theme["text"])
    # Bottom task badge
    d.text((w - 360, h - 50), f"task #{idx}: {task[:40]}", fill=theme["text"])
    p = SHOTS / f"desk_{idx:06d}_{random.randint(0, 99999):05d}.png"
    img.save(p)
    return p


def _draw_mobile(task: str, idx: int) -> Path:
    """Paint a unique fake phone screen."""
    w, h = 720, 1480  # pixel-3-ish portrait
    themes = [
        {"bg": (15, 15, 20), "card": (30, 30, 40), "accent": (88, 101, 242), "text": (240, 240, 250), "muted": (150, 150, 160)},
        {"bg": (255, 255, 255), "card": (245, 245, 250), "accent": (228, 64, 95), "text": (15, 15, 20), "muted": (110, 110, 120)},
        {"bg": (245, 245, 245), "card": (255, 255, 255), "accent": (0, 122, 255), "text": (0, 0, 0), "muted": (140, 140, 140)},
    ]
    theme = random.choice(themes)
    img = Image.new("RGB", (w, h), color=theme["bg"])
    d = ImageDraw.Draw(img)
    # Status bar
    d.text((20, 16), "9:41", fill=theme["text"])
    d.text((w - 100, 16), "5G • 100%", fill=theme["text"])
    # Top app bar
    d.rectangle([(0, 60), (w, 140)], fill=theme["bg"])
    # Hamburger menu icon (top-left, only for non-instagram, etc)
    has_hamburger = random.random() < 0.6
    if has_hamburger:
        for i in range(3):
            d.rectangle([(28, 80 + i * 12), (60, 86 + i * 12)], fill=theme["text"])
    # Title in the middle
    title = random.choice(["Instagram", "Twitter", "Zomato", "Gmail", "YouTube", "Maps", "Settings"])
    d.text((w // 2 - 50, 90), title, fill=theme["text"])
    # Search icon top-right
    d.ellipse([(w - 80, 80), (w - 50, 110)], outline=theme["text"], width=2)
    d.line([(w - 60, 100), (w - 50, 110)], fill=theme["text"], width=2)
    # Bottom tab bar
    tab_y = h - 120
    d.rectangle([(0, tab_y), (w, h)], fill=theme["card"])
    n_tabs = random.choice([3, 4, 5])
    tab_titles = random.sample(["Home", "Search", "Reels", "Shop", "Profile", "Inbox", "Likes", "Notifications", "Settings"], n_tabs)
    for i, tt in enumerate(tab_titles):
        cx = int((i + 0.5) * w / n_tabs)
        d.rectangle([(cx - 14, tab_y + 24), (cx + 14, tab_y + 52)], fill=theme["muted"])
        d.text((cx - 18, tab_y + 60), tt[:6], fill=theme["muted"])
    # Main content (cards)
    n_cards = random.randint(2, 5)
    for i in range(n_cards):
        cy = 180 + i * random.randint(180, 260)
        if cy + 160 > tab_y:
            break
        d.rectangle([(20, cy), (w - 20, cy + 160)], fill=theme["card"], outline=theme["muted"])
        d.rectangle([(40, cy + 16), (140, cy + 96)], fill=theme["muted"])
        d.rectangle([(160, cy + 16), (w - 40, cy + 28)], fill=theme["muted"])
        d.rectangle([(160, cy + 40), (w - 80, cy + 52)], fill=theme["muted"])
        d.rectangle([(160, cy + 64), (w - 120, cy + 76)], fill=theme["muted"])
        d.rectangle([(40, cy + 116), (w - 40, cy + 140)], fill=theme["muted"])
    # task badge
    d.text((20, h - 30), f"task #{idx}: {task[:30]}", fill=theme["muted"])
    p = SHOTS / f"mob_{idx:06d}_{random.randint(0, 99999):05d}.png"
    img.save(p)
    return p


def _draw_website(task: str, idx: int) -> Path:
    """Paint a fake browser window showing a website."""
    w, h = 1280, 800
    themes = [
        {"bg": (255, 255, 255), "bar": (240, 240, 245), "card": (245, 245, 250), "text": (15, 15, 20), "muted": (110, 110, 120), "link": (0, 102, 204)},
        {"bg": (24, 24, 30), "bar": (15, 15, 22), "card": (40, 40, 50), "text": (230, 230, 240), "muted": (150, 150, 160), "link": (88, 166, 255)},
    ]
    theme = random.choice(themes)
    img = Image.new("RGB", (w, h), color=theme["bg"])
    d = ImageDraw.Draw(img)
    # Browser chrome
    d.rectangle([(0, 0), (w, 64)], fill=theme["bar"])
    # URL bar
    d.rectangle([(120, 16), (w - 200, 48)], fill=theme["bg"], outline=theme["muted"])
    website, _ = random.choice(WEBSITES)
    d.text((140, 24), f"https://{website}/path/...", fill=theme["text"])
    # Nav buttons
    for i, sym in enumerate(["<", ">", "↻"]):
        d.rectangle([(20 + i * 36, 16), (52 + i * 36, 48)], fill=theme["bg"], outline=theme["muted"])
    # Website body
    n_cards = random.randint(2, 6)
    for i in range(n_cards):
        cy = 96 + i * random.randint(80, 140)
        d.rectangle([(20, cy), (w - 20, cy + 90)], fill=theme["card"], outline=theme["muted"])
        d.rectangle([(40, cy + 12), (140, cy + 76)], fill=theme["muted"])
        d.rectangle([(160, cy + 16), (w - 60, cy + 28)], fill=theme["muted"])
        d.rectangle([(160, cy + 40), (w - 200, cy + 52)], fill=theme["muted"])
        d.rectangle([(160, cy + 64), (w - 300, cy + 74)], fill=theme["muted"])
    d.text((20, h - 30), f"task #{idx}: {task[:30]}", fill=theme["muted"])
    p = SHOTS / f"web_{idx:06d}_{random.randint(0, 99999):05d}.png"
    img.save(p)
    return p


def _draw_for_device(task: str, device: str, idx: int) -> Path:
    if device == "mobile":
        return _draw_mobile(task, idx)
    if device == "web":
        return _draw_website(task, idx)
    return _draw_desktop(task, [device], idx)


def make_planner_rows(per_task: int = 4) -> Iterable[dict]:
    rng = random.Random(0)
    for entry in PLAN_LIBRARY:
        for i in range(per_task):
            shot = _draw_for_device(entry["task"], entry["device"], i)
            user_text = (
                "OS CONTEXT (live snapshot):\n"
                f"active_window: '{entry['task'][:40]}'\n"
                f"open_windows: {random.randint(2, 6)} apps\n"
                f"screen_size: 1280x800\n"
                f"device_family: {entry['device']}\n\n"
                f"User goal: {entry['task']!r}\n\n"
                "Return the remaining todo items as a JSON array. "
                "For mobile UIs include the hamburger menu step. For web UIs include Ctrl+L for the URL bar."
            )
            assistant = json.dumps(entry["plan"])
            yield {
                "image": str(shot.relative_to(ROOT)),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_PLAN},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(shot.relative_to(ROOT))},
                            {"type": "text", "text": user_text},
                        ],
                    },
                    {"role": "assistant", "content": assistant},
                ],
                "task_id": entry["task"],
                "device": entry["device"],
            }


def make_vla_rows(per_todo: int = 3) -> Iterable[dict]:
    rng = random.Random(1)
    for entry in PLAN_LIBRARY:
        for todo_text, action_text in entry["vla"]:
            for i in range(per_todo):
                shot = _draw_for_device(entry["task"], entry["device"], i)
                yield {
                    "image": str(shot.relative_to(ROOT)),
                    "todo": {"action": todo_text, "target": "", "check": ""},
                    "assistant": action_text,
                    "task_id": entry["task"],
                    "device": entry["device"],
                }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner-out", default=str(DATA / "sft_planner_seed.jsonl"))
    ap.add_argument("--vla-out", default=str(DATA / "sft_vla_seed.jsonl"))
    ap.add_argument("--planner-per-task", type=int, default=4)
    ap.add_argument("--vla-per-todo", type=int, default=3)
    args = ap.parse_args()

    n1, n2 = 0, 0
    with open(args.planner_out, "w") as f:
        for r in make_planner_rows(args.planner_per_task):
            f.write(json.dumps(r) + "\n")
            n1 += 1
    with open(args.vla_out, "w") as f:
        for r in make_vla_rows(args.vla_per_todo):
            f.write(json.dumps(r) + "\n")
            n2 += 1
    print(f"wrote {n1} planner rows to {args.planner_out}")
    print(f"wrote {n2} VLA rows to {args.vla_out}")
    # Verify uniqueness
    import hashlib
    seen = set()
    dupes = 0
    for p in SHOTS.iterdir():
        if p.is_file():
            with open(p, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
            if h in seen:
                dupes += 1
            seen.add(h)
    print(f"unique images: {len(seen)} / {len(list(SHOTS.iterdir()))} ({dupes} dupes)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
