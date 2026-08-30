"""Skill discovery - markdown-only, copy of Prime Agent's skills.ts.

Skills are markdown files in ~/.cozy/skills/<name>/SKILL.md. The
system prompt emits a short <available_skills> XML block listing
each skill; the full SKILL.md loads on demand (or never, in our
voice-first UX).
"""
from __future__ import annotations

import re
from pathlib import Path

SKILLS_DIR = Path.home() / ".cozy" / "skills"
NAME_RE = re.compile(r"^[a-z0-9-]+$")


def discover_skills() -> list[dict]:
    """Return a list of {name, title, description} for each skill on disk."""
    if not SKILLS_DIR.exists():
        return []
    skills = []
    for child in sorted(SKILLS_DIR.iterdir()):
        if not child.is_dir():
            continue
        if not _valid_name(child.name):
            continue
        skill_md = child / "SKILL.md"
        if not skill_md.exists():
            continue
        # Parse frontmatter
        try:
            text = skill_md.read_text()
        except OSError:
            continue
        meta = _parse_frontmatter(text)
        if not meta.get("name"):
            meta["name"] = child.name
        skills.append(meta)
    return skills


def _valid_name(name: str) -> bool:
    if not name or len(name) > 64:
        return False
    if name.startswith("-") or name.endswith("-"):
        return False
    if "--" in name:
        return False
    return bool(NAME_RE.match(name))


def _parse_frontmatter(text: str) -> dict:
    """Extract YAML-ish frontmatter from a SKILL.md."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    fm = text[3:end].strip()
    out = {}
    for line in fm.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def format_skills_for_prompt(skills: list[dict], max_chars: int = 2000) -> str:
    """Emit the <available_skills> XML block for the system prompt."""
    if not skills:
        return ""
    lines = ["<available_skills>"]
    n = 0
    for s in skills:
        name = s.get("name", "?")
        desc = s.get("description", "(no description)")
        line = f'  <skill name="{name}">{desc[:120]}</skill>'
        n += len(line)
        if n > max_chars:
            lines.append("  <!-- truncated -->")
            break
        lines.append(line)
    lines.append("</available_skills>")
    return "\n".join(lines)


if __name__ == "__main__":
    for s in discover_skills():
        print(s.get("name"), "-", s.get("description", ""))
    print()
    print(format_skills_for_prompt(discover_skills()))
