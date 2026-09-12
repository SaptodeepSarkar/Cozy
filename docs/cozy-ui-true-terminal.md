# Cozy UI — True Terminal v2 (locked)

Desktop Linux terminal only. No mobile app. No orange/amber wash, no paper
bubbles, no glossy cards, no glow shadows.

Inspiration: ghostty + claude-code + htop. Quiet, precise, hacker-calm.
The cat ASCII (`|\__/,|`) is the only playfulness.

## Tokens

- bg ink `#0b0d10`, surface `#12151b`, hairline `#23272f` 1px
- text phosphor `#d0d6e0`, muted `#5c6370`, faint `#3b3b3b`
- ready/success sage `#c3e88d` (dots + READY only)
- listening/active sky `#7aa2f7` (waveform + LISTENING only)
- thinking dim peach `#e0c07c` (spinner only)
- error muted `#ff6b6b` (✗ only)
- type: JetBrains Mono everywhere; Space Grotesk only for tiny
  `Cozy ui v0.1` wordmark
- radius 4px max, mostly square; elevation = brightness, not cards

## Layout (1280px desktop, 760px centered mono column)

Top: cat ASCII 3 lines left, `Cozy ui v0.1` right muted.
Second line: text-only pills
`[wake●●●][stt○○○][llm○○○][cleanup○○○][tts○○○]`
● done green, ◐ loading blue pulse, ○ pending gray, ✗ failed red.

Middle: text river only, `─` separators in `#23272f`.
- user: `> open firefox...` bright white
- cozy: plain bright paragraph, no bubble
- tools: dim `→ app.open(name=firefox) done 320ms` with ✓ green / ⟳ blue
- listening: `LISTENING 00:03` blue + 24-bar 2px flat waveform `#7aa2f7`
- thinking: `⠋ thinking qwen3-0.6b 420ms` peach spinner
- loading: `━━━━━━────── 60%` + `warming whisper-small CT2 int8...` dim

Bottom: single `> _` block-cursor line +
`ctrl+c quit • space talk` muted 10px. No FAB mic, no composer card.

## Stitch source of truth

Project `Cozy Voice Assistant - TUI Mockups`
(`projects/9382159678479596545`), design system
`Cozy True Terminal v2` (`assets/7718028822528960179`).

Refined screens (desktop only, mobile dropped):
- loading `d1b69ca1674b4d4a99b618d6cf5912f3`
- idle ready `b875b95ef05944208660441323af13a0`
- listening `15077ab3f51b47d4ba98c1f2ad4acb85`
- thinking `ec35d28564c145b8b753aaa69d7451c0`
- done `813b1f7943534d49a9c7508f848927da`

Superseded orange explorations (do not follow):
`40c7e722aef745e8bf5a4637bed66cc0`, `08210e6d223245ab9bd3e3f319950263`,
`6673e0c14a5f40bb9028e31e4e17a6f6`, `dd56e5aa2a074299a7a9f8eeb7e73c4e`,
`289c5847247f4eb3b01aca2168d74306`, mobile `2dbef3dc157141f5892b432d2f1dcd00`.

## Code status

`assistant/tui_textual.py` + `assistant/tui-node` already match this
direction (ink terminal, mono pills, `> _` input). No code change needed
for this shift; this doc locks the direction so future TUI work stays
terminal-native.
