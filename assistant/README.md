# Cozy Assistant (roadmap)

This folder will host the *brain* that sits behind the wake word:

1. **Listener daemon** (`listener.py`)
   - Streams the microphone through wakeword/models/cozy_v1.onnx
   - On detection: chime + hands audio to the STT engine
2. **Speech-to-text** — faster-whisper (small/int8) for local transcription
3. **Intent router** — maps utterances to skills:
   - agent skills: "cozy, resume the build", "cozy, review my diff"
     (bridges into local agent sessions such as DSH)
   - pc skills: launch apps, media keys, screenshots, clipboard, power
   - q&a skills: local LLM answers
4. **Executor + safety** — confirm-before-act policy, kill phrase
   "goodnight cozy", activity log

Status: **planned** — the wake-word foundation in ../wakeword is step one.
