# Team (multi-agent notes)

This folder holds the work of the multi-agent team that built the various
Cozy subsystems. Each agent had a role, and the channel.jsonl records the
back-and-forth between them.

## Files

| File | Purpose |
|---|---|
| `STATUS.md` | Status board — what each agent is working on |
| `channel.jsonl` | Agent-to-agent communication log |
| `tool_schema.json` | LLM tool-call schema (15 tools) |
| `wake_final_report.txt` | Old wakeword training report (pre-livekit) |
| `scripts/` | Legacy training scripts (from earlier agents) |
| `data/` | Shared data between agents |

## Subsystem ownership

| Subsystem | Agent | Output |
|---|---|---|
| Wake word | wakeword-agent | `../wakeword/output/hey_cozy/` |
| STT | stt-agent | `../stt-finetune/output/cozy_stt_v1_ct2_int8/` |
| LLM | llm-agent | `../assistant/model/cozy-llm-v1/` + adapter |

See `STATUS.md` for the latest status of each subsystem.
