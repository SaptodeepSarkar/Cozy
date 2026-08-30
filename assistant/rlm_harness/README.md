# Cozy RLM Harness

A small, fast harness for collecting **tool-call SFT data** for Cozy's
fine-tuned Qwen3-0.6B model. Two modes:

| Mode | What it does | When to use it |
|---|---|---|
| `dataset` | A human (or another AI) plays the oracle for every task. Every trace is dumped to JSONL in the same schema `make_dataset.py` produces. | When you want to grow the SFT set with real, validated examples. |
| `play`    | Run the current `cozy-llm-v1` on every task and log its predictions. | When you want to smoke-test the model or build DPO pairs. |
| `serve`   | REPL: type user messages, the model replies, every exchange is logged. | When you want to do both at once: chat, see what the model does, and (with manual edits) capture corrections. |

## Quick start

```bash
# from the repo root
python -m assistant.rlm_harness info                # task counts + tool list
python -m assistant.rlm_harness dataset --limit 10  # collect 10 traces by hand
python -m assistant.rlm_harness play    --limit 50  # log 50 model decisions
python -m assistant.rlm_harness merge     --source assistant/data/sft_extra.jsonl        # fold into training set
```

## Why this exists

The existing `make_dataset.py` synthesises training data from templates
and the STT seed file. That's great for bootstrapping but it caps out:
the model only learns phrasing variants of patterns we already wrote.
This harness closes the loop: whenever you (or a stronger model) want
to show the small model how to handle a new phrase, a new edge case, or
a new tool, you record one trace and it becomes training data.

## The dataset loop in detail

1. `dataset` reads tasks from `--source` (default: the curated
   `data/tasks_seed.jsonl`).
2. For each task it shows the user utterance. Optionally it shows the
   *current model's prediction* (so you can override the wrong ones).
3. You type one of:
   - `set volume to 40`                  -> assistant text reply
   - `tool system.volume.set level=40`   -> assistant emits that tool call
   - `exec` after a tool call            -> run the tool and feed the
                                            result back as a `tool` turn
   - `skip`                              -> discard the task
   - `more turns? user | tool | text`    -> extend the conversation
   - `done` / `quit`                     -> stop
4. The completed trace is appended to `--out` in the same JSONL schema
   as `assistant/data/sft_train.jsonl`. Drop the file into `data/` and
   rerun `sft_qwen.py` to fold it into the next training run.

## AI oracle (bulk)

If you don't want to type every example, pass `--oracle-ai` to use the
current model itself (or a stronger teacher) as the oracle:

```bash
python -m assistant.rlm_harness dataset --oracle-rule --limit 500
python -m assistant.rlm_harness dataset --oracle-ai   --limit 200     --out assistant/data/sft_rule_augmented.jsonl
```

The trace is identical to the human-oracle version, so you can mix them
freely.

## CLI reference

```text
rlm-harness dataset
  --source {seed|sft|stt|<path.jsonl>}
  --out    <path.jsonl>             (default: assistant/data/sft_extra.jsonl)
  --limit  N
  --categories [volume app browser ...]
  --difficulties [easy medium hard]
  --no-hint                         (skip the model-prediction peek)
  --oracle-ai  | --oracle-rule       (skip the human, bulk-generate)
  --model-dir  <path>  --adapter-dir <path>  --device {cuda|cpu}

rlm-harness play
  --backend {model|rule}            (default: model)
  --also-sft                        (also dump SFT-shaped records)

rlm-harness serve
  --backend {model|rule}            (default: model)
  --out <path>                      (default: rlm_harness/traces/serve.jsonl)

rlm-harness info                    (print task counts + tool schema)

rlm-harness merge --source <a.jsonl> [--destination <b.jsonl>]
                                   (default destination: sft_train.jsonl)
```

## Files

```
assistant/rlm_harness/
├── __init__.py
├── __main__.py            # CLI entry point
├── trace.py               # Trace / Turn data model (SFT-schema-compatible)
├── tasks.py               # task loader + curated seed
├── harness.py             # ModelBackend / RuleBackend / Harness
├── dataset_mode.py        # human + ai oracle collection loop
├── play_mode.py           # run-the-model eval loop
├── data/
│   └── tasks_seed.jsonl   # curated starter task set (auto-generated)
└── traces/                # play / serve traces land here
```

## Schema compatibility

`Trace.to_sft_record()` returns:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "set volume to 30"},
    {"role": "assistant", "content": "",
     "tool_calls": [{"type": "function",
                     "function": {"name": "system.volume.set",
                                  "arguments": "{\"level\": 30}"}}]},
    {"role": "tool", "name": "system.volume.set", "content": "OK: volume 30%"}
  ],
  "tools": [<the 15-tool schema>]
}
```

That is exactly the shape `make_dataset.py` writes and `sft_qwen.py`
trains on. No glue code needed.
