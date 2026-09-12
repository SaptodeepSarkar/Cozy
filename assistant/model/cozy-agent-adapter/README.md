# Cozy agent adapter

This directory receives the PEFT adapter from `train_agent_qlora.sh`. The
base model is deliberately not merged into it, so new tools only require a
new adapter run.

Recommended profile for the 6 GB target:

```bash
COZY_AGENT_BASE=Qwen/Qwen2.5-7B-Instruct bash train_agent_qlora.sh
```

This 7B-class model is the practical 9B-range choice for the machine: NF4
inference leaves room for Whisper and audio services, while a 9B fp16 model
does not. The adapter itself is only tens of MB.
