# Cozy model benchmark summary

Trained on RTX 3050 6 GB, all numbers from `models/benchmarks/`.

| model | metric | v1.0 | v1.1 |
|---|---|---|---|
| hey_cozy | FPPH (1/hr) | 1.66 | 0.00 |
| hey_cozy | Recall | 0.69 | 0.96 |
| hey_cozy | AUT | 0.020 | 0.002 |
| cozy_stt | WER | 0.220 | 0.095 |
| cozy_stt | RTF | 0.026 | 0.025 |
| cozy-llm | tool-call acc | 0.842 | 0.632 |
| cozy-llm | chitchat acc | 0.917 | 0.917 |
| cozy-llm | overall acc | 0.871 | 0.742 |

## v1.2 STT (Flux hard-case round)

 whisper-small LoRA trained on 1793 clips (1380 cv_indian + 45 santhosh_indian
 + 368 synthetic Flux-TTs) and evaluated on a harder 217-clip set: the 125
 legacy clips plus a 92-clip lexical holdout (whole sentences unseen in
 training) of Flux-TTs audio covering complex words, post-2023 vocab,
 disfluencies, company names, acronyms and Indian names, in 4 voices
 (flux-priya/naveen/alexis/marcus). Details in `stt_eval_v1.2.json`,
 plot in `stt_wer_v1_vs_v1.2.png`. Legacy numbers above are on the old
 125-clip eval, so compare within this table only.

| model | metric (217-clip eval) | v1.0 base | v1.1 | v1.2 |
|---|---|---|---|---|
| cozy_stt | WER overall | 0.142 | 0.166 | 0.119 |
| cozy_stt | WER cv_indian (120) | 0.082 | 0.076 | 0.060 |
| cozy_stt | WER flux_tts holdout (92) | 0.194 | 0.242 | 0.168 |
| cozy_stt | WER santhosh (5) | 0.203 | 0.313 | 0.250 |
| cozy_stt | RTF | 0.018 | 0.013 | 0.013 |

 v1.1 regressed vs base on the hard holdout (0.242 vs 0.194) — it overfit the
 legacy distribution (and trained on Santhosh labels polluted with
 `file\tid\t` prefixes, fixed in v1.2's `prepare_data.py`). v1.2 is the
 best on overall, legacy cv_indian and the hard holdout.
