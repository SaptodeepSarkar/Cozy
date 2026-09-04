#!/usr/bin/env python3
"""LoRA finetune whisper-small on Indian English (FLEURS en_in) + your voice.
Runs fully on-device (RTX 3050 6GB): fp16 + LoRA adapters only.

Run:  .venv/bin/python scripts/train_lora.py [--max-steps 20]   # smoke test
      .venv/bin/python scripts/train_lora.py                    # real run
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BASE_MODEL, CHECKPOINT_DIR, MANIFEST_DIR,
                    english_normalizer, read_manifest)  # noqa: E402


def build_datasets(max_audio_s=30.0):
    """Returns (train_ds, eval_ds, processor); every manifest row has audio_path."""
    import datasets as hfds
    import librosa
    import numpy as np
    import soundfile as sf
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="english",
                                                 task="transcribe")

    def featurize(rows):
        out = {"input_features": [], "labels": [], "source": []}
        skipped = 0
        for index, r in enumerate(rows, 1):
            try:
                audio, sr = sf.read(r["audio_path"], dtype="float32", always_2d=False)
                if getattr(audio, "ndim", 1) > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                if len(audio) > max_audio_s * 16000 or len(audio) < 1600:
                    skipped += 1
                    continue
                f = processor(audio, sampling_rate=16000).input_features[0]
                lab = processor.tokenizer(r["text"], truncation=True, max_length=224).input_ids
                out["input_features"].append(np.asarray(f, dtype=np.float32))
                out["labels"].append(lab)
                out["source"].append(r["source"])
            except (OSError, RuntimeError, ValueError) as exc:
                skipped += 1
                print(f"[data] skipped {r.get('audio_path')}: {exc}", flush=True)
            if index % 250 == 0:
                print(f"[data] {index}/{len(rows)} clips", flush=True)
        if skipped:
            print(f"[data] skipped {skipped}/{len(rows)} invalid/short clips", flush=True)
        return out

    def gen(name):
        rows = list(read_manifest(MANIFEST_DIR / name))
        d = featurize(rows)
        return hfds.Dataset.from_dict(d)

    train = gen("train.jsonl")
    evald = gen("eval.jsonl")
    return train, evald, processor


@dataclass
class SpeechCollator:
    """Official Whisper seq2seq collator (pads mel features; masks label pads)."""
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], Any]]]):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--out", default=str(CHECKPOINT_DIR / "lora_cozy_v1"))
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration)

    print("== Building datasets (mel features + token labels) ...")
    train_ds, eval_ds, processor = build_datasets()
    print(f"   train={len(train_ds)}  eval={len(eval_ds)}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for STT training")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.cuda.reset_peak_memory_stats()
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=dtype,
                                                            low_cpu_mem_usage=True)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # required with gradient checkpointing

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2,
                      target_modules=["q_proj", "v_proj"],
                      lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    norm = english_normalizer()
    eval_sources = eval_ds["source"]  # aligned with eval prediction order

    def compute_metrics(pred):
        tok = processor.tokenizer
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = tok.pad_token_id
        pred_str = tok.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tok.batch_decode(label_ids, skip_special_tokens=True)
        refs_n = [norm(l) for l in label_str]
        hyps_n = [norm(p) for p in pred_str]
        from common import wer as _wer
        metrics = {"wer": _wer(hyps_n, refs_n)}
        # per-source: guard against overfitting to YOUR voice at others' cost
        for name, sel in (("user", lambda s: s == "user"),
                          ("corpus", lambda s: s != "user")):
            r = [x for x, s in zip(refs_n, eval_sources) if sel(s)]
            h = [x for x, s in zip(hyps_n, eval_sources) if sel(s)]
            if r:
                metrics[f"wer_{name}"] = _wer(h, r)
        return metrics

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        tf32=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs=None if args.no_grad_ckpt else {"use_reentrant": False},
        eval_strategy="steps" if args.max_steps < 0 else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps" if args.max_steps < 0 else "no",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=args.max_steps < 0,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=224,
        logging_steps=1,
        remove_unused_columns=False,
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.workers > 0,
        dataloader_prefetch_factor=2 if args.workers > 0 else None,
        report_to=[],
        label_names=["labels"],
        seed=42,
    )

    import os
    parent = Seq2SeqTrainer

    if os.environ.get("COZY_DIAG"):
        class DiagTrainer(Seq2SeqTrainer):
            n = 0
            def compute_loss(self, model, inputs, return_outputs=False,
                             num_items_in_batch=None):
                out = super().compute_loss(model, inputs, return_outputs,
                                           num_items_in_batch=num_items_in_batch)
                if self.n < 3:
                    f, lab = inputs["input_features"], inputs["labels"]
                    lv = float(out[0].detach()) if return_outputs else float(out.detach())
                    with torch.no_grad():
                        ref = model(**inputs).loss.item()
                    print(f"[diag] micro={self.n} feat(std={f.float().std():.3f}) "
                          f"LOSS={lv:.3f} | same-batch-recompute={ref:.3f}", flush=True)
                    self.n += 1
                return out
        parent = DiagTrainer

    trainer = parent(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=SpeechCollator(processor),
        processing_class=processor,
        compute_metrics=compute_metrics,
    )
    started = time.perf_counter()
    train_result = trainer.train()
    metrics = trainer.evaluate(metric_key_prefix="final")
    print("FINAL EVAL:", metrics)
    model = trainer.model
    model.save_pretrained(str(Path(args.out) / "adapter"))
    processor.save_pretrained(str(Path(args.out) / "adapter"))
    report = {**train_result.metrics, **metrics, "wall_time_s": round(time.perf_counter() - started, 2),
              "dtype": str(dtype).removeprefix("torch."),
              "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
              "effective_batch_size": args.batch_size * args.grad_accum}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "training_metrics.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"LoRA adapters saved -> {args.out}/adapter")


if __name__ == "__main__":
    main()
