#!/usr/bin/env python3
"""Diagnostic: run the REAL Trainer for a few steps while inspecting every
batch/loss to find why Trainer-reported CE (~30) differs from manual forward
(~3.9). Prints signature of compute_loss, per-step stats, then exits.

Run: .venv/bin/python scripts/diag_train.py
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_MODEL, CHECKPOINT_DIR  # noqa: E402


def main():
    import torch
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
    from train_lora import SpeechCollator, build_datasets

    print("compute_loss signature:", inspect.signature(Seq2SeqTrainer.compute_loss))

    train_ds, eval_ds, processor = build_datasets()
    model = WhisperForConditionalGeneration = None  # placeholder name guard
    from transformers import WhisperForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    m = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    m.config.forced_decoder_ids = None
    m.config.suppress_tokens = []
    m.config.use_cache = False
    pm = get_peft_model(m, LoraConfig(r=32, lora_alpha=64,
                       target_modules=["q_proj", "v_proj"],
                       lora_dropout=0.05, bias="none"))

    class DiagTrainer(Seq2SeqTrainer):
        n = 0
        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            out = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                       num_items_in_batch=num_items_in_batch)
            if self.n < 4:
                f = inputs["input_features"]
                lab = inputs["labels"]
                loss_val = float(out[0].detach()) if return_outputs else float(out.detach())
                print(f"[diag] step={self.n} feat(mean={f.float().mean():.3f}, "
                      f"std={f.float().std():.3f}, shape={tuple(f.shape)}, "
                      f"dtype={f.dtype}) labels(shape={tuple(lab.shape)}, "
                      f"pad%={(lab == -100).float().mean():.2f}) LOSS={loss_val:.3f}",
                      flush=True)
                if self.n == 0:
                    # independent recomputation on the SAME batch, no autocast
                    with torch.no_grad():
                        ref = model(**{k: v for k, v in inputs.items()}).loss.item()
                    print(f"[diag] same-batch fp32 no-autocast loss={ref:.3f}", flush=True)
                self.n += 1
            return out

    targs = Seq2SeqTrainingArguments(
        output_dir=str(CHECKPOINT_DIR / "diag"),
        max_steps=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        warmup_steps=0,
        bf16=True,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=1,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to=[],
        label_names=["labels"],
        seed=42,
    )
    tr = DiagTrainer(model=pm, args=targs, train_dataset=train_ds,
                     data_collator=SpeechCollator(processor),
                     processing_class=processor)
    tr.train()
    print("DIAG DONE")


if __name__ == "__main__":
    main()
