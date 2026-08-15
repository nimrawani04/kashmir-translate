"""
Optional stretch goal: light LoRA fine-tuning of IndicTrans2 on the English-Kashmiri
portion of BPCC (https://huggingface.co/datasets/ai4bharat/BPCC).

The base checkpoint is already trained on full BPCC, so gains are expected to be small.
Skip this if the deadline is tight.

    export HF_TOKEN=hf_xxx
    python finetune.py --max-samples 50000 --epochs 1 --output-dir out/lora-kas
"""

from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    try:
        from IndicTransToolkit import IndicProcessor  # type: ignore
    except ImportError:
        class IndicProcessor:  # type: ignore
            """Pure Python fallback for IndicProcessor when IndicTransToolkit is not installed."""
            def __init__(self, inference: bool = True):
                self.inference = inference

            def preprocess_batch(self, batch: list[str], src_lang: str = "eng_Latn",
                                 tgt_lang: str = "kas_Arab") -> list[str]:
                return [f"{src_lang} {tgt_lang} {str(s).strip()}" for s in batch]

            def postprocess_batch(self, batch: list[str], lang: str = "kas_Arab") -> list[str]:
                cleaned = []
                for text in batch:
                    t = str(text).strip()
                    for tag in [lang, "eng_Latn", "kas_Arab", "kas_Deva"]:
                        if t.startswith(tag):
                            t = t[len(tag):].strip()
                    cleaned.append(t)
                return cleaned



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ai4bharat/indictrans2-en-indic-1B")
    p.add_argument("--dataset", default="ai4bharat/BPCC")
    p.add_argument("--config", default="bpcc-seed-latest",
                   help="BPCC config name; check the dataset card")
    p.add_argument("--pair", default="eng_Latn-kas_Arab")
    p.add_argument("--src-lang", default="eng_Latn")
    p.add_argument("--tgt-lang", default="kas_Arab")
    p.add_argument("--max-samples", type=int, default=50000)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--output-dir", default="out/lora-kas")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, trust_remote_code=True, token=token, torch_dtype=torch.float32
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        ),
    )
    model.print_trainable_parameters()

    ds = load_dataset(args.dataset, args.config, split="train", token=token,
                      streaming=False)
    # BPCC rows carry src/tgt text plus language identifiers; keep en->kas only.
    cols = ds.column_names
    if "src_lang" in cols and "tgt_lang" in cols:
        ds = ds.filter(lambda r: r["src_lang"] == args.src_lang
                       and r["tgt_lang"] == args.tgt_lang)
    if args.max_samples and len(ds) > args.max_samples:
        ds = ds.shuffle(seed=42).select(range(args.max_samples))

    src_col = "src" if "src" in cols else "source"
    tgt_col = "tgt" if "tgt" in cols else "target"
    ip = IndicProcessor(inference=False)

    def preprocess(batch):
        srcs = ip.preprocess_batch(batch[src_col], src_lang=args.src_lang,
                                   tgt_lang=args.tgt_lang)
        enc = tok(srcs, truncation=True, max_length=args.max_length)
        labels = tok(text_target=batch[tgt_col], truncation=True, max_length=args.max_length)
        enc["labels"] = labels["input_ids"]
        return enc

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=0.03,
            logging_steps=50,
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
            report_to=[],
        ),
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
