"""
KATHE 2026 — English -> Kashmiri machine translation inference.

Loads a publicly available pretrained IndicTrans2 checkpoint, translates every
sentence in englishdev.csv and writes submission.csv with exactly two columns:
ID, kashmiri_text (same IDs, same order as the input file).

No third-party translation API or LLM service is used at any point.

Usage:
    export HF_TOKEN=hf_xxx          # never hardcode the token
    python inference.py \
        --input data/englishdev.csv \
        --output submission.csv \
        --model ai4bharat/indictrans2-en-indic-1B \
        --tgt-lang kas_Arab \
        --batch-size 16
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:  # older toolkit layout
    from IndicTransToolkit import IndicProcessor  # type: ignore


DEFAULT_MODEL = "ai4bharat/indictrans2-en-indic-1B"
FALLBACK_MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IndicTrans2 en->kas inference")
    p.add_argument("--input", default="data/englishdev.csv")
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"HF model id (fallback: {FALLBACK_MODEL})")
    p.add_argument("--src-lang", default="eng_Latn")
    p.add_argument("--tgt-lang", default="kas_Arab",
                   choices=["kas_Arab", "kas_Deva"],
                   help="Match the script used in sample_submission.csv")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--device", default=None, help="cuda | cpu (auto by default)")
    p.add_argument("--fp16", action="store_true", help="half precision on GPU")
    p.add_argument("--limit", type=int, default=0, help="smoke-test N rows only")
    return p.parse_args()


def load_model(model_id: str, token: str | None, device: str, fp16: bool):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=token,
        torch_dtype=torch.float16 if (fp16 and device == "cuda") else torch.float32,
    )
    model.to(device).eval()
    return tok, model


def translate_batch(batch, tok, model, ip, args, device) -> list[str]:
    prepped = ip.preprocess_batch(batch, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    enc = tok(prepped, truncation=True, padding längste := True, max_length=256,
              return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            num_beams=args.num_beams,
            num_return_sequences=1,
            max_new_tokens=args.max_new_tokens,
            early_stopping=True,
        )
    decoded = tok.batch_decode(out, skip_special_tokens=True,
                               clean_up_tokenization_spaces=True)
    return ip.postprocess_batch(decoded, lang=args.tgt_lang)


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("WARNING: no HF_TOKEN in env. Gated repos will fail to download.",
              file=sys.stderr)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} model={args.model} tgt_lang={args.tgt_lang}")

    df = pd.read_csv(args.input)
    if "ID" not in df.columns or "sentence" not in df.columns:
        raise SystemExit("input CSV must have columns: ID, sentence")
    if args.limit:
        df = df.head(args.limit).copy()

    sentences = [str(s) if pd.notna(s) else "" for s in df["sentence"].tolist()]

    try:
        tok, model = load_model(args.model, token, device, args.fp16)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not load {args.model} ({exc}); falling back to {FALLBACK_MODEL}",
              file=sys.stderr)
        tok, model = load_model(FALLBACK_MODEL, token, device, args.fp16)

    ip = IndicProcessor(inference=True)

    translations: list[str] = []
    for i in tqdm(range(0, len(sentences), args.batch_size), desc="translating"):
        batch = sentences[i:i + args.batch_size]
        try:
            translations.extend(translate_batch(batch, tok, model, ip, args, device))
        except Exception as exc:  # noqa: BLE001 — never lose alignment
            print(f"batch {i} failed ({exc}); retrying one by one", file=sys.stderr)
            for s in batch:
                try:
                    translations.extend(
                        translate_batch([s], tok, model, ip, args, device))
                except Exception:  # noqa: BLE001
                    translations.append("")

    assert len(translations) == len(df), "row count mismatch — submission misaligned"

    out = pd.DataFrame({"ID": df["ID"].values, "kashmiri_text": translations})
    out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"wrote {args.output} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
