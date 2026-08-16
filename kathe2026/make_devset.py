"""Carve a held-out en->kas_Arab dev set out of BPCC so sweep.py has references.

    export HF_TOKEN=hf_xxx
    python make_devset.py --n 500 --out-dir data

Writes data/dev_src.csv (ID,sentence) and data/dev_ref.csv (ID,kashmiri_text).
Rows are deduped, length-filtered and NFC-normalised; anything not in
Perso-Arabic script is dropped. Keep --seed fixed so the dev set never leaks
into a later fine-tune (finetune.py can exclude these pairs by text match).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from rerank_utils import is_kas_arab, nfc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="data")
    p.add_argument("--config", default="bpcc-seed-latest")
    p.add_argument("--min-words", type=int, default=3)
    p.add_argument("--max-words", type=int, default=60)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ds = load_dataset("ai4bharat/BPCC", args.config, split="train")
    ds = ds.filter(lambda r: r.get("src_lang") == "eng_Latn"
                   and r.get("tgt_lang") == "kas_Arab")
    ds = ds.shuffle(seed=args.seed)

    seen, rows = set(), []
    for ex in ds:
        src, tgt = nfc(ex.get("src", "")), nfc(ex.get("tgt", ""))
        if not src or not tgt or (src, tgt) in seen:
            continue
        if not (args.min_words <= len(src.split()) <= args.max_words):
            continue
        if not is_kas_arab(tgt):
            continue
        seen.add((src, tgt))
        rows.append((len(rows) + 1, src, tgt))
        if len(rows) >= args.n:
            break

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["ID", "sentence", "kashmiri_text"])
    df[["ID", "sentence"]].to_csv(out / "dev_src.csv", index=False, encoding="utf-8")
    df[["ID", "kashmiri_text"]].to_csv(out / "dev_ref.csv", index=False, encoding="utf-8")
    print(f"wrote {len(df)} held-out pairs to {out}/dev_src.csv + dev_ref.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
