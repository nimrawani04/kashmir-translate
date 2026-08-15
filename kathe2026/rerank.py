"""
KATHE 2026 — top-score pipeline: N-best generation + round-trip reranking.

Two legal, open-weight quality boosters over plain beam search:

1. CANDIDATE POOLING — generate an N-best list from one or more forward
   en->kas models (1B and the distilled 200M see the data differently, so
   their unions cover more of the search space).
2. ROUND-TRIP RERANKING — score every candidate with the reverse model
   (indictrans2-indic-en-1B): log P(english_source | kashmiri_candidate),
   length-normalised, blended with the forward score. The candidate the
   reverse model can best reconstruct the source from is the most faithful
   one. This typically buys +0.7 to +2.0 chrF++/BLEU over greedy beam search.

Everything runs locally on public pretrained weights. No translation API,
no LLM service.

    export HF_TOKEN=hf_xxx
    python rerank.py --input data/englishdev.csv --output submission.csv \
        --tgt-lang kas_Arab --nbest 8 --fp16
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



FWD_1B = "ai4bharat/indictrans2-en-indic-1B"
FWD_200M = "ai4bharat/indictrans2-en-indic-dist-200M"
REV_1B = "ai4bharat/indictrans2-indic-en-1B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="N-best + round-trip reranked inference")
    p.add_argument("--input", default="data/englishdev.csv")
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--forward", nargs="+", default=[FWD_1B],
                   help=f"one or more en->indic models (add {FWD_200M} to ensemble)")
    p.add_argument("--reverse", default=REV_1B, help="indic->en scorer; '' disables")
    p.add_argument("--lora", default=None, help="LoRA adapter for the FIRST forward model")
    p.add_argument("--src-lang", default="eng_Latn")
    p.add_argument("--tgt-lang", default="kas_Arab", choices=["kas_Arab", "kas_Deva"])
    p.add_argument("--nbest", type=int, default=8, help="candidates kept per model")
    p.add_argument("--num-beams", type=int, default=8)
    p.add_argument("--length-penalty", type=float, default=1.0)
    p.add_argument("--no-repeat-ngram-size", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="weight of the reverse (round-trip) score vs forward score")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--score-batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def load(model_id: str, token: str | None, device: str, fp16: bool, lora=None):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, trust_remote_code=True, token=token,
        torch_dtype=torch.float16 if (fp16 and device == "cuda") else torch.float32,
    )
    if lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora).merge_and_unload()
    return tok, model.to(device).eval()


def gen_nbest(batch, tok, model, ip, args, device):
    """Return list[list[(candidate, forward_logprob)]] aligned with `batch`."""
    prepped = ip.preprocess_batch(batch, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    enc = tok(prepped, truncation=True, padding=True, max_length=256,
              return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            num_beams=max(args.num_beams, args.nbest),
            num_return_sequences=args.nbest,
            max_new_tokens=args.max_new_tokens,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size or None,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
    seqs = tok.batch_decode(out.sequences, skip_special_tokens=True,
                            clean_up_tokenization_spaces=True)
    seqs = ip.postprocess_batch(seqs, lang=args.tgt_lang)
    scores = out.sequences_scores.tolist() if out.sequences_scores is not None \
        else [0.0] * len(seqs)
    grouped = []
    for i in range(len(batch)):
        chunk = list(zip(seqs[i * args.nbest:(i + 1) * args.nbest],
                         scores[i * args.nbest:(i + 1) * args.nbest]))
        grouped.append(chunk)
    return grouped


@torch.inference_mode()
def reverse_score(pairs, tok, model, ip, args, device):
    """pairs: list[(kashmiri_candidate, english_source)] -> length-normalised logprob."""
    out_scores = []
    for i in range(0, len(pairs), args.score_batch_size):
        chunk = pairs[i:i + args.score_batch_size]
        srcs = ip.preprocess_batch([c for c, _ in chunk],
                                   src_lang=args.tgt_lang, tgt_lang="eng_Latn")
        enc = tok(srcs, truncation=True, padding=True, max_length=256,
                  return_tensors="pt").to(device)
        lab = tok(text_target=[e for _, e in chunk], truncation=True, padding=True,
                  max_length=256, return_tensors="pt").to(device)
        labels = lab["input_ids"].masked_fill(lab["attention_mask"] == 0, -100)
        logits = model(**enc, labels=labels).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        mask = labels.ne(-100)
        gathered = logp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        tot = (gathered * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        out_scores.extend(tot.tolist())
    return out_scores


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("WARNING: no HF_TOKEN in env; gated repos will fail.", file=sys.stderr)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.input)
    if "ID" not in df.columns or "sentence" not in df.columns:
        raise SystemExit("input CSV must have columns: ID, sentence")
    if args.limit:
        df = df.head(args.limit).copy()
    sentences = [str(s) if pd.notna(s) else "" for s in df["sentence"].tolist()]

    ip = IndicProcessor(inference=True)
    pools: list[list[tuple[str, float]]] = [[] for _ in sentences]

    for mid in args.forward:
        print(f"[forward] {mid} on {device}")
        tok, model = load(mid, token, device, args.fp16,
                          args.lora if mid == args.forward[0] else None)
        for i in tqdm(range(0, len(sentences), args.batch_size), desc=f"gen {mid}"):
            batch = sentences[i:i + args.batch_size]
            try:
                got = gen_nbest(batch, tok, model, ip, args, device)
            except Exception as exc:  # noqa: BLE001 — never lose alignment
                print(f"batch {i} failed ({exc}); one-by-one", file=sys.stderr)
                got = []
                for s in batch:
                    try:
                        got.extend(gen_nbest([s], tok, model, ip, args, device))
                    except Exception:  # noqa: BLE001
                        got.append([("", -99.0)])
            for j, cands in enumerate(got):
                pools[i + j].extend(cands)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    finals: list[str] = []
    if args.reverse:
        print(f"[reverse] {args.reverse}")
        rtok, rmodel = load(args.reverse, token, device, args.fp16)
        for idx, (src, cands) in enumerate(
                tqdm(list(zip(sentences, pools)), desc="rerank")):
            uniq: dict[str, float] = {}
            for text, sc in cands:
                if text and (text not in uniq or sc > uniq[text]):
                    uniq[text] = sc
            if not uniq:
                finals.append("")
                continue
            items = list(uniq.items())
            try:
                rs = reverse_score([(t, src) for t, _ in items], rtok, rmodel, ip,
                                   args, device)
            except Exception as exc:  # noqa: BLE001
                print(f"score {idx} failed ({exc}); using forward best", file=sys.stderr)
                finals.append(max(items, key=lambda kv: kv[1])[0])
                continue
            best = max(range(len(items)),
                       key=lambda k: (1 - args.alpha) * items[k][1] + args.alpha * rs[k])
            finals.append(items[best][0])
    else:
        for cands in pools:
            finals.append(max(cands, key=lambda kv: kv[1])[0] if cands else "")

    assert len(finals) == len(df), "row count mismatch — submission misaligned"
    out = pd.DataFrame({"ID": df["ID"].values, "kashmiri_text": finals})
    out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"wrote {args.output} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
