"""KATHE 2026 — automatic grid search over --alpha, --length-penalty (and MBR
weight), then writes the best submission.csv.

Why this is fast: generation is the expensive part, reranking is not.
So the sweep

  1. generates an N-best pool ONCE per length-penalty value (that is the only
     knob that changes generation),
  2. computes the reverse round-trip score and the MBR/chrF++ consensus score
     for every candidate ONCE and caches them to disk (--cache),
  3. grid-searches alpha x mbr_weight purely on those cached numbers — that
     part costs milliseconds, so the grid can be dense,
  4. scores every configuration with the actual competition metric
     sqrt(BLEU x chrF++) on your labelled dev set,
  5. re-applies the winning config to the FULL englishdev.csv and writes
     submission.csv (ID, kashmiri_text, NFC-normalised, kas_Arab enforced).

Usage:

    export HF_TOKEN=hf_xxx
    python sweep.py \
        --dev-input data/dev_src.csv --dev-refs data/dev_ref.csv \
        --input data/englishdev.csv --output submission.csv \
        --forward ai4bharat/indictrans2-en-indic-1B \
                  ai4bharat/indictrans2-en-indic-dist-200M \
        --lp-grid 0.8 1.0 1.2 1.4 --alpha-grid 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 \
        --mbr-grid 0 0.2 0.4 0.6 --nbest 12 --fp16

dev CSVs: dev_src.csv = ID,sentence   dev_ref.csv = ID,kashmiri_text
(carve ~500 held-out pairs out of BPCC kas_Arab-eng_Latn, or use your own
manual translations — anything the model never trained on).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

import rerank as R
from rerank_utils import kathe_score, mbr_scores, nfc, pick


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="grid search alpha / length-penalty")
    p.add_argument("--dev-input", default="data/dev_src.csv")
    p.add_argument("--dev-refs", default="data/dev_ref.csv")
    p.add_argument("--input", default="data/englishdev.csv")
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--forward", nargs="+", default=[R.FWD_1B])
    p.add_argument("--reverse", default=R.REV_1B)
    p.add_argument("--lora", default=None)
    p.add_argument("--src-lang", default="eng_Latn")
    p.add_argument("--tgt-lang", default="kas_Arab", choices=["kas_Arab", "kas_Deva"])
    p.add_argument("--lp-grid", nargs="+", type=float, default=[0.8, 1.0, 1.2, 1.4])
    p.add_argument("--alpha-grid", nargs="+", type=float,
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    p.add_argument("--mbr-grid", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8])
    p.add_argument("--nbest", type=int, default=12)
    p.add_argument("--num-beams", type=int, default=12)
    p.add_argument("--no-repeat-ngram-size", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--score-batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--dev-limit", type=int, default=500, help="dev rows used for the sweep")
    p.add_argument("--cache", default="out/sweep_cache.json")
    p.add_argument("--results", default="out/sweep_results.json")
    p.add_argument("--skip-final", action="store_true",
                   help="only sweep; do not regenerate the full submission")
    return p.parse_args()


def build_pools(sentences: list[str], args, length_penalty: float, token, device):
    """N-best pools for every forward model at one length-penalty."""
    gen_args = argparse.Namespace(**vars(args))
    gen_args.length_penalty = length_penalty
    ip = R.IndicProcessor(inference=True)
    pools: list[list[tuple[str, float]]] = [[] for _ in sentences]

    for mid in args.forward:
        print(f"[forward] {mid} lp={length_penalty}")
        tok, model = R.load(mid, token, device, args.fp16,
                            args.lora if mid == args.forward[0] else None)
        for i in tqdm(range(0, len(sentences), args.batch_size), desc=f"gen lp={length_penalty}"):
            batch = sentences[i:i + args.batch_size]
            try:
                got = R.gen_nbest(batch, tok, model, ip, gen_args, device)
            except Exception:  # noqa: BLE001 — never lose alignment
                got = []
                for s in batch:
                    try:
                        got.extend(R.gen_nbest([s], tok, model, ip, gen_args, device))
                    except Exception:  # noqa: BLE001
                        got.append([("", -99.0)])
            for j, cands in enumerate(got):
                pools[i + j].extend((nfc(t), sc) for t, sc in cands)
        del model
        _free(device)
    return pools


def _free(device: str) -> None:
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def dedupe(pool: list[tuple[str, float]]) -> tuple[list[str], list[float]]:
    best: dict[str, float] = {}
    for text, sc in pool:
        if text and (text not in best or sc > best[text]):
            best[text] = sc
    return list(best.keys()), list(best.values())


def annotate(sentences, pools, args, token, device):
    """Attach reverse round-trip and MBR scores to each candidate pool."""
    ip = R.IndicProcessor(inference=True)
    rows = []
    rtok = rmodel = None
    if args.reverse:
        rtok, rmodel = R.load(args.reverse, token, device, args.fp16)
    for src, pool in tqdm(list(zip(sentences, pools)), desc="score"):
        cands, fwd = dedupe(pool)
        if not cands:
            rows.append({"cands": [], "fwd": [], "rev": [], "mbr": []})
            continue
        rev = [0.0] * len(cands)
        if rmodel is not None:
            try:
                rev = R.reverse_score([(c, src) for c in cands], rtok, rmodel, ip, args, device)
            except Exception as exc:  # noqa: BLE001
                print(f"reverse score failed ({exc}); forward-only for this row", file=sys.stderr)
        rows.append({"cands": cands, "fwd": fwd, "rev": rev, "mbr": mbr_scores(cands)})
    if rmodel is not None:
        del rmodel
        _free(device)
    return rows


def apply_config(rows, alpha: float, mbr_weight: float) -> list[str]:
    return [pick(r["cands"], r["fwd"], r["rev"], r["mbr"], alpha, mbr_weight)
            for r in rows]


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)

    dev = pd.read_csv(args.dev_input)
    refs_df = pd.read_csv(args.dev_refs)
    dev = dev.merge(refs_df, on="ID", how="inner")
    if args.dev_limit:
        dev = dev.head(args.dev_limit).copy()
    dev_src = [str(s) for s in dev["sentence"].tolist()]
    dev_ref = [nfc(s) for s in dev["kashmiri_text"].tolist()]
    print(f"dev set: {len(dev_src)} pairs")

    cache: dict[str, list] = {}
    if Path(args.cache).exists():
        cache = json.loads(Path(args.cache).read_text(encoding="utf-8"))
        print(f"loaded cache with lp values: {sorted(cache)}")

    for lp in args.lp_grid:
        key = f"{lp:g}"
        if key in cache:
            continue
        pools = build_pools(dev_src, args, lp, token, device)
        cache[key] = annotate(dev_src, pools, args, token, device)
        Path(args.cache).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    results = []
    for lp in args.lp_grid:
        rows = cache[f"{lp:g}"]
        for alpha in args.alpha_grid:
            for mw in args.mbr_grid:
                hyps = apply_config(rows, alpha, mw)
                met = kathe_score(hyps, dev_ref)
                results.append({"length_penalty": lp, "alpha": alpha,
                                "mbr_weight": mw, **met})
                print(f"lp={lp:<4} alpha={alpha:<4} mbr={mw:<4} "
                      f"BLEU={met['bleu']:.2f} chrF++={met['chrf++']:.2f} "
                      f"SCORE={met['score']:.2f}")

    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0]
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== BEST CONFIG ===")
    print(json.dumps(best, indent=2))
    print("top 5:")
    for r in results[:5]:
        print(f"  lp={r['length_penalty']} alpha={r['alpha']} mbr={r['mbr_weight']} "
              f"-> {r['score']:.2f}")

    if args.skip_final:
        return 0

    print("\n[final] regenerating full submission with the best config")
    full = pd.read_csv(args.input)
    sents = [str(s) if pd.notna(s) else "" for s in full["sentence"].tolist()]
    pools = build_pools(sents, args, best["length_penalty"], token, device)
    rows = annotate(sents, pools, args, token, device)
    finals = apply_config(rows, best["alpha"], best["mbr_weight"])
    assert len(finals) == len(full), "row count mismatch — submission misaligned"
    out = pd.DataFrame({"ID": full["ID"].values, "kashmiri_text": [nfc(t) for t in finals]})
    out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"wrote {args.output} ({len(out)} rows) with {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
