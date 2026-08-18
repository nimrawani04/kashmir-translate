"""Shared scoring helpers for KATHE 2026 reranking + grid search.

Everything here is pure Python / sacrebleu — no models, so both rerank.py and
sweep.py can import it cheaply.
"""

from __future__ import annotations

import math
import unicodedata

try:
    import sacrebleu
except ImportError:  # sweep.py hard-requires it; rerank.py degrades gracefully
    sacrebleu = None  # type: ignore


# ---------------------------------------------------------------- text hygiene

def nfc(text: str) -> str:
    """Canonical NFC form. Perso-Arabic Kashmiri has multiple codepoint
    sequences that render identically; chrF++ counts them as different chars,
    so normalising is free points."""
    return unicodedata.normalize("NFC", str(text)).strip()


def is_kas_arab(text: str, threshold: float = 0.6) -> bool:
    chars = [c for c in text if not c.isspace() and c.isalpha()]
    if not chars:
        return False
    arab = sum(1 for c in chars if 0x0600 <= ord(c) <= 0x06FF or 0xFB50 <= ord(c) <= 0xFDFF)
    return arab / len(chars) >= threshold


# ---------------------------------------------------------------- score fusion

def zscore(values: list[float]) -> list[float]:
    n = len(values)
    if n <= 1:
        return [0.0] * n
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    sd = math.sqrt(var)
    if sd < 1e-9:
        return [0.0] * n
    return [(v - mean) / sd for v in values]


def mbr_scores(candidates: list[str]) -> list[float]:
    """Minimum Bayes Risk consensus: average chrF++ of each candidate against
    all the others. Picks the candidate most 'agreed on' by the N-best pool,
    which directly optimises the metric the competition scores on."""
    n = len(candidates)
    if n <= 1 or sacrebleu is None:
        return [0.0] * n
    chrf = sacrebleu.CHRF(word_order=2)
    out = []
    for i, hyp in enumerate(candidates):
        others = [c for j, c in enumerate(candidates) if j != i]
        out.append(sum(chrf.sentence_score(hyp, [o]).score for o in others) / len(others))
    return out


def combine(forward: list[float], reverse: list[float], mbr: list[float],
            alpha: float, mbr_weight: float) -> list[float]:
    """Blend the three normalised signals into one ranking score."""
    f, r, m = zscore(forward), zscore(reverse), zscore(mbr)
    return [(1 - alpha) * f[i] + alpha * r[i] + mbr_weight * m[i]
            for i in range(len(forward))]


def pick(candidates: list[str], forward: list[float], reverse: list[float],
         mbr: list[float], alpha: float, mbr_weight: float,
         enforce_script: bool = True) -> str:
    if not candidates:
        return ""
    total = combine(forward, reverse, mbr, alpha, mbr_weight)
    order = sorted(range(len(candidates)), key=lambda k: total[k], reverse=True)
    if enforce_script:
        for k in order:
            if is_kas_arab(candidates[k]):
                return candidates[k]
    return candidates[order[0]]


# ---------------------------------------------------------------- competition metric

def kathe_score(hyps: list[str], refs: list[str]) -> dict[str, float]:
    """Competition metric: geometric mean of BLEU and chrF++."""
    if sacrebleu is None:
        raise SystemExit("pip install sacrebleu to run scoring")
    hyps = [nfc(h) for h in hyps]
    refs = [nfc(r) for r in refs]
    bleu = sacrebleu.BLEU(tokenize="char").corpus_score(hyps, [refs]).score
    chrf = sacrebleu.CHRF(word_order=2).corpus_score(hyps, [refs]).score
    return {"bleu": bleu, "chrf++": chrf, "score": math.sqrt(max(bleu, 0.0) * max(chrf, 0.0))}
