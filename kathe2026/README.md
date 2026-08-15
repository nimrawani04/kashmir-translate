# KATHE 2026 — English → Kashmiri Machine Translation

Submission pipeline for **KATHE 2026** (Kaggle), organized by Gaash Lab, NIT Srinagar
with the Bureau of Indian Standards. Task: translate English sentences into Kashmiri.

Licensed under the MIT License (see `LICENSE`) to satisfy the competition's
open-source requirement.

## Approach

- **Model:** [`ai4bharat/indictrans2-en-indic-1B`](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B),
  a publicly available pretrained multilingual NMT model already trained on the full
  BPCC corpus, which includes English↔Kashmiri. Fallback for limited GPU time:
  `ai4bharat/indictrans2-en-indic-dist-200M`.
- **Preprocessing:** `IndicTransToolkit`'s `IndicProcessor` handles the language-tag
  prefixing, normalization and entity placeholders IndicTrans2 expects, and the matching
  postprocessing (script conversion / detokenization) on the output.
- **Decoding:** beam search (`num_beams=5`), `max_new_tokens=256`, batched.
- **Languages:** `src_lang="eng_Latn"`, `tgt_lang="kas_Arab"` (Perso-Arabic Kashmiri).
  Verify against `sample_submission.csv` on the competition Data tab; if the reference
  text is Devanagari, rerun with `--tgt-lang kas_Deva`.
- **No third-party translation API or LLM service is used.** All translation happens
  locally with open weights, which is what the rules permit.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or CPU build
pip install -r requirements.txt
```

The IndicTrans2 checkpoints (and BPCC, if fine-tuning) are gated on Hugging Face:
accept the terms on each model/dataset page, then export a token:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

The token is read from the environment only — it is never stored in this repo.

## Run inference

Smoke test first (20 rows, seconds on GPU):

```bash
python inference.py --limit 20 --output /tmp/smoke.csv
```

Full run:

```bash
python inference.py \
  --input data/englishdev.csv \
  --output submission.csv \
  --model ai4bharat/indictrans2-en-indic-1B \
  --tgt-lang kas_Arab \
  --batch-size 16 --fp16
```

Runtime: roughly 5–15 min for 1,731 sentences on a single mid-range GPU with the 1B
model; the 200M distilled model is ~4× faster. CPU-only works but is slow (hours).

## Output format

`submission.csv` has exactly two columns and preserves input order:

```csv
ID,kashmiri_text
1,<kashmiri translation>
2,<kashmiri translation>
```

Row count and ID order are asserted against `data/englishdev.csv` before writing, and a
failing batch is retried sentence-by-sentence so alignment can never drift.

## Optional: LoRA fine-tuning

`finetune.py` does light LoRA fine-tuning on the English–Kashmiri portion of
[BPCC](https://huggingface.co/datasets/ai4bharat/BPCC). This is a stretch goal only —
the base checkpoint is already trained on BPCC, so the expected gain is small. Skip it
if the deadline is tight.

```bash
python finetune.py --max-samples 50000 --epochs 1 --output-dir out/lora-kas
python inference.py --model ai4bharat/indictrans2-en-indic-1B \
  --lora out/lora-kas --output submission_lora.csv
```

## Winning Setup: N-Best Candidate Ensemble & Round-Trip Reranking

For top leaderboard performance, `rerank.py` implements two legal, open-weight quality boosters over standard beam search:
1. **Candidate Pooling (Ensembling):** Generates an N-best list from one or more forward `en->kas` models (e.g. `indictrans2-en-indic-1B` + `indictrans2-en-indic-dist-200M`).
2. **Round-Trip Reranking:** Scores each candidate sentence with the reverse model (`ai4bharat/indictrans2-indic-en-1B`) to compute length-normalized log $P(\text{english\_source} \mid \text{kashmiri\_candidate})$. The candidate that best reconstructs the source sentence is selected, yielding typical gains of **+0.7 to +2.0 chrF++/BLEU**.

### Execution Steps for High Score

```bash
# 1. Smoke test (2 min)
python rerank.py --limit 20 --output /tmp/smoke.csv --nbest 4 --fp16

# 2. Strong single-model run (~25-40 min on one GPU)
python rerank.py --forward ai4bharat/indictrans2-en-indic-1B \
  --nbest 8 --num-beams 8 --alpha 0.5 --tgt-lang kas_Arab \
  --output submission.csv --fp16

# 3. Best run: two-model ensemble + reranking (~1-1.5 h)
python rerank.py \
  --forward ai4bharat/indictrans2-en-indic-1B ai4bharat/indictrans2-en-indic-dist-200M \
  --nbest 8 --num-beams 10 --alpha 0.5 --length-penalty 1.0 \
  --output submission_ens.csv --fp16

# 4. Pre-flight validation & Kaggle submission
python validate_submission.py submission_ens.csv data/englishdev.csv
kaggle competitions submit -c kathe-2026 -f submission_ens.csv -m "1B+200M ensemble, round-trip rerank"
```

### Ablation Ladder Strategy (3 Daily Submissions)
- **Submission A:** Baseline `inference.py` beam-5 run.
- **Submission B:** `rerank.py` single model (1B).
- **Submission C:** `rerank.py` ensemble (1B + 200M) + reverse reranking.

Tune `--alpha` (`0.3`, `0.5`, `0.7`) and `--length-penalty` (`0.8`, `1.0`, `1.2`) across submission budgets. If fine-tuning with `finetune.py`, pass `--lora out/lora-kas` to `rerank.py` to stack LoRA adapter weights on top of reranking.

## Repo layout

```
data/englishdev.csv    competition dev set (ID, sentence, Usage)
inference.py           translate -> submission.csv
rerank.py              N-best candidate ensemble + round-trip reranking
validate_submission.py pre-flight check (columns, row count, alignment, non-empty, script)
finetune.py            optional LoRA fine-tuning on BPCC en-kas
check_script.py        target script verifier (kas_Arab vs kas_Deva)
kathe2026_colab.ipynb  Google Colab / Kaggle GPU notebook
requirements.txt
LICENSE                MIT
```

## Verify the target script

```bash
python check_script.py path/to/sample_submission.csv
```

Prints whether the sample submission uses Perso-Arabic or Devanagari and the exact
`--tgt-lang` value to pass to `inference.py` and `rerank.py`.

