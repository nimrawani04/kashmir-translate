# Kashmiri Companion

I'm building a submission for KATHE 2026, a Kaggle machine-translation

competition (English → Kashmiri), organized by Gaash Lab, NIT Srinagar with

the Bureau of Indian Standards. Help me build and run the full pipeline.

HARD CONSTRAINTS (competition rules — don't violate these):

- May NOT use a third-party translation API/service (Google Translate, an LLM,

  etc.) to directly generate the submission. Disqualifying.

- MAY use a publicly available pretrained model (IndicTrans2, NLLB, mBART) as

  a starting point, and MAY fine-tune it on the BPCC corpus

  (https://huggingface.co/datasets/ai4bharat/BPCC).

- Submission must be a CSV with exactly two columns: ID, kashmiri_text — IDs

  must match englishdev.csv exactly, same order.

- Deadline is August 17, 2026 — prioritize a working end-to-end pipeline over

  experimentation.

- ALL participants (not just winners) must open-source their code under a

  permissive license (MIT/Apache-2.0) by the deadline to stay eligible.

APPROACH:

- Use ai4bharat/indictrans2-en-indic-1B via Hugging Face transformers +

  IndicTransToolkit (fall back to ai4bharat/indictrans2-en-indic-dist-200M if

  GPU time is limited). Both models and the BPCC dataset are gated on HF —

  I'll need to accept terms and pass an HF token.

- src_lang="eng_Latn", tgt_lang="kas_Arab" (Perso-Arabic Kashmiri script —

  double check this against sample_submission.csv on the competition's Data

  tab before running at scale; switch to "kas_Deva" only if that's what's

  used there).

- Batch-translate the ~1,731 sentences in englishdev.csv (columns: ID,

  sentence, Usage).

- Optional stretch goal only if time allows: light LoRA fine-tuning on the

  Kashmiri portion of BPCC using the scripts in AI4Bharat/IndicTrans2's

  huggingface_interface/ folder — skip if it risks the deadline, since the

  base model is already trained on full BPCC.

DELIVERABLES:

1. requirements.txt

2. inference.py — loads the model, translates englishdev.csv, writes submission.csv

3. rerank.py — candidate ensembling (1B + 200M) + reverse model round-trip reranking

4. validate_submission.py — pre-flight submission integrity check

5. check_script.py — checks sample submission for Perso-Arabic vs Devanagari script

6. kathe2026_colab.ipynb — end-to-end GPU workflow for Google Colab / Kaggle

7. README.md documenting the winning approach and open-source license

8. A LICENSE file (MIT)

9. (optional) finetune.py for LoRA fine-tuning on BPCC en-kas


Set this up as a small git repo I can push publicly to GitHub before the

deadline. Ask me for my Hugging Face token rather than hardcoding it.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/206acc05-2084-4839-88d1-2444c883b122).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
