"""
Quick check: is the Kashmiri text in sample_submission.csv Perso-Arabic or Devanagari?

    python check_script.py path/to/sample_submission.csv
"""

import sys

import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_submission.csv"
df = pd.read_csv(path)
col = "kashmiri_text" if "kashmiri_text" in df.columns else df.columns[-1]
text = "".join(str(x) for x in df[col].head(200).tolist())

arab = sum(0x0600 <= ord(c) <= 0x06FF or 0x0750 <= ord(c) <= 0x077F for c in text)
deva = sum(0x0900 <= ord(c) <= 0x097F for c in text)

print(f"column={col}  perso-arabic chars={arab}  devanagari chars={deva}")
if arab == deva == 0:
    print("-> no Kashmiri script found (sample may be blank/placeholder)")
else:
    print("-> use --tgt-lang " + ("kas_Arab" if arab >= deva else "kas_Deva"))
