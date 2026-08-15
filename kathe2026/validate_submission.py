"""
Pre-flight check before uploading to Kaggle.

    python validate_submission.py submission.csv data/englishdev.csv
"""

import sys

import pandas as pd

sub_path = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
src_path = sys.argv[2] if len(sys.argv) > 2 else "data/englishdev.csv"

sub = pd.read_csv(sub_path)
src = pd.read_csv(src_path)

ok = True


def check(cond, msg):
    global ok
    print(("PASS  " if cond else "FAIL  ") + msg)
    ok = ok and cond


check(list(sub.columns) == ["ID", "kashmiri_text"],
      f"columns are exactly ID,kashmiri_text (got {list(sub.columns)})")
check(len(sub) == len(src), f"row count {len(sub)} == input {len(src)} (Kaggle wants 1730)")
check(sub["ID"].tolist() == src["ID"].tolist(), "IDs match input order exactly")
check(sub["ID"].is_unique, "IDs unique")
blank = int(sub["kashmiri_text"].isna().sum() + (sub["kashmiri_text"].astype(str).str.strip() == "").sum())
check(blank == 0, f"no empty translations (found {blank})")

text = "".join(str(x) for x in sub["kashmiri_text"].head(300))
arab = sum(
    0x0600 <= ord(c) <= 0x06FF or 0x0750 <= ord(c) <= 0x077F or
    0x08A0 <= ord(c) <= 0x08FF or 0xFB50 <= ord(c) <= 0xFDFF or 0xFE70 <= ord(c) <= 0xFEFF
    for c in text
)
deva = sum(0x0900 <= ord(c) <= 0x097F for c in text)
latin = sum(c.isascii() and c.isalpha() for c in text)
print(f"      script mix: perso-arabic={arab} devanagari={deva} latin={latin}")
check(max(arab, deva) > latin, "output is Kashmiri script, not leaked English")

dupes = len(sub) - sub["kashmiri_text"].nunique()
print(f"      duplicate translations: {dupes}")

print("\nREADY TO SUBMIT" if ok else "\nDO NOT SUBMIT — fix the FAILs above")
raise SystemExit(0 if ok else 1)
