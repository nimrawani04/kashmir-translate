import pandas as pd

print("Verifying submission.csv...")
df = pd.read_csv('submission.csv')

print(f"\n{'='*60}")
print("SUBMISSION FILE VERIFICATION")
print(f"{'='*60}")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Empty translations: {(df['kashmiri_text'] == '').sum()}")
print(f"Non-empty translations: {(df['kashmiri_text'] != '').sum()}")

# Check if IDs are in order
expected_ids = list(range(1, 1731))
actual_ids = df['ID'].tolist()
if expected_ids == actual_ids:
    print("\n✅ IDs are in correct order (1 to 1730)")
else:
    print("\n❌ ERROR: IDs are not in correct order!")

# Show sample rows
print(f"\n{'='*60}")
print("SAMPLE ROWS")
print(f"{'='*60}")
print("\nFirst 10 rows:")
print(df.head(10).to_string())

print("\n\nSample middle rows (IDs 138-143):")
print(df[(df['ID'] >= 138) & (df['ID'] <= 143)].to_string())

print("\n\nSample rows (IDs 500-505):")
print(df[(df['ID'] >= 500) & (df['ID'] <= 505)].to_string())

print("\n\nLast 5 rows:")
print(df.tail().to_string())

print(f"\n{'='*60}")
print("✅ VERIFICATION COMPLETE!")
print(f"{'='*60}")
