import pandas as pd

print("Creating improved submission.csv...")
print("="*60)

# Read the new translations for IDs 2-136
print("\n1. Reading improved translations (IDs 2-136)...")
improved_batch = pd.read_csv('batch_2_136_output.csv')
print(f"   Loaded {len(improved_batch)} improved translations")

# Read current submission (has manual translations for IDs 1, 137-1730)
print("\n2. Reading current submission (for manual translations)...")
current_submission = pd.read_csv('submission.csv')
print(f"   Loaded {len(current_submission)} existing translations")

# Read source data to ensure we have all IDs
print("\n3. Reading source data...")
source_df = pd.read_csv('data/englishdev.csv')
print(f"   Total sentences needed: {len(source_df)}")

# Create translation dictionary
translations = {}

# Add improved translations for IDs 2-136
for _, row in improved_batch.iterrows():
    translations[row['ID']] = row['kashmiri_text']

# Add current submission translations (will keep manual translations for IDs not in improved batch)
for _, row in current_submission.iterrows():
    # Only add if not already in translations (so we keep improved batch for IDs 2-136)
    if row['ID'] not in translations:
        translations[row['ID']] = row['kashmiri_text']

# Create final submission
print("\n4. Creating final submission...")
final_data = []
missing_ids = []

for id_val in source_df['ID']:
    if id_val in translations:
        final_data.append({
            'ID': id_val,
            'kashmiri_text': translations[id_val]
        })
    else:
        missing_ids.append(id_val)
        final_data.append({
            'ID': id_val,
            'kashmiri_text': ''
        })

final_df = pd.DataFrame(final_data)

# Save
final_df.to_csv('submission.csv', index=False, encoding='utf-8')

print("\n" + "="*60)
print("IMPROVED SUBMISSION SUMMARY")
print("="*60)
print(f"Total IDs: {len(final_df)}")
print(f"Translations completed: {len([x for x in final_df['kashmiri_text'] if x != ''])}")
print(f"Missing translations: {len(missing_ids)}")

if missing_ids:
    print(f"\nMissing IDs: {missing_ids}")
else:
    print("\n✅ ALL TRANSLATIONS COMPLETE!")

print("\nTranslation sources:")
print(f"  - IDs 2-136: Improved (re-translated with IndicTrans2)")
print(f"  - IDs 1, 137-1730: Manual translations")

# Show samples
print("\n" + "="*60)
print("SAMPLE IMPROVED TRANSLATIONS (IDs 2-10)")
print("="*60)
print(final_df[final_df['ID'].between(2, 10)][['ID', 'kashmiri_text']].to_string())

print("\n" + "="*60)
print("✅ submission.csv updated successfully!")
print("="*60)
