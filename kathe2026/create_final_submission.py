import pandas as pd

# Read both CSV files
print("Reading submission_base.csv...")
base_df = pd.read_csv('submission_base.csv')
print(f"Loaded {len(base_df)} translations from submission_base.csv")

print("\nReading manual_translations.csv...")
manual_df = pd.read_csv('manual_translations.csv')
print(f"Loaded {len(manual_df)} translations from manual_translations.csv")

# Read the source file to get the complete list of IDs
print("\nReading data/englishdev.csv...")
source_df = pd.read_csv('data/englishdev.csv')
print(f"Total sentences in source: {len(source_df)}")

# Create a dictionary for quick lookup
translations = {}

# Add translations from submission_base.csv
for _, row in base_df.iterrows():
    translations[row['ID']] = row['kashmiri_text']

# Add/override with translations from manual_translations.csv
for _, row in manual_df.iterrows():
    translations[row['ID']] = row['kashmiri_text']

# Create final submission dataframe in exact order
print("\nCreating final submission.csv...")
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
            'kashmiri_text': ''  # Empty for missing translations
        })

# Create final dataframe
final_df = pd.DataFrame(final_data)

# Save to submission.csv
final_df.to_csv('submission.csv', index=False, encoding='utf-8')

print(f"\n{'='*60}")
print("FINAL SUBMISSION SUMMARY")
print(f"{'='*60}")
print(f"Total IDs in submission.csv: {len(final_df)}")
print(f"Translations completed: {len([x for x in final_df['kashmiri_text'] if x != ''])}")
print(f"Missing translations: {len(missing_ids)}")

if missing_ids:
    print(f"\nMissing ID ranges:")
    # Group missing IDs into ranges
    def group_ranges(ids):
        if not ids:
            return []
        ids = sorted(ids)
        ranges = []
        start = ids[0]
        end = ids[0]
        
        for i in range(1, len(ids)):
            if ids[i] == end + 1:
                end = ids[i]
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f'{start}-{end}')
                start = ids[i]
                end = ids[i]
        
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f'{start}-{end}')
        
        return ranges
    
    missing_ranges = group_ranges(missing_ids)
    for r in missing_ranges:
        print(f"  {r}")
else:
    print("\n✅ ALL TRANSLATIONS COMPLETE!")

print(f"\n{'='*60}")
print("✅ submission.csv created successfully!")
print(f"{'='*60}")

# Validate the submission
print("\nValidating submission.csv...")
print(f"Column names: {list(final_df.columns)}")
print(f"First 5 rows:")
print(final_df.head())
print(f"\nLast 5 rows:")
print(final_df.tail())
