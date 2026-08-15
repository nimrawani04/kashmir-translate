import pandas as pd

# Read manual translations
df = pd.read_csv('manual_translations.csv')
print(f'Total manual translations: {len(df)}')
print(f'ID range: {df["ID"].min()} to {df["ID"].max()}')
print()

# Find missing IDs
all_ids = set(range(1, 1731))
present_ids = set(df['ID'])
missing_ids = sorted(all_ids - present_ids)

print(f'Total missing IDs: {len(missing_ids)}')
print()

# Group missing IDs into ranges for better readability
def group_ranges(ids):
    if not ids:
        return []
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
print('Missing ID ranges:')
for r in missing_ranges:
    print(f'  {r}')
print()

print(f'Total missing: {len(missing_ids)} IDs')
print()

# Show all missing IDs in chunks
print('Complete list of missing IDs:')
for i in range(0, len(missing_ids), 50):
    chunk = missing_ids[i:i+50]
    print(f'  {chunk}')
