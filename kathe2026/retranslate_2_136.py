"""
Retranslate IDs 2-136 with better quality settings
"""
import pandas as pd

# Create input CSV with just IDs 2-136
source_df = pd.read_csv('data/englishdev.csv')
batch_df = source_df[(source_df['ID'] >= 2) & (source_df['ID'] <= 136)].copy()

print(f"Creating input for {len(batch_df)} sentences (IDs 2-136)...")
batch_df.to_csv('batch_2_136_input.csv', index=False)
print("✅ Created: batch_2_136_input.csv")
print("\nNow run:")
print("python inference.py --input batch_2_136_input.csv --output batch_2_136_output.csv --batch-size 4 --num-beams 5 --fp16")
