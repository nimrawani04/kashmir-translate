"""
Translate IDs 2-136 using the existing inference.py script approach
This will replace the low-quality baseline translations with better ones
"""
import pandas as pd
import subprocess
import os

# Read the English sentences for IDs 2-136
print("Reading englishdev.csv...")
source_df = pd.read_csv('data/englishdev.csv')
batch_df = source_df[(source_df['ID'] >= 2) & (source_df['ID'] <= 136)].copy()

print(f"Found {len(batch_df)} sentences to translate (IDs 2-136)")

# Create a temporary input file
temp_input = 'temp_batch_2_136_input.csv'
batch_df.to_csv(temp_input, index=False)

print(f"\nCreated temporary input file: {temp_input}")
print(f"Running translation using inference.py...")
print("This may take a few minutes...")

# Run the inference script
try:
    result = subprocess.run(
        ['python', 'inference.py'],
        capture_output=True,
        text=True,
        timeout=600  # 10 minute timeout
    )
    
    if result.returncode == 0:
        print("✅ Translation completed successfully!")
        print(result.stdout)
    else:
        print("❌ Translation failed:")
        print(result.stderr)
except subprocess.TimeoutExpired:
    print("⚠️ Translation timed out after 10 minutes")
except Exception as e:
    print(f"❌ Error running inference: {e}")

# Check if output was created
if os.path.exists('temp_batch_2_136_output.csv'):
    print("\n✅ Output file created: temp_batch_2_136_output.csv")
else:
    print("\n❌ Output file not found")
