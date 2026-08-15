"""Translate in batches of 150 sentences and merge results."""
import pandas as pd
import sys
import subprocess
import time
import os

def translate_batch(start_idx, end_idx, input_file, output_file, model, tgt_lang):
    """Translate a batch of sentences."""
    # Read full input
    df = pd.read_csv(input_file)
    
    # Extract batch
    batch_df = df.iloc[start_idx:end_idx].copy()
    batch_input = f"batch_{start_idx}_{end_idx}_input.csv"
    batch_output = f"batch_{start_idx}_{end_idx}_output.csv"
    
    # Save batch input
    batch_df.to_csv(batch_input, index=False, encoding='utf-8')
    
    # Run translation
    print(f"Translating rows {start_idx+1} to {end_idx} ({len(batch_df)} sentences)...")
    
    # Set environment variable to limit GPU memory fragmentation
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    cmd = [
        "python", "inference.py",
        "--input", batch_input,
        "--output", batch_output,
        "--model", model,
        "--tgt-lang", tgt_lang,
        "--batch-size", "1",
        "--fp16"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    # Clean up batch input file
    if os.path.exists(batch_input):
        os.remove(batch_input)
    
    if result.returncode != 0:
        print(f"❌ Translation failed")
        if "out of memory" in result.stderr.lower():
            print("VRAM exhausted - waiting 10 seconds for GPU cleanup...")
            time.sleep(10)
        return None
    
    # Read batch output
    try:
        batch_result = pd.read_csv(batch_output)
        print(f"✅ Completed {len(batch_result)} translations")
        
        # Clean up batch output file after reading
        if os.path.exists(batch_output):
            os.remove(batch_output)
        
        return batch_result
    except Exception as e:
        print(f"Error reading output: {e}")
        return None

def main():
    input_file = "data/englishdev.csv"
    output_file = "submission.csv"
    model = "ai4bharat/indictrans2-en-indic-1B"
    tgt_lang = "kas_Arab"
    batch_size = 100  # Reduced to avoid VRAM issues
    
    # Read input
    df = pd.read_csv(input_file)
    total_rows = len(df)
    print(f"Total sentences to translate: {total_rows}")
    
    # Prepare output structure
    results = []
    
    # Process in batches
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        
        print(f"\n{'='*60}")
        print(f"BATCH {start_idx//batch_size + 1}: Rows {start_idx+1}-{end_idx}")
        print(f"{'='*60}")
        
        batch_result = translate_batch(start_idx, end_idx, input_file, output_file, model, tgt_lang)
        
        if batch_result is not None:
            results.append(batch_result)
            # Wait between batches to allow GPU memory cleanup
            if end_idx < total_rows:
                print("⏳ Waiting 5 seconds before next batch...")
                time.sleep(5)
        else:
            print(f"❌ Failed to translate batch {start_idx}-{end_idx}")
            print("⚠️ Saving partial results...")
            if results:
                partial_df = pd.concat(results, ignore_index=True)
                partial_output = f"partial_{output_file}"
                partial_df.to_csv(partial_output, index=False, encoding='utf-8')
                print(f"Saved {len(partial_df)} translations to {partial_output}")
            sys.exit(1)
    
    # Merge all results
    print(f"\n{'='*60}")
    print("MERGING ALL BATCHES...")
    print(f"{'='*60}")
    
    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n✅ SUCCESS!")
    print(f"Total rows translated: {len(final_df)}")
    print(f"Output saved to: {output_file}")
    
    # Validate
    empty = final_df['kashmiri_text'].isna().sum() + (final_df['kashmiri_text'] == '').sum()
    print(f"Empty translations: {empty}")
    print(f"Complete translations: {len(final_df) - empty}")

if __name__ == "__main__":
    main()
