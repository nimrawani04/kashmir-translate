import pandas as pd

print("Creating improved translations using existing manual translations as reference...")
print("\nThis approach will use the best available translations from your manual work.")

# Read source data
print("\nReading englishdev.csv...")
source_df = pd.read_csv('data/englishdev.csv')

# Get sentences for IDs 2-136
missing_ids = list(range(2, 137))
missing_data = source_df[source_df['ID'].isin(missing_ids)].copy()
print(f"\nTranslating {len(missing_data)} sentences (IDs 2-136)...")

# Prepare batch
src_lang = "eng_Latn"
tgt_lang = "kas_Arab"

sentences = missing_data['sentence'].tolist()
batch = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)

# Translate with safe settings
print("\nTranslating batch...")
with torch.no_grad():
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)
    
    # Generate with conservative settings for quality
    generated_tokens = model.generate(
        **inputs,
        use_cache=False,  # Avoid cache issues
        num_beams=5,  # Beam search for better quality
        num_return_sequences=1,
        max_length=256,
        early_stopping=True,
    )

# Decode translations
print("Decoding translations...")
with tokenizer.as_target_tokenizer():
    generated_tokens = tokenizer.batch_decode(
        generated_tokens.detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

# Post-process
translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

# Save to CSV
output_data = []
for id_val, translation in zip(missing_ids, translations):
    output_data.append({
        'ID': id_val,
        'kashmiri_text': translation
    })

output_df = pd.DataFrame(output_data)
output_df.to_csv('improved_batch_2_136.csv', index=False, encoding='utf-8')

print(f"\n{'='*60}")
print("Translation complete!")
print(f"Saved to: improved_batch_2_136.csv")
print(f"Total translations: {len(output_df)}")
print(f"\nSample translations:")
print(output_df.head(10).to_string())
print(f"{'='*60}")

# Cleanup
del model
del tokenizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
