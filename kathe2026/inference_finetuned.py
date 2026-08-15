"""
Generate translations using the fine-tuned model for 20+ score.

Usage:
    python inference_finetuned.py --model-dir out/lora-kas-improved
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    try:
        from IndicTransToolkit import IndicProcessor
    except ImportError:
        class IndicProcessor:
            def __init__(self, inference: bool = True):
                self.inference = inference

            def preprocess_batch(self, batch: list[str], src_lang: str = "eng_Latn",
                                 tgt_lang: str = "kas_Arab") -> list[str]:
                return [f"{src_lang} {tgt_lang} {str(s).strip()}" for s in batch]

            def postprocess_batch(self, batch: list[str], lang: str = "kas_Arab") -> list[str]:
                cleaned = []
                for text in batch:
                    t = str(text).strip()
                    for tag in [lang, "eng_Latn", "kas_Arab", "kas_Deva"]:
                        if t.startswith(tag):
                            t = t[len(tag):].strip()
                    cleaned.append(t)
                return cleaned


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference with fine-tuned model")
    
    # Model paths
    p.add_argument("--base-model", default="ai4bharat/indictrans2-en-indic-1B",
                   help="Base IndicTrans2 model")
    p.add_argument("--model-dir", default="out/lora-kas-improved",
                   help="Fine-tuned LoRA adapter directory")
    
    # Data paths
    p.add_argument("--input", default="data/englishdev.csv",
                   help="Input CSV with English sentences")
    p.add_argument("--output", default="submission_finetuned.csv",
                   help="Output CSV with Kashmiri translations")
    
    # Generation settings
    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch size for inference")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--num-beams", type=int, default=5,
                   help="Number of beams for beam search (higher = better quality)")
    p.add_argument("--temperature", type=float, default=0.6,
                   help="Sampling temperature (lower = more conservative)")
    p.add_argument("--top-p", type=float, default=0.9,
                   help="Nucleus sampling threshold")
    p.add_argument("--repetition-penalty", type=float, default=1.2,
                   help="Penalty for repetition")
    
    # Language settings
    p.add_argument("--src-lang", default="eng_Latn")
    p.add_argument("--tgt-lang", default="kas_Arab")
    
    return p.parse_args()


def load_finetuned_model(base_model: str, adapter_dir: str, device: str):
    """Load base model with LoRA adapter."""
    print(f"📦 Loading base model: {base_model}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True
    )
    
    # Load base model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    
    # Load LoRA adapter
    print(f"🔧 Loading fine-tuned adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    
    return model, tokenizer


def translate_batch(
    sentences: list[str],
    model,
    tokenizer,
    ip: IndicProcessor,
    args,
    device: str
) -> list[str]:
    """Translate a batch of sentences."""
    
    # Preprocess
    preprocessed = ip.preprocess_batch(
        sentences,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang
    )
    
    # Tokenize
    inputs = tokenizer(
        preprocessed,
        truncation=True,
        max_length=args.max_length,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_length=args.max_length,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            early_stopping=True,
            use_cache=True,
            num_return_sequences=1
        )
    
    # Decode
    translations = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    # Postprocess
    translations = ip.postprocess_batch(translations, lang=args.tgt_lang)
    
    return translations


def main() -> int:
    args = parse_args()
    
    print("="*60)
    print("🚀 KATHE 2026 - Fine-tuned Model Inference")
    print("="*60)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model, tokenizer = load_finetuned_model(
        args.base_model,
        args.model_dir,
        device
    )
    
    # Initialize processor
    ip = IndicProcessor(inference=True)
    
    # Load input data
    print(f"\n📚 Loading input data: {args.input}")
    df = pd.read_csv(args.input)
    
    if 'sentence' not in df.columns:
        print("❌ Error: 'sentence' column not found in input CSV")
        return 1
    
    sentences = df['sentence'].astype(str).tolist()
    ids = df['ID'].tolist() if 'ID' in df.columns else list(range(1, len(sentences) + 1))
    
    print(f"   ✓ Loaded {len(sentences)} sentences")
    
    # Generate translations
    print(f"\n🔄 Generating translations...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Num beams: {args.num_beams}")
    print(f"   Temperature: {args.temperature}")
    
    translations = []
    
    for i in tqdm(range(0, len(sentences), args.batch_size), desc="Translating"):
        batch = sentences[i:i + args.batch_size]
        
        try:
            batch_translations = translate_batch(
                batch,
                model,
                tokenizer,
                ip,
                args,
                device
            )
            translations.extend(batch_translations)
            
        except Exception as e:
            print(f"\n⚠️ Error in batch {i//args.batch_size + 1}: {e}")
            # Fallback to individual sentences
            for sent in batch:
                try:
                    trans = translate_batch([sent], model, tokenizer, ip, args, device)
                    translations.extend(trans)
                except:
                    translations.append("")
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'ID': ids,
        'kashmiri_text': translations
    })
    
    # Save
    print(f"\n💾 Saving translations to: {args.output}")
    output_df.to_csv(args.output, index=False, encoding='utf-8')
    
    # Statistics
    print(f"\n📊 Translation Statistics:")
    print(f"   Total: {len(translations)}")
    empty = sum(1 for t in translations if not t.strip())
    print(f"   Empty: {empty}")
    print(f"   Completed: {len(translations) - empty}")
    
    if empty > 0:
        print(f"\n⚠️ Warning: {empty} translations are empty!")
    
    print("\n" + "="*60)
    print("✅ Inference complete!")
    print("="*60)
    print(f"\n📁 Output saved to: {args.output}")
    print(f"\n🎯 Next steps:")
    print(f"   1. Validate the output")
    print(f"   2. Compare with manual translations")
    print(f"   3. Submit to Kaggle")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
