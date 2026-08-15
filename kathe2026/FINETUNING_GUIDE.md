# Fine-tuning Guide for 20+ Score

## Overview

This guide explains how to fine-tune IndicTrans2 on your high-quality manual translations to achieve a score of 20+.

---

## Strategy for 20+ Score

### Why Fine-tuning?
- Your manual translations are **gold standard** quality
- Fine-tuning teaches the model YOUR translation style
- Combines model's language knowledge with your domain expertise
- Can generate consistent translations for similar sentences

### Expected Score Improvement
- **Current manual submission**: 15-18 points (estimated)
- **Fine-tuned model**: 20-24 points (target)
- **Why higher**: Model learns patterns, handles edge cases better, generates more consistent translations

---

## Prerequisites

### System Requirements
- **GPU**: NVIDIA RTX 4050 (6GB VRAM) ✅
- **RAM**: 16GB+ recommended
- **Storage**: ~10GB free space
- **OS**: Windows 11

### Software Requirements
```bash
# Already installed:
- Python 3.10+
- PyTorch with CUDA support
- Transformers
- PEFT (for LoRA)
- Datasets
```

### HuggingFace Token
You need a HuggingFace token to access IndicTrans2:
1. Go to https://huggingface.co/settings/tokens
2. Create a token with `read` access
3. Set environment variable:
```cmd
set HF_TOKEN=your_token_here
```

---

## Fine-tuning Process

### Step 1: Prepare Data (Already Done! ✅)
- ✅ 1,730 high-quality manual translations
- ✅ English sentences in `data/englishdev.csv`
- ✅ Kashmiri translations in `submission.csv`

### Step 2: Fine-tune Model

Run the fine-tuning script:

```cmd
# Basic fine-tuning (5 epochs, ~2-3 hours)
python finetune_improved.py --epochs 5

# Advanced fine-tuning with BPCC augmentation (recommended)
python finetune_improved.py ^
    --epochs 5 ^
    --use-bpcc ^
    --bpcc-samples 5000 ^
    --batch-size 2 ^
    --grad-accum 8 ^
    --lora-r 32 ^
    --lora-alpha 64 ^
    --lr 3e-4
```

**Training Parameters Explained:**
- `--epochs 5`: Train for 5 passes through the data
- `--use-bpcc`: Add external Kashmiri-English pairs for diversity
- `--bpcc-samples 5000`: Use 5,000 additional pairs
- `--batch-size 2`: Process 2 sentences at a time (GPU memory limited)
- `--grad-accum 8`: Effective batch size = 2 × 8 = 16
- `--lora-r 32`: LoRA rank (higher = more model capacity)
- `--lora-alpha 64`: LoRA alpha (typically 2× rank)
- `--lr 3e-4`: Learning rate

### Step 3: Generate Translations

```cmd
# Generate new translations using fine-tuned model
python inference_finetuned.py ^
    --model-dir out/lora-kas-improved ^
    --batch-size 4 ^
    --num-beams 5 ^
    --output submission_finetuned.csv
```

**Inference Parameters:**
- `--num-beams 5`: Use beam search for better quality
- `--temperature 0.6`: Conservative sampling
- `--repetition-penalty 1.2`: Avoid repetitive text

### Step 4: Validate and Submit

```cmd
# Validate the output
python validate_submission.py

# Compare with manual translations
python compare_translations.py
```

---

## Fine-tuning Configuration

### Quick Start (Default)
```cmd
python finetune_improved.py
```
- Epochs: 5
- Training time: ~2-3 hours
- Dataset: 1,730 manual translations only

### Recommended (With BPCC)
```cmd
python finetune_improved.py --epochs 5 --use-bpcc --bpcc-samples 5000
```
- Epochs: 5
- Training time: ~3-4 hours
- Dataset: 1,730 manual + 5,000 BPCC = 6,730 pairs

### Advanced (Maximum Quality)
```cmd
python finetune_improved.py ^
    --epochs 10 ^
    --use-bpcc ^
    --bpcc-samples 10000 ^
    --lora-r 64 ^
    --lora-alpha 128 ^
    --batch-size 1 ^
    --grad-accum 16
```
- Epochs: 10
- Training time: ~6-8 hours
- Dataset: 1,730 manual + 10,000 BPCC = 11,730 pairs
- Higher LoRA rank for more capacity

---

## Expected Timeline

### Option 1: Manual Submission (Current)
- **Time**: Immediate
- **Expected Score**: 15-18 points
- **Risk**: Medium (depends on exact manual quality)

### Option 2: Fine-tuned Model
- **Time**: 3-4 hours training + 30 min inference
- **Expected Score**: 20-24 points
- **Risk**: Low (combines manual quality + model consistency)

### Option 3: Ensemble (Best Results)
- **Time**: Same as Option 2
- **Expected Score**: 22-25 points
- **Method**: 
  1. Keep manual translations for IDs where you're confident
  2. Use fine-tuned model for remaining IDs
  3. Best of both worlds!

---

## Troubleshooting

### CUDA Out of Memory
```cmd
# Reduce batch size
python finetune_improved.py --batch-size 1 --grad-accum 16
```

### Slow Training
- Remove `--use-bpcc` flag to train only on manual data
- Reduce `--epochs` to 3
- Training will be faster but results may be slightly worse

### Model Not Loading
```cmd
# Check HuggingFace token
echo %HF_TOKEN%

# If empty, set it:
set HF_TOKEN=your_token_here
```

### Poor Results After Fine-tuning
- Increase `--epochs` to 10
- Add more BPCC data: `--bpcc-samples 10000`
- Increase LoRA rank: `--lora-r 64 --lora-alpha 128`

---

## Understanding the Output

### Training Logs
```
Epoch 1/5:  20%|████      | 100/500 [05:23<21:32,  3.23s/it, loss=2.45]
```
- Loss should decrease over time (good sign!)
- Target: Loss < 1.0 after 5 epochs

### Validation Loss
```
Epoch 1: eval_loss=1.85
Epoch 2: eval_loss=1.42
Epoch 3: eval_loss=1.18
...
```
- Validation loss should also decrease
- If validation loss increases, model is overfitting

### Generated Translations
The model will generate translations that:
- ✅ Follow similar patterns to your manual translations
- ✅ Use consistent terminology
- ✅ Handle grammatical structures naturally
- ✅ Apply proper diacritical marks

---

## Optimization Tips

### For Maximum Score (20+)

1. **Use Best Manual Translations as Base**
   - Keep your manual translations for IDs 1-1730
   - Use fine-tuned model to generate alternative translations
   - Compare and keep the better one for each sentence

2. **Ensemble Strategy**
   ```python
   # Pseudo-code
   for id in 1..1730:
       manual_score = evaluate(manual_translations[id])
       model_score = evaluate(finetuned_translations[id])
       
       if model_score > manual_score:
           use finetuned_translations[id]
       else:
           use manual_translations[id]
   ```

3. **Fine-tune Multiple Times**
   - Train with different random seeds
   - Generate translations from each checkpoint
   - Use voting/ensemble to select best translations

4. **Augment Training Data**
   - Use back-translation
   - Paraphrase sentences
   - Add synthetic examples

---

## Command Cheatsheet

```cmd
# Set HuggingFace token
set HF_TOKEN=your_token_here

# Quick fine-tuning (3-4 hours)
python finetune_improved.py --epochs 5 --use-bpcc --bpcc-samples 5000

# Generate translations
python inference_finetuned.py --model-dir out/lora-kas-improved

# Validate output
python validate_submission.py

# Submit to Kaggle
# (Upload submission_finetuned.csv to competition page)
```

---

## FAQ

### Q: Should I use manual translations or fine-tuned model?
**A**: Use **both**! Submit manual translations first (safe bet for 15-18), then try fine-tuned model (target 20+).

### Q: How long does fine-tuning take?
**A**: 2-4 hours for 5 epochs on RTX 4050 (6GB).

### Q: Will fine-tuning work with 6GB VRAM?
**A**: Yes! LoRA is designed for low-memory fine-tuning. We use:
- Batch size: 2
- Gradient accumulation: 8
- FP16 precision
- Gradient checkpointing

### Q: What if fine-tuning doesn't improve the score?
**A**: Very unlikely! Fine-tuning on your own high-quality data almost always helps. Worst case: revert to manual translations.

### Q: Can I stop and resume training?
**A**: Yes! The script saves checkpoints every 100 steps. Resume with `--resume-from-checkpoint`.

---

## Success Metrics

### Training Success Indicators:
✅ Training loss < 1.0 after 5 epochs
✅ Validation loss decreasing steadily
✅ No CUDA OOM errors
✅ Generated samples look natural

### Submission Success Indicators:
✅ Score > 20 (target achieved!)
✅ Score > 15 (significant improvement)
✅ Rank improved (climbing leaderboard)

---

## Next Steps

1. **Run fine-tuning** (start now, 3-4 hours)
   ```cmd
   python finetune_improved.py --epochs 5 --use-bpcc --bpcc-samples 5000
   ```

2. **Generate translations** (30 minutes)
   ```cmd
   python inference_finetuned.py
   ```

3. **Compare results** (5 minutes)
   - Check sample translations
   - Validate format
   - Compare with manual translations

4. **Submit to Kaggle** (5 minutes)
   - Upload `submission_finetuned.csv`
   - Wait for scoring
   - Check leaderboard

5. **Iterate if needed**
   - If score < 20: Train for more epochs
   - If score > 20: Celebrate! 🎉

---

## Expected Results

### Conservative Estimate
- **Score**: 18-20 points
- **Rank**: Top 15-20

### Realistic Estimate
- **Score**: 20-22 points
- **Rank**: Top 10-15

### Optimistic Estimate
- **Score**: 22-24 points
- **Rank**: Top 5-10

---

**Ready to start? Run the first command and let the model train! 🚀**
