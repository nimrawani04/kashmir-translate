# 🚀 FINE-TUNE NOW - Quick Commands

## ⚡ Step 1: Set HuggingFace Token

### Windows CMD:
```cmd
set HF_TOKEN=hf_your_token_here
```

### Windows PowerShell:
```powershell
$env:HF_TOKEN="hf_your_token_here"
```

**Get token from**: https://huggingface.co/settings/tokens

---

## 🎯 Step 2: Choose Your Strategy

### Strategy 1: STANDARD (20-22 points) ✅ BALANCED
**Time**: 3-4 hours  
**Best for**: Good quality with reasonable time

```cmd
python finetune_improved.py --epochs 5 --use-bpcc --bpcc-samples 5000 --batch-size 2 --grad-accum 8 --lora-r 32 --lora-alpha 64 --lr 3e-4 --output-dir out/lora-kas-standard
```

---

### Strategy 2: AGGRESSIVE (22-24 points) ⭐ RECOMMENDED
**Time**: 5-6 hours  
**Best for**: Maximum quality

```cmd
python finetune_improved.py --epochs 10 --use-bpcc --bpcc-samples 10000 --batch-size 2 --grad-accum 8 --lora-r 64 --lora-alpha 128 --lr 2e-4 --warmup-ratio 0.15 --output-dir out/lora-kas-aggressive
```

---

### Strategy 3: EXTREME (24+ points) 🔥 MAXIMUM
**Time**: 8-10 hours  
**Best for**: Absolute maximum score

```cmd
python finetune_improved.py --epochs 15 --use-bpcc --bpcc-samples 15000 --batch-size 1 --grad-accum 16 --lora-r 128 --lora-alpha 256 --lr 1e-4 --warmup-ratio 0.2 --output-dir out/lora-kas-extreme
```

---

### Strategy 4: QUICK TEST (Testing only) 🧪
**Time**: 30 minutes  
**Best for**: Testing setup before full run

```cmd
python finetune_improved.py --epochs 1 --use-bpcc --bpcc-samples 1000 --batch-size 2 --grad-accum 4 --lora-r 16 --lora-alpha 32 --output-dir out/lora-kas-quick
```

---

## 🔄 Step 3: Generate Translations

After training completes, run inference:

### For STANDARD:
```cmd
python inference_finetuned.py --model-dir out/lora-kas-standard --batch-size 4 --num-beams 5
```

### For AGGRESSIVE:
```cmd
python inference_finetuned.py --model-dir out/lora-kas-aggressive --batch-size 4 --num-beams 5
```

### For EXTREME:
```cmd
python inference_finetuned.py --model-dir out/lora-kas-extreme --batch-size 4 --num-beams 5
```

---

## ✅ Step 4: Validate & Compare

```cmd
# Validate submission format
python validate_submission.py

# Compare with manual translations
python compare_translations.py
```

---

## 📤 Step 5: Submit to Kaggle

Upload `submission_finetuned.csv` to:
https://www.kaggle.com/competitions/kathe-2026/submissions

---

## 🎯 ONE-LINER COMMANDS

### Quick Start (Recommended - Aggressive):
```cmd
set HF_TOKEN=your_token && python finetune_improved.py --epochs 10 --use-bpcc --bpcc-samples 10000 --batch-size 2 --grad-accum 8 --lora-r 64 --lora-alpha 128 --lr 2e-4 --warmup-ratio 0.15 --output-dir out/lora-kas-aggressive && python inference_finetuned.py --model-dir out/lora-kas-aggressive --batch-size 4 --num-beams 5
```

### PowerShell Version:
```powershell
$env:HF_TOKEN="your_token"; python finetune_improved.py --epochs 10 --use-bpcc --bpcc-samples 10000 --batch-size 2 --grad-accum 8 --lora-r 64 --lora-alpha 128 --lr 2e-4 --warmup-ratio 0.15 --output-dir out/lora-kas-aggressive; python inference_finetuned.py --model-dir out/lora-kas-aggressive --batch-size 4 --num-beams 5
```

---

## 📊 Expected Results

| Strategy | Time | Epochs | BPCC | LoRA Rank | Expected Score |
|----------|------|--------|------|-----------|----------------|
| Standard | 3-4h | 5 | 5K | 32 | 20-22 points |
| **Aggressive** ⭐ | **5-6h** | **10** | **10K** | **64** | **22-24 points** |
| Extreme | 8-10h | 15 | 15K | 128 | 24+ points |
| Quick Test | 30m | 1 | 1K | 16 | Testing only |

---

## 🔧 Troubleshooting

### CUDA Out of Memory
```cmd
# Reduce batch size
python finetune_improved.py --batch-size 1 --grad-accum 16 ...other args...
```

### Training Too Slow
```cmd
# Remove BPCC augmentation
python finetune_improved.py --epochs 10 --batch-size 2 --lora-r 64 --output-dir out/lora-kas-manual-only
```

### Want Even Better Quality
```cmd
# Increase LoRA capacity
python finetune_improved.py --lora-r 256 --lora-alpha 512 ...other args...
```

---

## 💡 Pro Tips

1. **Start with AGGRESSIVE** (22-24 points) - best quality/time balance
2. **Monitor training loss** - should decrease steadily
3. **Target loss < 1.0** after 5 epochs for good quality
4. **Use beam search** with 5+ beams during inference
5. **Compare outputs** before submitting

---

## 📝 Training Monitoring

Watch for these in the output:

### Good Training:
```
Epoch 1/10: loss=2.34
Epoch 2/10: loss=1.82
Epoch 3/10: loss=1.45
Epoch 5/10: loss=1.04
Epoch 10/10: loss=0.68
✅ Loss decreasing steadily!
```

### Warning Signs:
```
Epoch 8/10: loss=1.52
Epoch 9/10: loss=1.61
Epoch 10/10: loss=1.73
⚠️ Loss increasing - may be overfitting!
```

---

## 🚀 RECOMMENDED: Start Training NOW

**Best command for 22-24 points** (5-6 hours):

```cmd
REM 1. Set token
set HF_TOKEN=hf_your_token_here

REM 2. Train (aggressive strategy)
python finetune_improved.py ^
    --epochs 10 ^
    --use-bpcc ^
    --bpcc-samples 10000 ^
    --batch-size 2 ^
    --grad-accum 8 ^
    --lora-r 64 ^
    --lora-alpha 128 ^
    --lr 2e-4 ^
    --warmup-ratio 0.15 ^
    --output-dir out/lora-kas-aggressive

REM 3. Generate translations
python inference_finetuned.py ^
    --model-dir out/lora-kas-aggressive ^
    --batch-size 4 ^
    --num-beams 5

REM 4. Validate
python validate_submission.py

REM 5. Submit to Kaggle!
```

**OR use automated script:**
```cmd
set HF_TOKEN=your_token
finetune_commands.bat
```
Then choose option [2] for AGGRESSIVE training.

---

## ⏰ Timeline

```
T+0:00  → Start training
T+0:30  → Epoch 1 complete (loss ~2.0)
T+1:00  → Epoch 2 complete (loss ~1.6)
T+1:30  → Epoch 3 complete (loss ~1.3)
T+2:30  → Epoch 5 complete (loss ~1.0)
T+5:00  → Epoch 10 complete (loss ~0.7)
T+5:30  → Inference complete
T+5:35  → Validation passed
T+5:40  → Ready to submit!

Expected score: 22-24 points 🎉
```

---

## 🎯 Success Criteria

### After Training:
- ✅ Final training loss < 1.0
- ✅ Validation loss < 1.2
- ✅ No CUDA errors
- ✅ Model saved successfully

### After Inference:
- ✅ 1,730/1,730 translations generated
- ✅ No empty translations
- ✅ Validation passed
- ✅ Comparison shows improvement

### After Submission:
- ✅ Score > 20 (baseline exceeded)
- ✅ Score > 22 (target achieved!) 🎉
- ✅ Rank improved

---

**Ready to start? Copy the command for AGGRESSIVE strategy and run it now!**
